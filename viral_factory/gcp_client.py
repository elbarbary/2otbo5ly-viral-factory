from __future__ import annotations

import os
import random
import socket
import struct
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, TypeVar

_T = TypeVar("_T")

from .config import AppConfig
from .io_utils import ensure_dir, extract_json_blob

# Force IPv4-first for Google API endpoints — httpx tries IPv6 by default which
# causes TLS handshake timeouts on networks where IPv6 routes don't work.
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_first_getaddrinfo(*args, **kwargs):
    results = _orig_getaddrinfo(*args, **kwargs)
    return sorted(results, key=lambda r: r[0] == socket.AF_INET6)


socket.getaddrinfo = _ipv4_first_getaddrinfo


class VertexClientError(RuntimeError):
    pass


def _with_retry(fn: Callable[[], _T], *, max_attempts: int = 7, base_delay: float = 5.0) -> _T:
    """Call *fn* up to *max_attempts* times with exponential backoff + jitter.
    429 RESOURCE_EXHAUSTED and 500 INTERNAL errors get a longer fixed wait."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_attempts:
                raise
            exc_str = str(exc)
            if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                delay = 60.0 + random.uniform(0, 15)
                print(f"  [rate-limit retry {attempt}/{max_attempts - 1}] waiting {delay:.0f}s for quota reset…", flush=True)
            elif "500" in exc_str or "INTERNAL" in exc_str:
                delay = 30.0 + random.uniform(0, 10)
                print(f"  [server-error retry {attempt}/{max_attempts - 1}] waiting {delay:.0f}s…", flush=True)
            else:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
                print(f"  [retry {attempt}/{max_attempts - 1}] {exc!r} — retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def _pcm_to_wav(pcm: bytes, *, sample_rate: int, num_channels: int, sample_width: int) -> bytes:
    """Wrap raw PCM bytes in a minimal RIFF/WAV container."""
    num_frames = len(pcm) // (num_channels * sample_width)
    data_size = num_frames * num_channels * sample_width
    byte_rate = sample_rate * num_channels * sample_width
    block_align = num_channels * sample_width
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,   # overall file size - 8
        b"WAVE",
        b"fmt ",
        16,               # PCM sub-chunk size
        1,                # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        sample_width * 8, # bits per sample
        b"data",
        data_size,
    )
    return header + pcm[:data_size]


class VertexFactoryClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._genai = None
        self._genai_types = None
        self._text_client = None
        self._video_client = None
        self._tts_client = None
        self._client_lock = __import__("threading").Lock()

    def _load_genai(self) -> None:
        if self._genai is not None:
            return
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise VertexClientError(
                "Missing google-genai. Install dependencies with `pip install -r requirements.txt`."
            ) from exc
        self._genai = genai
        self._genai_types = types

    def _make_client(self, location: str, timeout_ms: int = 120_000):
        """Vertex AI client — used only for Veo video generation."""
        self._load_genai()
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise VertexClientError("Set GOOGLE_CLOUD_PROJECT before running live Vertex AI calls.")
        return self._genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=self._genai_types.HttpOptions(api_version="v1", timeout=timeout_ms)
        )

    def _make_api_key_client(self, timeout_ms: int = 120_000):
        """Gemini API key client — used for text, images, and TTS."""
        self._load_genai()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise VertexClientError("Set GEMINI_API_KEY before running Gemini API calls.")
        return self._genai.Client(
            api_key=api_key,
            http_options=self._genai_types.HttpOptions(timeout=timeout_ms)
        )

    @property
    def text_client(self):
        if self._text_client is None:
            with self._client_lock:
                if self._text_client is None:
                    self._text_client = self._make_api_key_client(timeout_ms=120_000)
        return self._text_client

    @property
    def video_client(self):
        if self._video_client is None:
            with self._client_lock:
                if self._video_client is None:
                    self._video_client = self._make_api_key_client(timeout_ms=1_800_000)
        return self._video_client

    def _response_to_dict(self, response: Any) -> Dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "to_json_dict"):
            return response.to_json_dict()
        if hasattr(response, "__dict__"):
            return dict(response.__dict__)
        return {"raw": str(response)}

    def _extract_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return text
        parts: List[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                value = getattr(part, "text", None)
                if value:
                    parts.append(value)
        if parts:
            return "\n".join(parts)
        raise VertexClientError("Vertex AI response did not include text.")

    _BINARY_KEYS = {"video_bytes", "image_bytes", "data", "raw_ref_image", "encoded_image"}

    def _strip_binary_fields(self, value: Any) -> Any:
        if isinstance(value, dict):
            cleaned: Dict[str, Any] = {}
            for key, item in value.items():
                if key in self._BINARY_KEYS:
                    continue
                # Skip any bytes values (e.g. base64-decoded blobs that got into response)
                if isinstance(item, (bytes, bytearray)):
                    continue
                cleaned[key] = self._strip_binary_fields(item)
            return cleaned
        if isinstance(value, list):
            return [self._strip_binary_fields(item) for item in value]
        if isinstance(value, (bytes, bytearray)):
            return None
        return value

    def _tools(self, use_google_search: bool, datastore: str) -> List[Any]:
        self._load_genai()
        tools: List[Any] = []
        if use_google_search:
            tools.append(self._genai_types.Tool(google_search=self._genai_types.GoogleSearch()))
        if datastore:
            tools.append(
                self._genai_types.Tool(
                    retrieval=self._genai_types.Retrieval(
                        vertex_ai_search=self._genai_types.VertexAISearch(datastore=datastore)
                    )
                )
            )
        return tools

    def _resolve_operation_result(self, operation: Any, *, poll_interval_seconds: int = 10) -> Any:
        current_operation = operation
        deadline = time.monotonic() + (30 * 60)
        transient_poll_failures = 0

        while getattr(current_operation, "done", None) is not True:
            if time.monotonic() > deadline:
                raise VertexClientError("Timed out waiting for Vertex AI video generation to complete.")
            if not getattr(current_operation, "name", None):
                raise VertexClientError("Vertex AI video generation operation did not include a pollable operation name.")
            try:
                current_operation = self.video_client.operations.get(current_operation)
                transient_poll_failures = 0
            except Exception as exc:
                transient_poll_failures += 1
                if transient_poll_failures >= 6 or time.monotonic() > deadline:
                    raise VertexClientError(
                        f"Vertex AI video generation polling failed repeatedly: {exc}"
                    ) from exc
                time.sleep(poll_interval_seconds)
                continue
            if getattr(current_operation, "done", None) is not True:
                time.sleep(poll_interval_seconds)

        error = getattr(current_operation, "error", None)
        if error:
            raise VertexClientError(f"Vertex AI video generation failed: {error}")

        result_attr = getattr(current_operation, "result", None)
        if callable(result_attr):
            result = result_attr()
        else:
            result = result_attr

        if result is None:
            response_attr = getattr(current_operation, "response", None)
            if callable(response_attr):
                result = response_attr()
            else:
                result = response_attr

        if result is None:
            raise VertexClientError("Vertex AI video generation completed without a result payload.")

        return result

    def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        use_google_search: bool,
        datastore: str = ""
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._load_genai()
        tools = self._tools(use_google_search=use_google_search, datastore=datastore)
        config_kwargs: Dict[str, Any] = {
            "temperature": temperature,
        }
        if tools:
            config_kwargs["tools"] = tools
        else:
            config_kwargs["response_mime_type"] = "application/json"
        config = self._genai_types.GenerateContentConfig(**config_kwargs)
        def _call():
            response = self.text_client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            text = self._extract_text(response)
            return extract_json_blob(text), self._response_to_dict(response)
        return _with_retry(_call)

    def generate_image(self, *, prompt: str, output_path: Path, reference_images: list = None) -> Dict[str, Any]:
        def _call():
            self._load_genai()
            types = self._genai_types
            contents = []
            if reference_images:
                for ref_path in reference_images:
                    ref_bytes = Path(ref_path).read_bytes()
                    suffix = Path(ref_path).suffix.lower()
                    mime = "image/png" if suffix == ".png" else "image/jpeg"
                    contents.append(types.Part.from_bytes(data=ref_bytes, mime_type=mime))
            contents.append(prompt)
            response = self.text_client.models.generate_content(
                model=self.config.models.image_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            for candidate in (response.candidates or []):
                for part in (getattr(candidate.content, "parts", None) or []):
                    blob = getattr(part, "inline_data", None)
                    if blob and getattr(blob, "mime_type", None) and blob.mime_type.startswith("image/"):
                        ensure_dir(output_path.parent)
                        output_path.write_bytes(blob.data)
                        return self._strip_binary_fields(self._response_to_dict(response))
            raise VertexClientError("Gemini 3 Pro Image returned no image in response.")
        return _with_retry(_call)

    def generate_video(
        self,
        *,
        prompt: str,
        output_dir: Path,
        shot_index: int,
        negative_prompt: str = "",
        seed: int | None = None,
        style_reference_image: Path | None = None,
        source_image: Path | None = None,
    ) -> Dict[str, Any]:
        self._load_genai()
        config_kwargs: Dict[str, Any] = {
            "aspect_ratio": self.config.assets.video_aspect_ratio,
            "duration_seconds": self.config.assets.video_duration_seconds,
            "number_of_videos": 1,
        }
        if negative_prompt:
            config_kwargs["negative_prompt"] = negative_prompt
        if seed is not None:
            config_kwargs["seed"] = seed
        if self.config.assets.video_output_gcs_uri:
            config_kwargs["output_gcs_uri"] = self.config.assets.video_output_gcs_uri
        video_config = self._genai_types.GenerateVideosConfig(**config_kwargs)
        request_kwargs: Dict[str, Any] = {
            "model": self.config.models.video_model,
            "prompt": prompt,
            "config": video_config,
        }
        if source_image is not None:
            request_kwargs["image"] = self._genai_types.Image.from_file(location=str(source_image))
        operation = self.video_client.models.generate_videos(**request_kwargs)
        result = self._resolve_operation_result(operation)
        payload = self._strip_binary_fields(self._response_to_dict(result))
        ensure_dir(output_dir)
        videos = getattr(result, "generated_videos", []) or []
        if not videos:
            print(f"  [video] WARNING: generated_videos is empty. result attrs: {[a for a in dir(result) if not a.startswith('_')]}", flush=True)
        if videos:
            video_obj = getattr(videos[0], "video", None)
            if video_obj is not None:
                uri = getattr(video_obj, "uri", None)
                if uri:
                    payload["gcs_uri"] = uri
                video_bytes = getattr(video_obj, "video_bytes", None)
                if not video_bytes and uri:
                    # Gemini API Veo returns a download URI instead of inline bytes.
                    # Try SDK download first, then fall back to authenticated HTTP.
                    try:
                        video_bytes = self.video_client.files.download(file=uri)
                        print(f"  [video] downloaded via SDK from {uri}", flush=True)
                    except Exception as sdk_exc:
                        print(f"  [video] SDK download failed ({sdk_exc!r}), trying HTTP…", flush=True)
                        api_key = os.getenv("GEMINI_API_KEY", "")
                        download_url = f"{uri}?key={api_key}&alt=media"
                        req = urllib.request.Request(download_url, headers={"User-Agent": "viral-factory/1.0"})
                        with urllib.request.urlopen(req, timeout=300) as resp:
                            video_bytes = resp.read()
                        print(f"  [video] downloaded via HTTP ({len(video_bytes)} bytes)", flush=True)
                if video_bytes:
                    video_path = output_dir / f"shot-{shot_index:02d}.mp4"
                    video_path.write_bytes(video_bytes)
                    payload["local_file"] = str(video_path)
        return payload

    def synthesize_speech(self, *, text: str, prompt: str, output_path: Path) -> Dict[str, Any]:
        """Synthesize Arabic speech using Gemini TTS (Egyptian accent)."""
        self._load_genai()
        types = self._genai_types
        model = self.config.models.tts_model or "gemini-2.5-pro-preview-tts"
        speech_cfg = types.SpeechConfig(
            language_code=self.config.assets.tts_language_code,
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=self.config.assets.tts_voice_name,
                )
            ),
        )
        def _call():
            response = self.text_client.models.generate_content(
                model=model,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=speech_cfg,
                ),
            )
            pcm: bytes = b""
            for c in (response.candidates or []):
                for p in (getattr(c.content, "parts", None) or []):
                    blob = getattr(p, "inline_data", None)
                    if blob and blob.data:
                        pcm += blob.data
            if not pcm:
                raise VertexClientError("Gemini TTS returned no audio.")
            return pcm

        pcm = _with_retry(_call, max_attempts=2)
        wav = _pcm_to_wav(pcm, sample_rate=24000, num_channels=1, sample_width=2)
        ensure_dir(output_path.parent)
        output_path.write_bytes(wav)
        return {"audio_file": str(output_path), "bytes": len(wav)}

    def synthesize_dialogue(
        self,
        *,
        dialogue: List[Dict[str, str]],
        character_voices: Dict[str, str],
        default_voice: str,
        language_code: str,
        output_path: Path,
    ) -> Dict[str, Any]:
        """Synthesize multi-character dialogue using Gemini TTS (Egyptian accent)."""
        self._load_genai()
        types = self._genai_types
        model = self.config.models.tts_model or "gemini-2.5-pro-preview-tts"
        all_pcm: bytes = b""
        for entry in dialogue:
            speaker = entry.get("speaker", "")
            text = entry.get("text", "").strip()
            if not text:
                continue
            voice_name = character_voices.get(speaker, default_voice)
            speech_cfg = types.SpeechConfig(
                language_code=language_code,
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                ),
            )
            def _call(t=text, sc=speech_cfg):
                resp = self.text_client.models.generate_content(
                    model=model,
                    contents=t,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=sc,
                    ),
                )
                seg: bytes = b""
                for c in (resp.candidates or []):
                    for p in (getattr(c.content, "parts", None) or []):
                        blob = getattr(p, "inline_data", None)
                        if blob and blob.data:
                            seg += blob.data
                if not seg:
                    raise VertexClientError(f"TTS returned no audio for '{speaker}'.")
                return seg
            all_pcm += _with_retry(_call, max_attempts=2)

        if not all_pcm:
            raise VertexClientError("Gemini TTS returned no audio for dialogue.")

        wav = _pcm_to_wav(all_pcm, sample_rate=24000, num_channels=1, sample_width=2)
        ensure_dir(output_path.parent)
        output_path.write_bytes(wav)
        return {"audio_file": str(output_path), "bytes": len(wav)}
