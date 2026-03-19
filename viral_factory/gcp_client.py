from __future__ import annotations

import os
import random
import socket
import struct
import time
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
        """timeout_ms: HttpOptions.timeout is in milliseconds per the SDK spec."""
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

    @property
    def text_client(self):
        if self._text_client is None:
            with self._client_lock:
                if self._text_client is None:
                    self._text_client = self._make_client(os.getenv("GOOGLE_CLOUD_LOCATION", "global"), timeout_ms=120_000)
        return self._text_client

    @property
    def video_client(self):
        if self._video_client is None:
            with self._client_lock:
                if self._video_client is None:
                    self._video_client = self._make_client(os.getenv("GOOGLE_CLOUD_VIDEO_LOCATION", "us-central1"), timeout_ms=1_800_000)
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

    def generate_image(self, *, prompt: str, output_path: Path) -> Dict[str, Any]:
        def _call():
            self._load_genai()
            response = self.text_client.models.generate_images(
                model=self.config.models.image_model,
                prompt=prompt,
                config=self._genai_types.GenerateImagesConfig(image_size=self.config.assets.image_size)
            )
            ensure_dir(output_path.parent)
            response.generated_images[0].image.save(str(output_path))
            return self._strip_binary_fields(self._response_to_dict(response))
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
        if style_reference_image is not None and source_image is not None:
            raise VertexClientError("Use either style_reference_image or source_image for video generation, not both.")
        config_kwargs: Dict[str, Any] = {
            "aspect_ratio": self.config.assets.video_aspect_ratio,
            "duration_seconds": self.config.assets.video_duration_seconds,
            "enhance_prompt": self.config.assets.video_enhance_prompt,
            "number_of_videos": 1,
            "person_generation": self.config.assets.video_person_generation,
            "resolution": self.config.assets.video_resolution
        }
        if negative_prompt:
            config_kwargs["negative_prompt"] = negative_prompt
        if seed is not None:
            config_kwargs["seed"] = seed
        if style_reference_image is not None:
            config_kwargs["reference_images"] = [
                self._genai_types.VideoGenerationReferenceImage(
                    image=self._genai_types.Image.from_file(location=str(style_reference_image)),
                    reference_type=self._genai_types.VideoGenerationReferenceType.STYLE,
                )
            ]
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
        if videos:
            video_obj = getattr(videos[0], "video", None)
            if video_obj is not None:
                uri = getattr(video_obj, "uri", None)
                if uri:
                    payload["gcs_uri"] = uri
                video_bytes = getattr(video_obj, "video_bytes", None)
                if video_bytes:
                    video_path = output_dir / f"shot-{shot_index:02d}.mp4"
                    video_path.write_bytes(video_bytes)
                    payload["local_file"] = str(video_path)
        return payload

    def synthesize_speech(self, *, text: str, prompt: str, output_path: Path) -> Dict[str, Any]:
        """Synthesize Arabic speech using Gemini TTS via the genai client.

        Audio is returned as raw PCM (16-bit LE, 24 kHz, mono) and written to
        *output_path* as a WAV file (with a minimal WAV header).
        """
        self._load_genai()
        types = self._genai_types
        model = self.config.models.tts_model or "gemini-2.5-flash-preview-tts"

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
            pcm_bytes: bytes = b""
            for candidate in (response.candidates or []):
                for part in (getattr(candidate.content, "parts", None) or []):
                    blob = getattr(part, "inline_data", None)
                    if blob and blob.data:
                        pcm_bytes += blob.data
            if not pcm_bytes:
                raise VertexClientError("Gemini TTS returned no audio data.")
            return pcm_bytes

        pcm_bytes = _with_retry(_call, max_attempts=2)
        wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate=24000, num_channels=1, sample_width=2)
        ensure_dir(output_path.parent)
        output_path.write_bytes(wav_bytes)
        return {"audio_file": str(output_path), "bytes": len(wav_bytes)}

    def synthesize_dialogue(
        self,
        *,
        dialogue: List[Dict[str, str]],
        character_voices: Dict[str, str],
        default_voice: str,
        language_code: str,
        output_path: Path,
    ) -> Dict[str, Any]:
        """Synthesize multi-character dialogue into a single WAV file.

        Each entry in *dialogue* is ``{"speaker": "...", "text": "..."}``.
        The voice for each speaker is looked up in *character_voices*; if not
        found, *default_voice* is used.  All segments are synthesized as raw
        PCM (24 kHz, 16-bit LE, mono) and concatenated before writing.
        """
        self._load_genai()
        types = self._genai_types
        model = self.config.models.tts_model or "gemini-2.5-flash-preview-tts"

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
                for candidate in (resp.candidates or []):
                    for part in (getattr(candidate.content, "parts", None) or []):
                        blob = getattr(part, "inline_data", None)
                        if blob and blob.data:
                            seg += blob.data
                if not seg:
                    raise VertexClientError(f"TTS returned no audio for speaker '{speaker}'.")
                return seg
            all_pcm += _with_retry(_call, max_attempts=2)

        if not all_pcm:
            raise VertexClientError("Gemini TTS returned no audio data for dialogue.")

        wav_bytes = _pcm_to_wav(all_pcm, sample_rate=24000, num_channels=1, sample_width=2)
        ensure_dir(output_path.parent)
        output_path.write_bytes(wav_bytes)
        return {"audio_file": str(output_path), "bytes": len(wav_bytes)}
