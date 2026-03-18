from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from viral_factory.config import load_config
from viral_factory.gcp_client import VertexFactoryClient


class FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTool:
    def __init__(self, google_search=None, retrieval=None):
        self.google_search = google_search
        self.retrieval = retrieval


class FakeGoogleSearch:
    pass


class FakeSpeechConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeVoiceConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakePrebuiltVoiceConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTypes:
    GenerateContentConfig = FakeGenerateContentConfig
    GenerateVideosConfig = FakeGenerateContentConfig
    Tool = FakeTool
    GoogleSearch = FakeGoogleSearch
    SpeechConfig = FakeSpeechConfig
    VoiceConfig = FakeVoiceConfig
    PrebuiltVoiceConfig = FakePrebuiltVoiceConfig
    HttpOptions = FakeGenerateContentConfig

    class VideoGenerationReferenceType:
        STYLE = "STYLE"

    class Image:
        @staticmethod
        def from_file(*, location):
            return {"location": location}

    class VideoGenerationReferenceImage:
        def __init__(self, *, image, reference_type):
            self.image = image
            self.reference_type = reference_type


class FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def model_dump(self):
        return {"text": self.text}


class FakeVideo:
    def __init__(self, uri=None, video_bytes=None):
        self.uri = uri
        self.video_bytes = video_bytes


class FakeGeneratedVideo:
    def __init__(self, video):
        self.video = video


class FakeVideoResult:
    def __init__(self, generated_videos):
        self.generated_videos = generated_videos

    def model_dump(self):
        dumped_videos = []
        for item in self.generated_videos:
            dumped_videos.append(
                {
                    "video": {
                        "uri": item.video.uri,
                        "video_bytes": item.video.video_bytes,
                    }
                }
            )
        return {"generated_videos": dumped_videos}


class FakeVideoOperation:
    def __init__(self, *, done, result=None, error=None, name="operations/test-video"): 
        self.done = done
        self.result = result
        self.error = error
        self.name = name


class FakeBlob:
    def __init__(self, data, mime_type="audio/pcm;rate=24000"):
        self.data = data
        self.mime_type = mime_type


class FakeAudioPart:
    def __init__(self, data):
        self.inline_data = FakeBlob(data)


class FakeAudioContent:
    def __init__(self, data):
        self.parts = [FakeAudioPart(data)]


class FakeAudioCandidate:
    def __init__(self, data):
        self.content = FakeAudioContent(data)


class FakeAudioResponse:
    def __init__(self, data):
        self.candidates = [FakeAudioCandidate(data)]


class FakeModels:
    def __init__(self):
        self.last_config = None
        self.video_operation = None
        self.last_video_request = None
        self.tts_audio_data: bytes = b''

    def generate_content(self, *, model, contents, config):
        self.last_config = config
        # If caller requested AUDIO modality, return fake audio response
        modalities = getattr(config, 'kwargs', {}).get('response_modalities', [])
        if 'AUDIO' in modalities:
            return FakeAudioResponse(self.tts_audio_data)
        return FakeResponse('{"ok": true}')

    def generate_videos(self, *, model, prompt, config, image=None):
        self.last_config = config
        self.last_video_request = {
            "model": model,
            "prompt": prompt,
            "config": config,
            "image": image,
        }
        return self.video_operation


class FakeOperations:
    def __init__(self, responses):
        self.responses = list(responses)

    def get(self, operation):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeTextClient:
    def __init__(self):
        self.models = FakeModels()
        self.operations = FakeOperations([])


class VertexClientConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(Path("configs/2otbo5ly.json"))

    def _make_client(self) -> VertexFactoryClient:
        client = VertexFactoryClient(self.config)
        client._genai = object()
        client._genai_types = FakeTypes
        client._text_client = FakeTextClient()
        client._video_client = client._text_client
        return client

    def test_generate_json_uses_json_mime_without_tools(self) -> None:
        client = self._make_client()

        payload, _ = client.generate_json(
            model="gemini-test",
            prompt="prompt",
            temperature=0.5,
            use_google_search=False,
        )

        config_kwargs = client.text_client.models.last_config.kwargs
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(config_kwargs["response_mime_type"], "application/json")
        self.assertNotIn("tools", config_kwargs)

    def test_generate_json_skips_json_mime_with_google_search(self) -> None:
        client = self._make_client()

        payload, _ = client.generate_json(
            model="gemini-test",
            prompt="prompt",
            temperature=0.5,
            use_google_search=True,
        )

        config_kwargs = client.text_client.models.last_config.kwargs
        self.assertEqual(payload, {"ok": True})
        self.assertIn("tools", config_kwargs)
        self.assertNotIn("response_mime_type", config_kwargs)

    def test_generate_video_polls_operation_result_property(self) -> None:
        client = self._make_client()
        result = FakeVideoResult([
            FakeGeneratedVideo(FakeVideo(uri="gs://bucket/shot-01.mp4", video_bytes=b"video-bytes"))
        ])
        client.text_client.models.video_operation = FakeVideoOperation(done=False)
        client.text_client.operations = FakeOperations([
            FakeVideoOperation(done=True, result=result)
        ])

        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = client.generate_video(
                prompt="prompt",
                output_dir=Path(tmp_dir),
                shot_index=1,
            )

            output_path = Path(tmp_dir) / "shot-01.mp4"
            self.assertTrue(output_path.exists())

        self.assertEqual(payload["gcs_uri"], "gs://bucket/shot-01.mp4")
        self.assertEqual(payload["local_file"], str(output_path))
        self.assertNotIn("video_bytes", str(payload))

    def test_generate_video_uses_response_when_result_missing(self) -> None:
        client = self._make_client()
        result = FakeVideoResult([
            FakeGeneratedVideo(FakeVideo(uri="gs://bucket/shot-01.mp4"))
        ])
        client.text_client.models.video_operation = FakeVideoOperation(done=True, result=None)
        client.text_client.models.video_operation.response = result

        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = client.generate_video(
                prompt="prompt",
                output_dir=Path(tmp_dir),
                shot_index=1,
            )

        self.assertEqual(payload["gcs_uri"], "gs://bucket/shot-01.mp4")

    def test_generate_video_passes_seed_and_negative_prompt(self) -> None:
        client = self._make_client()
        result = FakeVideoResult([
            FakeGeneratedVideo(FakeVideo(uri="gs://bucket/shot-01.mp4"))
        ])
        client.text_client.models.video_operation = FakeVideoOperation(done=True, result=result)

        with tempfile.TemporaryDirectory() as tmp_dir:
            client.generate_video(
                prompt="prompt",
                negative_prompt="avoid text",
                seed=1234,
                output_dir=Path(tmp_dir),
                shot_index=1,
            )

        config_kwargs = client.text_client.models.last_config.kwargs
        self.assertEqual(config_kwargs["negative_prompt"], "avoid text")
        self.assertEqual(config_kwargs["seed"], 1234)

    def test_generate_video_passes_style_reference_image(self) -> None:
        client = self._make_client()
        result = FakeVideoResult([
            FakeGeneratedVideo(FakeVideo(uri="gs://bucket/shot-01.mp4"))
        ])
        client.text_client.models.video_operation = FakeVideoOperation(done=True, result=result)

        with tempfile.TemporaryDirectory() as tmp_dir:
            style_path = Path(tmp_dir) / "style.png"
            style_path.write_bytes(b"style-image")
            client.generate_video(
                prompt="prompt",
                output_dir=Path(tmp_dir),
                shot_index=1,
                style_reference_image=style_path,
            )

        config_kwargs = client.text_client.models.last_config.kwargs
        self.assertIn("reference_images", config_kwargs)
        self.assertEqual(len(config_kwargs["reference_images"]), 1)
        self.assertEqual(config_kwargs["reference_images"][0].reference_type, "STYLE")
        self.assertEqual(
            config_kwargs["reference_images"][0].image,
            {"location": str(style_path)},
        )

    def test_generate_video_passes_source_image(self) -> None:
        client = self._make_client()
        result = FakeVideoResult([
            FakeGeneratedVideo(FakeVideo(uri="gs://bucket/shot-01.mp4"))
        ])
        client.text_client.models.video_operation = FakeVideoOperation(done=True, result=result)

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "frame.png"
            source_path.write_bytes(b"frame-image")
            client.generate_video(
                prompt="prompt",
                output_dir=Path(tmp_dir),
                shot_index=1,
                source_image=source_path,
            )

        self.assertEqual(
            client.text_client.models.last_video_request["image"],
            {"location": str(source_path)},
        )

    def test_generate_video_retries_transient_poll_error(self) -> None:
        client = self._make_client()
        result = FakeVideoResult([
            FakeGeneratedVideo(FakeVideo(uri="gs://bucket/shot-01.mp4"))
        ])
        client.text_client.models.video_operation = FakeVideoOperation(done=False)
        client.text_client.operations = FakeOperations([
            RuntimeError("temporary read error"),
            FakeVideoOperation(done=True, result=result),
        ])

        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = client.generate_video(
                prompt="prompt",
                output_dir=Path(tmp_dir),
                shot_index=1,
            )

        self.assertEqual(payload["gcs_uri"], "gs://bucket/shot-01.mp4")

    def test_synthesize_speech_uses_gemini_tts(self) -> None:
        # 10 ms of silence as raw 16-bit LE PCM @ 24 kHz (480 samples * 2 bytes)
        fake_pcm = b'\x00\x00' * 480
        client = self._make_client()
        client.text_client.models.tts_audio_data = fake_pcm

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "voiceover.wav"
            payload = client.synthesize_speech(
                text="مرحبا",
                prompt="say hello in Arabic",
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())
            wav_data = output_path.read_bytes()

        # WAV header is 44 bytes; payload size = 44 + PCM
        self.assertEqual(payload["audio_file"], str(output_path))
        self.assertGreater(payload["bytes"], len(fake_pcm))  # includes WAV header
        self.assertTrue(wav_data.startswith(b"RIFF"))
        self.assertIn(b"WAVE", wav_data[:12])
        # Verify the TTS model config was passed correctly
        last_cfg = client.text_client.models.last_config
        self.assertIn("AUDIO", last_cfg.kwargs.get("response_modalities", []))


if __name__ == "__main__":
    unittest.main()