from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BrandConfig:
    name: str
    handle: str
    app_summary: str
    launch_window_days: int
    goal: str
    audience: str
    voice_rules: List[str] = field(default_factory=list)
    cities: List[str] = field(default_factory=list)
    price_points_egp: List[int] = field(default_factory=list)
    content_pillars: List[str] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)
    banned_phrases: List[str] = field(default_factory=list)
    viral_edit_text_bank: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrandConfig":
        return cls(
            name=data["name"],
            handle=data["handle"],
            app_summary=data["app_summary"],
            launch_window_days=int(data["launch_window_days"]),
            goal=data["goal"],
            audience=data["audience"],
            voice_rules=list(data.get("voice_rules", [])),
            cities=list(data.get("cities", [])),
            price_points_egp=[int(value) for value in data.get("price_points_egp", [])],
            content_pillars=list(data.get("content_pillars", [])),
            pain_points=list(data.get("pain_points", [])),
            banned_phrases=list(data.get("banned_phrases", [])),
            viral_edit_text_bank=list(data.get("viral_edit_text_bank", [])),
        )


@dataclass
class SeriesConfig:
    enabled: bool = False
    title: str = ""
    logline: str = ""
    visual_mode: str = "live_action"
    animation_style: str = ""
    food_rendering: str = ""
    continuity_state_path: str = "runs/series/continuity.json"
    update_memory: bool = True
    protagonist: Dict[str, Any] = field(default_factory=dict)
    supporting_cast: List[Dict[str, Any]] = field(default_factory=list)
    location_bible: List[str] = field(default_factory=list)
    style_bible: List[str] = field(default_factory=list)
    act_structure: List[Dict[str, Any]] = field(default_factory=list)
    episode_guide: List[Dict[str, Any]] = field(default_factory=list)
    character_voices: Dict[str, str] = field(default_factory=dict)

    def get_episode(self, episode_number: int) -> Optional[Dict[str, Any]]:
        """Return the episode guide entry for a specific episode number, or None."""
        for ep in self.episode_guide:
            if int(ep.get("episode_number", 0)) == episode_number:
                return ep
        return None

    def get_episode_guide_summary(self) -> List[Dict[str, str]]:
        """Return a compact summary of all episodes (number, title, one-line beat) for prompt injection."""
        return [
            {
                "ep": ep["episode_number"],
                "act": ep.get("act", ""),
                "title_en": ep.get("title_en", ""),
                "title_ar": ep.get("title_ar", ""),
                "beat_summary": (ep.get("beat", "")[:120] + "…") if len(ep.get("beat", "")) > 120 else ep.get("beat", ""),
            }
            for ep in self.episode_guide
        ]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeriesConfig":
        return cls(
            enabled=bool(data.get("enabled", False)),
            title=str(data.get("title", "")),
            logline=str(data.get("logline", "")),
            visual_mode=str(data.get("visual_mode", "live_action")),
            animation_style=str(data.get("animation_style", "")),
            food_rendering=str(data.get("food_rendering", "")),
            continuity_state_path=str(data.get("continuity_state_path", "runs/series/continuity.json")),
            update_memory=bool(data.get("update_memory", True)),
            protagonist=dict(data.get("protagonist", {})),
            supporting_cast=list(data.get("supporting_cast", [])),
            location_bible=list(data.get("location_bible", [])),
            style_bible=list(data.get("style_bible", [])),
            act_structure=list(data.get("act_structure", [])),
            episode_guide=list(data.get("episode_guide", [])),
            character_voices=dict(data.get("character_voices", {})),
        )


@dataclass
class PipelineConfig:
    concept_pool_size: int = 8
    videos_per_run: int = 3
    script_iterations: int = 10
    video_prompt_iterations: int = 10
    research_temperature: float = 1.0
    writing_temperature: float = 0.9
    critic_temperature: float = 0.5
    use_google_search: bool = True
    vertex_ai_search_datastore: str = ""
    keep_top_revisions: int = 3
    run_root: str = "runs"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        return cls(
            concept_pool_size=int(data.get("concept_pool_size", 8)),
            videos_per_run=int(data.get("videos_per_run", 3)),
            script_iterations=int(data.get("script_iterations", 10)),
            video_prompt_iterations=int(data.get("video_prompt_iterations", 10)),
            research_temperature=float(data.get("research_temperature", 1.0)),
            writing_temperature=float(data.get("writing_temperature", 0.9)),
            critic_temperature=float(data.get("critic_temperature", 0.5)),
            use_google_search=bool(data.get("use_google_search", True)),
            vertex_ai_search_datastore=str(data.get("vertex_ai_search_datastore", "")),
            keep_top_revisions=int(data.get("keep_top_revisions", 3)),
            run_root=str(data.get("run_root", "runs")),
        )


@dataclass
class ModelConfig:
    research_model: str
    writing_model: str
    critic_model: str
    image_model: str
    video_model: str
    tts_model: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(
            research_model=data["research_model"],
            writing_model=data["writing_model"],
            critic_model=data["critic_model"],
            image_model=data["image_model"],
            video_model=data["video_model"],
            tts_model=data["tts_model"],
        )


@dataclass
class AssetConfig:
    generate_images: bool = False
    generate_videos: bool = False
    generate_audio: bool = False
    image_size: str = "2K"
    video_aspect_ratio: str = "9:16"
    video_duration_seconds: int = 8
    video_count_per_script: int = 3
    video_resolution: str = "720p"
    video_person_generation: str = "allow_adult"
    video_enhance_prompt: bool = False
    video_output_gcs_uri: str = ""
    tts_language_code: str = "ar-EG"
    tts_voice_name: str = "Kore"
    tts_file_name: str = "voiceover.wav"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssetConfig":
        return cls(
            generate_images=bool(data.get("generate_images", False)),
            generate_videos=bool(data.get("generate_videos", False)),
            generate_audio=bool(data.get("generate_audio", False)),
            image_size=str(data.get("image_size", "2K")),
            video_aspect_ratio=str(data.get("video_aspect_ratio", "9:16")),
            video_duration_seconds=int(data.get("video_duration_seconds", 8)),
            video_count_per_script=int(data.get("video_count_per_script", 3)),
            video_resolution=str(data.get("video_resolution", "720p")),
            video_person_generation=str(data.get("video_person_generation", "allow_adult")),
            video_enhance_prompt=bool(data.get("video_enhance_prompt", False)),
            video_output_gcs_uri=str(data.get("video_output_gcs_uri", "")),
            tts_language_code=str(data.get("tts_language_code", "ar-EG")),
            tts_voice_name=str(data.get("tts_voice_name", "Kore")),
            tts_file_name=str(data.get("tts_file_name", "voiceover.wav")),
        )


@dataclass
class AppConfig:
    brand: BrandConfig
    series: SeriesConfig
    pipeline: PipelineConfig
    models: ModelConfig
    assets: AssetConfig
    config_path: Path


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return AppConfig(
        brand=BrandConfig.from_dict(raw["brand"]),
        series=SeriesConfig.from_dict(raw.get("series", {})),
        pipeline=PipelineConfig.from_dict(raw.get("pipeline", {})),
        models=ModelConfig.from_dict(raw["models"]),
        assets=AssetConfig.from_dict(raw.get("assets", {})),
        config_path=config_path,
    )
