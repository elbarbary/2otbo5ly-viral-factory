from __future__ import annotations

import json
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import AppConfig
from .gcp_client import VertexFactoryClient
from .io_utils import ensure_dir, slugify, utc_timestamp, write_json, write_text
from .prompts import (
    caption_prompt,
    concept_prompt,
    initial_script_prompt,
    initial_video_plan_prompt,
    research_prompt,
    script_iteration_prompt,
    video_iteration_prompt
)
from .scoring import score_script_heuristics


@dataclass
class IterationResult:
    version_index: int
    payload: Dict[str, Any]
    heuristic_score: Dict[str, Any]


class ViralFactoryPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = VertexFactoryClient(config)
        self._continuity_lock = threading.Lock()  # serialise continuity.json read-writes across threads

    def _repo_root(self) -> Path:
        return self.config.config_path.parent.parent

    def _run_root(self) -> Path:
        return ensure_dir(self._repo_root() / self.config.pipeline.run_root)

    def _build_run_dir(self, topic_hint: str) -> Path:
        return ensure_dir(self._run_root() / f"{utc_timestamp()}-{slugify(topic_hint)[:36]}")

    def _series_state_path(self) -> Path:
        raw_path = Path(self.config.series.continuity_state_path)
        if raw_path.is_absolute():
            return raw_path
        return self._repo_root() / raw_path

    def _load_series_state(self) -> Dict[str, Any]:
        if not self.config.series.enabled:
            return {"series_enabled": False, "episodes": [], "next_episode_number": 1}

        state_path = self._series_state_path()
        if state_path.exists():
            return json.loads(state_path.read_text(encoding="utf-8"))

        state = {
            "series_enabled": True,
            "title": self.config.series.title,
            "visual_mode": self.config.series.visual_mode,
            "next_episode_number": 1,
            "active_visual_locks": list(self.config.series.style_bible),
            "episodes": []
        }
        write_json(state_path, state)
        return state

    def _update_series_state_locked(self, packs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Thread-safe wrapper: re-reads current state, merges, writes, returns updated."""
        with self._continuity_lock:
            current = self._load_series_state()
            return self._update_series_state(current, packs)

    def _update_series_state(self, continuity_state: Dict[str, Any], packs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.config.series.enabled or not self.config.series.update_memory:
            return continuity_state

        next_episode_number = int(continuity_state.get("next_episode_number", 1))
        active_visual_locks = list(continuity_state.get("active_visual_locks", []))

        for pack in packs:
            concept = pack["concept"]
            final_script = pack["final_script"]
            final_video_plan = pack["final_video_plan"]

            hinted_episode = int(concept.get("episode_number_hint", 0) or 0)
            # Also check the script itself for episode_number
            script_episode = int(final_script.get("episode_number", 0) or 0)
            episode_number = hinted_episode or script_episode or next_episode_number

            continuity_state.setdefault("episodes", []).append(
                {
                    "episode_number": episode_number,
                    "name": concept.get("name", ""),
                    "summary": final_script.get("episode_summary", ""),
                    "cliffhanger": final_script.get("cliffhanger", ""),
                    "continuity_updates": final_script.get("continuity_updates", []),
                    "continuity_goal": concept.get("continuity_goal", ""),
                    "character_sheet_prompt_en": final_video_plan.get("character_sheet_prompt_en", ""),
                    "run_dir": pack["run_dir"]
                }
            )
            active_visual_locks.extend(final_video_plan.get("style_guardrails", []))
            next_episode_number = max(next_episode_number, episode_number + 1)

        continuity_state["next_episode_number"] = next_episode_number
        continuity_state["active_visual_locks"] = list(dict.fromkeys(active_visual_locks))
        write_json(self._series_state_path(), continuity_state)
        return continuity_state

    def _generate_json(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float,
        use_google_search: bool
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.client.generate_json(
            model=model,
            prompt=prompt,
            temperature=temperature,
            use_google_search=use_google_search,
            datastore=self.config.pipeline.vertex_ai_search_datastore
        )

    def _resolve_episode_beat(self, episode_number: Optional[int]) -> Optional[Dict[str, Any]]:
        """Look up the episode beat from the series episode guide."""
        if episode_number is None or not self.config.series.enabled:
            return None
        beat = self.config.series.get_episode(episode_number)
        if beat is None:
            raise ValueError(
                f"Episode {episode_number} not found in the episode guide. "
                f"Available episodes: 1–{len(self.config.series.episode_guide)}"
            )
        return beat

    def _build_topic_hint_for_episode(self, episode_beat: Dict[str, Any]) -> str:
        """Build a rich topic hint string from an episode beat for the series 'الشيف كريم'."""
        ep_num = episode_beat.get("episode_number", "?")
        title_en = episode_beat.get("title_en", "")
        title_ar = episode_beat.get("title_ar", "")
        act = episode_beat.get("act", "")
        beat = episode_beat.get("beat", "")
        tone = episode_beat.get("emotion_tone", "")
        return (
            f"الشيف كريم — Episode {ep_num} (Act {act}): \"{title_en}\" / \"{title_ar}\". "
            f"Emotional tone: {tone}. Story beat: {beat}"
        )

    @staticmethod
    def _log(ep: Optional[int], msg: str) -> None:
        prefix = f"[EP{ep:02d}]" if ep is not None else "[--]"
        print(f"{prefix} {msg}", flush=True)

    def plan(self, topic_hint: str, run_dir: Path | None = None, episode_number: Optional[int] = None) -> Dict[str, Any]:
        ep = episode_number
        self._log(ep, "researching…")
        # Resolve episode beat from guide
        episode_beat = self._resolve_episode_beat(episode_number)

        # When a specific episode is pinned, override topic_hint with a rich episode description
        if episode_beat is not None:
            topic_hint = self._build_topic_hint_for_episode(episode_beat)

        active_run_dir = run_dir or self._build_run_dir(
            f"episode-{episode_number}-of" if episode_number else topic_hint
        )
        logs_dir = ensure_dir(active_run_dir / "logs")
        write_text(logs_dir / "topic.txt", topic_hint)

        if episode_beat:
            write_json(active_run_dir / "episode-beat.json", episode_beat)

        continuity_state = self._load_series_state()
        write_json(active_run_dir / "series-state-before.json", continuity_state)

        research, research_raw = self._generate_json(
            prompt=research_prompt(self.config, topic_hint, continuity_state, episode_beat),
            model=self.config.models.research_model,
            temperature=self.config.pipeline.research_temperature,
            use_google_search=self.config.pipeline.use_google_search
        )
        write_json(active_run_dir / "research.json", research)
        write_json(active_run_dir / "logs" / "research-response.json", research_raw)

        self._log(ep, "writing concepts…")
        concepts_bundle, concepts_raw = self._generate_json(
            prompt=concept_prompt(self.config, research, topic_hint, continuity_state, episode_beat),
            model=self.config.models.writing_model,
            temperature=self.config.pipeline.writing_temperature,
            use_google_search=False
        )
        write_json(active_run_dir / "concepts.json", concepts_bundle)
        write_json(active_run_dir / "logs" / "concepts-response.json", concepts_raw)

        # When a specific episode is pinned, generate exactly 1 video (the episode itself)
        videos_per_run = 1 if episode_beat else self.config.pipeline.videos_per_run
        concepts = sorted(
            concepts_bundle.get("concepts", []),
            key=lambda item: int(item.get("model_score", 0)),
            reverse=True
        )[:videos_per_run]

        packs: List[Dict[str, Any]] = []
        for index, concept in enumerate(concepts, start=1):
            concept_dir = ensure_dir(active_run_dir / f"{index:02d}-{slugify(concept.get('slug', concept.get('name', 'concept')))}")
            packs.append(
                self._plan_single_concept(
                    run_dir=concept_dir,
                    concept=concept,
                    research=research,
                    continuity_state=continuity_state,
                    episode_beat=episode_beat
                )
            )

        updated_series_state = self._update_series_state_locked(packs)
        write_json(active_run_dir / "series-state-after.json", updated_series_state)

        manifest = {
            "topic_hint": topic_hint,
            "episode_number": episode_number,
            "episode_beat": episode_beat,
            "research": research,
            "concept_count": len(concepts),
            "packs": packs,
            "series_state": updated_series_state
        }
        write_json(active_run_dir / "run_manifest.json", manifest)
        return manifest

    def _plan_single_concept(
        self,
        *,
        run_dir: Path,
        concept: Dict[str, Any],
        research: Dict[str, Any],
        continuity_state: Dict[str, Any],
        episode_beat: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        ep = int(episode_beat.get("episode_number", 0)) if episode_beat else None
        self._log(ep, "writing script…")
        initial_script, initial_script_raw = self._generate_json(
            prompt=initial_script_prompt(self.config, research, concept, continuity_state, episode_beat),
            model=self.config.models.writing_model,
            temperature=self.config.pipeline.writing_temperature,
            use_google_search=False
        )
        initial_heuristic = score_script_heuristics(initial_script)
        write_json(run_dir / "script-00-initial.json", initial_script)
        write_json(run_dir / "logs" / "script-00-response.json", initial_script_raw)

        script_iterations: List[IterationResult] = [
            IterationResult(
                version_index=0,
                payload={"score_breakdown": {"total": 0}, "critique": {}, "script": initial_script},
                heuristic_score=initial_heuristic
            )
        ]
        current_script = initial_script
        for iteration_index in range(1, self.config.pipeline.script_iterations + 1):
            self._log(ep, f"script revision {iteration_index}/{self.config.pipeline.script_iterations}…")
            revised, revised_raw = self._generate_json(
                prompt=script_iteration_prompt(
                    self.config,
                    research,
                    concept,
                    current_script,
                    iteration_index,
                    continuity_state,
                    episode_beat
                ),
                model=self.config.models.critic_model,
                temperature=self.config.pipeline.critic_temperature,
                use_google_search=False
            )
            script_payload = revised["script"]
            heuristic = score_script_heuristics(script_payload)
            script_iterations.append(
                IterationResult(
                    version_index=iteration_index,
                    payload=revised,
                    heuristic_score=heuristic
                )
            )
            current_script = script_payload
            write_json(run_dir / f"script-{iteration_index:02d}.json", revised)
            write_json(run_dir / "logs" / f"script-{iteration_index:02d}-response.json", revised_raw)

        best_script_iteration = max(
            script_iterations,
            key=lambda item: (
                int(item.payload.get("score_breakdown", {}).get("total", 0)),
                int(item.heuristic_score.get("total", 0))
            )
        )
        final_script = best_script_iteration.payload["script"]
        final_script["heuristic_score"] = best_script_iteration.heuristic_score
        write_json(run_dir / "script-final.json", final_script)

        self._log(ep, "writing video plan…")
        initial_video_plan, initial_video_plan_raw = self._generate_json(
            prompt=initial_video_plan_prompt(self.config, research, concept, final_script, continuity_state, episode_beat),
            model=self.config.models.writing_model,
            temperature=self.config.pipeline.writing_temperature,
            use_google_search=False
        )
        write_json(run_dir / "video-plan-00-initial.json", initial_video_plan)
        write_json(run_dir / "logs" / "video-plan-00-response.json", initial_video_plan_raw)

        video_iterations: List[Dict[str, Any]] = [
            {
                "critique": {},
                "score_breakdown": {"total": 0},
                "video_plan": initial_video_plan
            }
        ]
        current_video_plan = initial_video_plan
        for iteration_index in range(1, self.config.pipeline.video_prompt_iterations + 1):
            self._log(ep, f"video plan revision {iteration_index}/{self.config.pipeline.video_prompt_iterations}…")
            revised_plan, revised_plan_raw = self._generate_json(
                prompt=video_iteration_prompt(
                    self.config,
                    research,
                    concept,
                    final_script,
                    current_video_plan,
                    iteration_index,
                    continuity_state,
                    episode_beat
                ),
                model=self.config.models.critic_model,
                temperature=self.config.pipeline.critic_temperature,
                use_google_search=False
            )
            video_iterations.append(revised_plan)
            current_video_plan = revised_plan["video_plan"]
            write_json(run_dir / f"video-plan-{iteration_index:02d}.json", revised_plan)
            write_json(run_dir / "logs" / f"video-plan-{iteration_index:02d}-response.json", revised_plan_raw)

        best_video_plan = max(
            video_iterations,
            key=lambda item: int(item.get("score_breakdown", {}).get("total", 0))
        )["video_plan"]
        write_json(run_dir / "video-plan-final.json", best_video_plan)

        captions, captions_raw = self._generate_json(
            prompt=caption_prompt(self.config, final_script, best_video_plan, continuity_state, episode_beat),
            model=self.config.models.writing_model,
            temperature=self.config.pipeline.writing_temperature,
            use_google_search=False
        )
        write_json(run_dir / "captions.json", captions)
        write_json(run_dir / "logs" / "captions-response.json", captions_raw)

        pack = {
            "concept": concept,
            "episode_beat": episode_beat,
            "final_script": final_script,
            "best_script_revision": best_script_iteration.version_index,
            "script_iterations": [
                {
                    "version_index": item.version_index,
                    "score_breakdown": item.payload.get("score_breakdown", {}),
                    "heuristic_score": item.heuristic_score,
                    "critique": item.payload.get("critique", {}),
                    "script": item.payload.get("script", {})
                }
                for item in script_iterations
            ],
            "final_video_plan": best_video_plan,
            "video_iterations": video_iterations,
            "captions": captions,
            "series_context": continuity_state,
            "run_dir": str(run_dir)
        }
        write_json(run_dir / "content-pack.json", pack)
        return pack

    def _series_character_sheet_path(self) -> Path:
        """Canonical character sheet shared across ALL episodes — generated once, reused forever."""
        return self._series_state_path().parent / "character-sheet.png"

    def generate_assets(self, pack_path: Path) -> Dict[str, Any]:
        pack = self._read_pack(pack_path)
        run_dir = Path(pack["run_dir"])
        media_dir = ensure_dir(run_dir / "media")
        asset_manifest: Dict[str, Any] = {}
        ep = int(pack.get("episode_beat", {}).get("episode_number", 0)) or None

        # --- Character sheet: generate once, reuse across all episodes (thread-safe) ---
        series_character_sheet: Optional[Path] = None

        if self.config.assets.generate_images and self.config.series.enabled:
            canonical_sheet = self._series_character_sheet_path()
            with self._continuity_lock:
                if not canonical_sheet.exists():
                    self._log(ep, "generating character sheet…")
                    character_sheet_prompt = pack["final_video_plan"].get("character_sheet_prompt_en", "")
                    if character_sheet_prompt:
                        ensure_dir(canonical_sheet.parent)
                        self.client.generate_image(
                            prompt=character_sheet_prompt,
                            output_path=canonical_sheet
                        )
            if canonical_sheet.exists():
                shutil.copy(canonical_sheet, media_dir / "character-sheet.png")
                series_character_sheet = canonical_sheet
                asset_manifest["character_sheet"] = {"reused_from": str(canonical_sheet)}

            location_path = media_dir / "location-sheet.png"
            location_sheet_prompt = pack["final_video_plan"].get("location_sheet_prompt_en", "")
            if location_sheet_prompt and not location_path.exists():
                self._log(ep, "generating location sheet…")
                asset_manifest["location_sheet"] = self.client.generate_image(
                    prompt=location_sheet_prompt,
                    output_path=location_path
                )
            elif location_path.exists():
                asset_manifest["location_sheet"] = {"reused_from": str(location_path)}

        if self.config.assets.generate_images:
            cover_path = media_dir / "cover.png"
            if not cover_path.exists():
                if self.config.series.enabled:
                    image_prompt = (
                        "Vertical 2D cartoon animated poster for an Egyptian TikTok social comedy series. "
                        f"Series title: {self.config.series.title}. "
                        f"Episode idea: {pack['concept']['angle_summary']}. "
                        f"Arabic cover text idea: {pack['final_video_plan'].get('cover_text_ar', '')}. "
                        "Protagonist: Egyptian man in mid-20s, messy dark curly hair, warm brown eyes, short trimmed beard, "
                        "green rubber wristband on right hand, home apron over blue hoodie. "
                        "Bold cartoon outlines, expressive faces, warm Cairo palette (ochre, terracotta, warm yellow), "
                        "craveable stylized food. No photoreal humans."
                    )
                else:
                    image_prompt = (
                        "Vertical premium food cover image for Egyptian TikTok. "
                        f"Theme: {pack['concept']['angle_summary']}. "
                        f"Use this Arabic cover text idea: {pack['final_video_plan'].get('cover_text_ar', '')}. "
                        "High contrast food close-up, cinematic appetizing texture, no watermark, no brand logos."
                    )
                self._log(ep, "generating cover…")
                asset_manifest["cover_image"] = self.client.generate_image(
                    prompt=image_prompt,
                    output_path=cover_path
                )
            else:
                asset_manifest["cover_image"] = {"reused_from": str(cover_path)}

        if self.config.assets.generate_videos:
            shot_assets: List[Dict[str, Any]] = []
            shots = pack["final_video_plan"].get("shots", [])[:self.config.assets.video_count_per_script]

            # Distribute dialogue lines evenly across shots for per-shot audio
            dialogue_all = pack["final_script"].get("dialogue", [])
            n_shots = len(shots)

            for shot_pos, shot in enumerate(shots):
                idx = int(shot["shot_index"])
                video_path = media_dir / f"shot-{idx:02d}.mp4"

                # --- Per-shot audio: use shot-level voiceover_ar, fall back to proportional slice ---
                if self.config.assets.generate_audio:
                    shot_audio_path = media_dir / f"shot-{idx:02d}-audio.wav"
                    if not shot_audio_path.exists():
                        self._log(ep, f"shot {shot_pos+1}/{n_shots} audio…")
                        shot_voiceover = shot.get("voiceover_ar", "").strip()
                        try:
                            if shot_voiceover and self.config.series.character_voices:
                                # New path: shot has its own written voiceover — use it directly
                                self.client.synthesize_dialogue(
                                    dialogue=[{"speaker": "كريم", "text": shot_voiceover}],
                                    character_voices=self.config.series.character_voices,
                                    default_voice=self.config.assets.tts_voice_name,
                                    language_code=self.config.assets.tts_language_code,
                                    output_path=shot_audio_path,
                                )
                            elif dialogue_all and self.config.series.character_voices:
                                # Fallback: proportional slice of global dialogue (old episodes)
                                start = (shot_pos * len(dialogue_all)) // n_shots
                                end = ((shot_pos + 1) * len(dialogue_all)) // n_shots
                                shot_dialogue = dialogue_all[start:end]
                                if not shot_dialogue:
                                    shot_dialogue = [{"speaker": "narrator", "text": shot.get("overlay_text_ar", "")}]
                                self.client.synthesize_dialogue(
                                    dialogue=shot_dialogue,
                                    character_voices=self.config.series.character_voices,
                                    default_voice=self.config.assets.tts_voice_name,
                                    language_code=self.config.assets.tts_language_code,
                                    output_path=shot_audio_path,
                                )
                            else:
                                self.client.synthesize_speech(
                                    text=shot_voiceover or shot.get("overlay_text_ar", ""),
                                    prompt="Speak in energetic colloquial Egyptian Arabic, fast and punchy.",
                                    output_path=shot_audio_path,
                                )
                        except Exception as exc:
                            self._log(ep, f"  ⚠ shot {shot_pos+1}/{n_shots} audio failed (will use silence): {exc}")

                if video_path.exists():
                    metadata: Dict[str, Any] = {"local_file": str(video_path), "reused": True}
                else:
                    # Generate a per-shot reference image for image-to-video consistency
                    ref_image_path = media_dir / f"shot-{idx:02d}-ref.png"
                    source_image: Path | None = None
                    shot_image_prompt = shot.get("shot_image_prompt_en", "")
                    if shot_image_prompt and self.config.assets.generate_images:
                        if not ref_image_path.exists():
                            self._log(ep, f"shot {shot_pos+1}/{n_shots} ref image…")
                            self.client.generate_image(
                                prompt=shot_image_prompt,
                                output_path=ref_image_path,
                            )
                        source_image = ref_image_path
                    self._log(ep, f"shot {shot_pos+1}/{n_shots} video (Veo, ~4 min)…")
                    metadata = self.client.generate_video(
                        prompt=shot["veo_prompt_en"],
                        negative_prompt=shot.get("negative_prompt_en", ""),
                        output_dir=media_dir,
                        shot_index=idx,
                        source_image=source_image,
                    )
                shot_assets.append(
                    {
                        "shot_index": shot["shot_index"],
                        "overlay_text_ar": shot["overlay_text_ar"],
                        "continuity_lock": shot.get("continuity_lock", ""),
                        "metadata": metadata
                    }
                )
            asset_manifest["videos"] = shot_assets

        write_json(run_dir / "media-manifest.json", asset_manifest)

        # --- ffmpeg: assemble final episode video ---
        if self.config.assets.generate_videos:
            self._log(ep, "assembling final episode video…")
            self._assemble_episode(run_dir, media_dir, asset_manifest)

        return asset_manifest

    def _assemble_episode(self, run_dir: Path, media_dir: Path, asset_manifest: Dict[str, Any]) -> None:
        """Concatenate shot mp4s, each muxed with its own audio clip, into a final episode file.

        Each shot-NN.mp4 is paired with shot-NN-audio.wav. The audio is padded/trimmed
        to the shot duration so audio and video are frame-aligned.
        """
        shot_dur = self.config.assets.video_duration_seconds
        shot_files = sorted(media_dir.glob("shot-??.mp4"))
        if not shot_files:
            return

        final_path = run_dir / "episode-final.mp4"
        n = len(shot_files)

        # Build inputs: interleaved video + audio for each shot
        cmd: List[str] = ["ffmpeg", "-y"]
        for sf in shot_files:
            cmd += ["-i", str(sf)]
            # Corresponding per-shot audio (fall back to silence if missing)
            audio_file = media_dir / (sf.stem + "-audio.wav")
            if audio_file.exists():
                cmd += ["-i", str(audio_file)]
            else:
                # Generate silent audio as a fallback via lavfi
                cmd += ["-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={shot_dur}"]

        # Per-shot: pad/trim audio to shot_dur, stereo upmix, loudnorm
        audio_parts: List[str] = []
        for i in range(n):
            v_idx = i * 2       # video input index
            a_idx = i * 2 + 1   # audio input index
            audio_parts.append(
                f"[{a_idx}:a]atrim=duration={shot_dur},"
                f"apad=pad_dur={shot_dur},"
                f"atrim=duration={shot_dur},"
                f"loudnorm,aresample=48000,"
                f"pan=stereo|c0=c0|c1=c0[a{i}]"
            )

        video_concat_inputs = "".join(f"[{i*2}:v]" for i in range(n))
        audio_concat_inputs = "".join(f"[a{i}]" for i in range(n))

        filter_complex = (
            ";".join(audio_parts) + ";"
            f"{video_concat_inputs}concat=n={n}:v=1:a=0,"
            f"eq=saturation=1.12:contrast=1.05:gamma=0.98,format=yuv420p[v];"
            f"{audio_concat_inputs}concat=n={n}:v=0:a=1[a]"
        )

        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(final_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        asset_manifest["final_video"] = str(final_path)

    def _read_pack(self, pack_path: Path) -> Dict[str, Any]:
        return json.loads(pack_path.read_text(encoding="utf-8"))
