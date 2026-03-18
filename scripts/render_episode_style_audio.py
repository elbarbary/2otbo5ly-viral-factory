from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from viral_factory.config import load_config
from viral_factory.gcp_client import VertexClientError, VertexFactoryClient
from viral_factory.io_utils import ensure_dir, write_json

STYLE_PROMPT = (
    "Vertical 9:16 style key art for a 2D Egyptian social-comedy animation. "
    "Same warm Cairo kitchen at night in every scene: amber overhead light, worn marble counter, "
    "fridge with magnets and handwritten notes, cheap ring light. "
    "Karim is a mid-20s Egyptian chef with curly dark hair, short trimmed beard, "
    "off-white apron over a dark t-shirt, and a green wristband on his right hand. "
    "Ziad is a young Egyptian man in a hoodie with a cheap gimbal and a cracked-corner Android phone. "
    "Bold clean outlines, expressive eyebrows, consistent face proportions, warm orange-gold palette, "
    "craveable glossy dessert on the counter, polished cartoon finish, not photoreal, not 3D, not painterly."
)

COMMON_STILL_PREFIX = (
    "Vertical 9:16 animation keyframe for the same 2D Egyptian social-comedy episode. "
    "Same warm Cairo kitchen at night in every scene: amber overhead light, worn marble counter, fridge with magnets, cheap ring light. "
    "Same Karim in every shot: mid-20s Egyptian man, curly dark hair, short trimmed beard, off-white apron over dark t-shirt, green wristband on right hand. "
    "Same Ziad when present: young Egyptian man in a hoodie with a cheap gimbal and a cracked-corner Android phone. "
    "Bold clean outlines, expressive eyebrows, consistent face proportions, warm orange-gold palette, polished 2D finish, craveable glossy food, no readable text, no logos, no watermarks."
)

SHOT_KEYFRAME_PROMPTS = {
    1: (
        COMMON_STILL_PREFIX
        + " Ziad steps beside Karim and raises the cracked-corner phone toward him to show a trendy dessert challenge. "
        "Karim shifts from surprise into calm competitive confidence. Simple clear acting, friendly energy, same kitchen framing."
    ),
    2: (
        COMMON_STILL_PREFIX
        + " Karim takes the phone from Ziad, points to himself, then points toward the camera with a proud playful challenge. "
        "Ziad reacts with impressed excitement in the background."
    ),
    3: (
        COMMON_STILL_PREFIX
        + " A simple grocery bag sits on the counter beside Karim. Karim looks at the bag, glances at the ingredients, then gives Ziad a sheepish smile and tiny shrug as he realizes the budget pressure. "
        "Gentle family-comedy tone, very simple staging."
    ),
    4: (
        COMMON_STILL_PREFIX
        + " Rapid dessert-prep montage in the same kitchen. Karim chops, whisks, pours, plates, and moves with confident skill. "
        "Show visible steam, glossy ingredients, warm highlights, and crisp cartoon motion."
    ),
    5: (
        COMMON_STILL_PREFIX
        + " Karim proudly presents the finished dessert close to camera. The dessert looks glossy, steaming, carefully plated, and highly appetizing. "
        "Karim smiles with proud triumph while the camera pushes in slightly on the food."
    ),
    6: (
        COMMON_STILL_PREFIX
        + " Karim's cracked-corner phone vibrates on the counter while a few generic notification dots pulse softly beside it. "
        "Ziad points at the phone in comic alarm, and Karim's proud smile turns into concern."
    ),
    7: (
        COMMON_STILL_PREFIX
        + " Karim checks the same phone, counts quickly on his fingers, and gives a comedic overwhelmed face. "
        "Use only tiny abstract interface glows near the phone, keep the composition simple, playful, and text-free."
    ),
    8: (
        COMMON_STILL_PREFIX
        + " Closing beat: Karim gently sets the phone on the counter, exhales, and gives a tired but funny sideways look as if wondering what comes next. "
        "Softly overwhelmed, playful rather than dramatic, same wardrobe and same kitchen."
    ),
}

SHOT_MOTION_PROMPTS = {
    1: "Subtle animated motion as Ziad lifts the phone and Karim shifts from surprise to confidence. Keep the same framing and same polished 2D look.",
    2: "Subtle animated motion as Karim takes the phone, points to himself, and gives a proud challenge look. Keep the same character proportions and kitchen details.",
    3: "Subtle animated motion as Karim glances at the grocery bag and gives a sheepish shrug. Keep the moment gentle and simple.",
    4: "Animated cooking motion with quick hands, steam, and energetic prep while preserving the same illustration style and kitchen layout.",
    5: "Animated hero reveal motion with a slight push-in toward the plated dessert and Karim's proud smile.",
    6: "Animated reaction motion as the phone vibrates, Ziad points, and Karim's expression turns concerned.",
    7: "Animated reaction motion as Karim checks the phone, counts quickly on his fingers, and gives a playful overwhelmed face.",
    8: "Animated closing motion as Karim sets the phone down, exhales, and gives a tired sideways look.",
}

SHOT_AUDIO = {
    1: "زياد ورّى كريم أكلة ترند وميزانية صعبة قوي.",
    2: "كريم بص للتليفون وقال: طب أنا هوريكم شغل البيت الصح.",
    3: "بس أول ما شاف المقاضي، فهم إن التحدي هيبقى على الضيق.",
    4: "ومع كده دخل المطبخ واشتغل بسرعة وشطارة.",
    5: "وفي الآخر، الطبق طلع تحفة تفتح النفس.",
    6: "أول ما الفيديو انتشر، تليفون كريم بدأ يولع طلبات.",
    7: "رسائل كتير وميزانيات مختلفة، وكريم ابتدى يتوتر.",
    8: "وفي الآخر سأل نفسه: مين هيلم الفوضى دي كلها؟",
}

NEGATIVE_PROMPT = (
    "photoreal humans, realistic live-action skin, 3d render, painterly texture, different wardrobe, "
    "missing green wristband, different phone, different kitchen, readable text, subtitles, logos, watermarks, "
    "extra people, deformed hands, distorted faces, low detail, dull colors, horror mood, crying, aggressive action"
)

VOICEOVER_PROMPT = (
    "Speak in energetic colloquial Egyptian Arabic suitable for a short-form social-comedy video. "
    "Keep it conversational, punchy, and clear."
)

FINAL_GRADE_FILTER = (
    "eq=saturation=1.12:contrast=1.06:brightness=0.01:gamma=0.98,"
    "unsharp=5:5:0.6:5:5:0.0,format=yuv420p"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a style-anchored narrated episode.")
    parser.add_argument("--config", default="configs/2otbo5ly.json")
    parser.add_argument(
        "--episode-dir",
        default="runs/20260314-182320-episode-1-of/01-fired-famous-ep1",
    )
    parser.add_argument("--output-dir-name", default="media-style-audio")
    parser.add_argument("--final-name", default="episode-01-style-audio.mp4")
    parser.add_argument(
        "--shots",
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated shot indices to generate.",
    )
    parser.add_argument("--seed", type=int, default=24031677)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_shots(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def write_text_file(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_style_anchor(client: VertexFactoryClient, output_dir: Path, force: bool) -> Path:
    style_anchor_path = output_dir / "style-anchor.png"
    if style_anchor_path.exists() and not force:
        print(f"STYLE anchor exists: {style_anchor_path}", flush=True)
        return style_anchor_path
    print(f"STYLE generating anchor: {style_anchor_path}", flush=True)
    metadata = client.generate_image(prompt=STYLE_PROMPT, output_path=style_anchor_path)
    write_json(style_anchor_path.with_suffix(".json"), metadata)
    print(f"STYLE generated anchor: {style_anchor_path}", flush=True)
    return style_anchor_path


def generate_keyframes(client: VertexFactoryClient, output_dir: Path, shot_indices: list[int], force: bool) -> Path:
    keyframe_dir = ensure_dir(output_dir / "keyframes")
    for shot_index in shot_indices:
        frame_path = keyframe_dir / f"shot-{shot_index:02d}.png"
        if frame_path.exists() and not force:
            print(f"KEYFRAME skip {shot_index}: {frame_path}", flush=True)
            continue
        print(f"KEYFRAME start {shot_index}", flush=True)
        metadata = client.generate_image(
            prompt=SHOT_KEYFRAME_PROMPTS[shot_index],
            output_path=frame_path,
        )
        write_json(keyframe_dir / f"shot-{shot_index:02d}.json", metadata)
        print(f"KEYFRAME done {shot_index}: {frame_path}", flush=True)
    return keyframe_dir


def generate_shots(
    client: VertexFactoryClient,
    output_dir: Path,
    keyframe_dir: Path,
    shot_indices: list[int],
    seed: int,
    force: bool,
) -> None:
    for shot_index in shot_indices:
        shot_path = output_dir / f"shot-{shot_index:02d}.mp4"
        metadata_path = output_dir / f"shot-{shot_index:02d}-metadata.json"
        if shot_path.exists() and metadata_path.exists() and not force:
            print(f"VIDEO skip {shot_index}: {shot_path}", flush=True)
            continue
        frame_path = keyframe_dir / f"shot-{shot_index:02d}.png"
        print(f"VIDEO start {shot_index}", flush=True)
        metadata = client.generate_video(
            prompt=SHOT_MOTION_PROMPTS[shot_index],
            negative_prompt=NEGATIVE_PROMPT,
            seed=seed,
            output_dir=output_dir,
            shot_index=shot_index,
            source_image=frame_path,
        )
        compact = {
            "shot_index": shot_index,
            "local_file": metadata.get("local_file"),
            "gcs_uri": metadata.get("gcs_uri"),
            "generated_videos": bool(metadata.get("generated_videos")),
            "rai_media_filtered_count": metadata.get("rai_media_filtered_count"),
            "rai_media_filtered_reasons": metadata.get("rai_media_filtered_reasons"),
            "status": "generated" if metadata.get("local_file") else "no-local-file",
            "source_image": str(frame_path),
        }
        write_json(output_dir / f"shot-{shot_index:02d}-raw-metadata.json", metadata)
        write_json(metadata_path, compact)
        if not metadata.get("local_file"):
            raise VertexClientError(f"Shot {shot_index} did not produce a local file: {compact}")
        print(f"VIDEO done {shot_index}: {shot_path}", flush=True)


def generate_audio_segments(
    client: VertexFactoryClient,
    output_dir: Path,
    shot_indices: list[int],
    force: bool,
) -> Path:
    audio_dir = ensure_dir(output_dir / "audio")
    segment_list_path = audio_dir / "segments.txt"
    segment_lines: list[str] = []
    for shot_index in shot_indices:
        raw_audio_path = audio_dir / f"shot-{shot_index:02d}-raw.wav"
        segment_path = audio_dir / f"shot-{shot_index:02d}.wav"
        if force or not raw_audio_path.exists():
            print(f"AUDIO synth {shot_index}", flush=True)
            client.synthesize_speech(
                text=SHOT_AUDIO[shot_index],
                prompt=VOICEOVER_PROMPT,
                output_path=raw_audio_path,
            )
        if force or not segment_path.exists():
            print(f"AUDIO normalize {shot_index}", flush=True)
            run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(raw_audio_path),
                    "-af",
                    "loudnorm,apad=pad_dur=8",
                    "-ar",
                    "48000",
                    "-t",
                    "8",
                    str(segment_path),
                ]
            )
        segment_lines.append(f"file '{segment_path.name}'")
    write_text_file(segment_list_path, segment_lines)
    full_audio_path = audio_dir / "episode-voiceover.wav"
    print(f"AUDIO concat {full_audio_path}", flush=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(segment_list_path),
            "-c",
            "copy",
            str(full_audio_path),
        ]
    )
    return full_audio_path


def render_final_video(
    episode_dir: Path,
    output_dir: Path,
    shot_indices: list[int],
    voiceover_path: Path,
    final_name: str,
) -> Path:
    temp_video_path = episode_dir / "episode-01-style-silent.mp4"
    final_output_path = episode_dir / final_name
    print(f"VIDEO concat silent master: {temp_video_path}", flush=True)
    ffmpeg_command = ["ffmpeg", "-y"]
    for shot_index in shot_indices:
        ffmpeg_command.extend(["-i", str(output_dir / f"shot-{shot_index:02d}.mp4")])
    concat_inputs = "".join(f"[{index}:v]" for index in range(len(shot_indices)))
    ffmpeg_command.extend(
        [
            "-filter_complex",
            f"{concat_inputs}concat=n={len(shot_indices)}:v=1:a=0,{FINAL_GRADE_FILTER}[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(temp_video_path),
        ]
    )
    run_command(ffmpeg_command)
    print(f"MUX final episode: {final_output_path}", flush=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_video_path),
            "-i",
            str(voiceover_path),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-shortest",
            str(final_output_path),
        ]
    )
    return final_output_path


def main() -> None:
    args = parse_args()
    shot_indices = parse_shots(args.shots)
    config = load_config(args.config)
    config.assets.video_output_gcs_uri = ""
    client = VertexFactoryClient(config)
    episode_dir = Path(args.episode_dir)
    output_dir = ensure_dir(episode_dir / args.output_dir_name)

    style_anchor_path = generate_style_anchor(client, output_dir, args.force)
    keyframe_dir = generate_keyframes(client, output_dir, shot_indices, args.force)
    generate_shots(client, output_dir, keyframe_dir, shot_indices, args.seed, args.force)

    if shot_indices != [1, 2, 3, 4, 5, 6, 7, 8]:
        print(
            json.dumps(
                {
                    "status": "partial-video-only",
                    "style_anchor": str(style_anchor_path),
                    "keyframes": str(keyframe_dir),
                }
            )
        )
        return

    voiceover_path = generate_audio_segments(client, output_dir, shot_indices, args.force)
    final_output_path = render_final_video(
        episode_dir=episode_dir,
        output_dir=output_dir,
        shot_indices=shot_indices,
        voiceover_path=voiceover_path,
        final_name=args.final_name,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "style_anchor": str(style_anchor_path),
                "voiceover": str(voiceover_path),
                "final_output": str(final_output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
