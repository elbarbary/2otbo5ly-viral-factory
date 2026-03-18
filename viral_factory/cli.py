from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import ViralFactoryPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viral-factory",
        description="Build grounded Egyptian TikTok content packs on Vertex AI."
    )
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Research, iterate, and write content packs.")
    plan_parser.add_argument(
        "--topic",
        default="Pre-launch cartoon series for 2otbo5ly with Egyptian food TikTok energy",
        help="Topic or campaign hint used for grounded research. Ignored when --episode is set."
    )
    plan_parser.add_argument(
        "--episode",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Generate content for a specific episode number from the series episode guide (1–50). "
            "When set, the topic hint is automatically built from the episode beat and only 1 video is produced."
        )
    )

    run_parser = subparsers.add_parser("run", help="Plan content packs and optionally generate media assets.")
    run_parser.add_argument(
        "--topic",
        default="Pre-launch cartoon series for 2otbo5ly with Egyptian food TikTok energy",
        help="Topic or campaign hint used for grounded research. Ignored when --episode is set."
    )
    run_parser.add_argument(
        "--episode",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Generate content for a specific episode number from the series episode guide (1–50). "
            "When set, the topic hint is automatically built from the episode beat and only 1 video is produced."
        )
    )

    generate_parser = subparsers.add_parser("generate", help="Generate assets for an existing content pack.")
    generate_parser.add_argument("--pack", required=True, help="Path to a content-pack.json file.")

    list_parser = subparsers.add_parser("episodes", help="List all episodes in the series episode guide.")
    list_parser.add_argument(
        "--act",
        type=int,
        default=None,
        metavar="N",
        help="Filter by act number (1–5)."
    )

    return parser


def _cmd_episodes(config_path: str, act_filter: int | None) -> None:
    from .config import load_config as _lc
    config = _lc(config_path)
    guide = config.series.episode_guide
    if not guide:
        print("No episode guide found in config.")
        return

    print(f"\n{'الشيف كريم'} — Episode Guide ({len(guide)} episodes)\n")
    current_act = None
    for ep in guide:
        act = ep.get("act")
        if act_filter is not None and act != act_filter:
            continue
        if act != current_act:
            current_act = act
            act_entry = next((a for a in config.series.act_structure if a.get("act") == act), {})
            print(f"\n── ACT {act}: {act_entry.get('title', '')} (Episodes {act_entry.get('episodes', '')}) ──")
        ep_num = ep.get("episode_number", "?")
        title_ar = ep.get("title_ar", "")
        title_en = ep.get("title_en", "")
        tone = ep.get("emotion_tone", "")
        chars = ", ".join(ep.get("key_characters", []))
        print(f"  EP{ep_num:02d}  {title_ar} / {title_en}")
        print(f"        Tone: {tone} | Characters: {chars}")
    print()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "episodes":
        _cmd_episodes(args.config, getattr(args, "act", None))
        return

    config = load_config(args.config)
    pipeline = ViralFactoryPipeline(config)

    if args.command == "plan":
        pipeline.plan(topic_hint=args.topic, episode_number=args.episode)
        return

    if args.command == "run":
        manifest = pipeline.plan(topic_hint=args.topic, episode_number=args.episode)
        if any([config.assets.generate_images, config.assets.generate_audio, config.assets.generate_videos]):
            for pack in manifest["packs"]:
                pipeline.generate_assets(Path(pack["run_dir"]) / "content-pack.json")
        return

    if args.command == "generate":
        pipeline.generate_assets(Path(args.pack).expanduser().resolve())
        return

    parser.error("Unknown command.")
