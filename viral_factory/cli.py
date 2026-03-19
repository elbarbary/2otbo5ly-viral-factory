from __future__ import annotations

import argparse
import sys
import time
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

    series_parser = subparsers.add_parser("run-series", help="Generate all episodes in the series sequentially.")
    series_parser.add_argument(
        "--from-episode", type=int, default=1, metavar="N",
        help="Start from this episode number (default: 1). Use to resume after a failure."
    )
    series_parser.add_argument(
        "--to-episode", type=int, default=None, metavar="N",
        help="Stop after this episode number (default: last episode in guide)."
    )
    series_parser.add_argument(
        "--plan-only", action="store_true",
        help="Run planning stage only (no image/video/audio generation)."
    )

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

    if args.command == "run-series":
        _cmd_run_series(config, pipeline, args)
        return

    parser.error("Unknown command.")


def _cmd_run_series(config, pipeline: ViralFactoryPipeline, args) -> None:
    guide = config.series.episode_guide
    if not guide:
        print("ERROR: No episode guide found in config.", file=sys.stderr)
        sys.exit(1)

    total = len(guide)
    from_ep = args.from_episode
    to_ep = args.to_episode if args.to_episode is not None else guide[-1]["episode_number"]
    episodes = [ep for ep in guide if from_ep <= ep["episode_number"] <= to_ep]

    print(f"\n=== الشيف كريم — Full Series Run ===")
    print(f"Episodes {from_ep}–{to_ep} ({len(episodes)} of {total} total)\n", flush=True)

    generate = not args.plan_only and any([
        config.assets.generate_images,
        config.assets.generate_audio,
        config.assets.generate_videos,
    ])

    series_start = time.monotonic()
    for idx, ep in enumerate(episodes, start=1):
        ep_num = ep["episode_number"]
        t0 = time.monotonic()
        print(f"\n[{idx}/{len(episodes)}] EP{ep_num:02d}: {ep.get('title_en', '')} — planning…", flush=True)
        try:
            manifest = pipeline.plan(topic_hint="", episode_number=ep_num)
            if generate:
                for pack in manifest["packs"]:
                    print(f"  → generating assets…", flush=True)
                    pipeline.generate_assets(Path(pack["run_dir"]) / "content-pack.json")
            elapsed = time.monotonic() - t0
            total_elapsed = time.monotonic() - series_start
            remaining = len(episodes) - idx
            eta_s = (total_elapsed / idx) * remaining if idx > 0 else 0
            print(
                f"  ✓ EP{ep_num:02d} done in {elapsed/60:.1f}m | "
                f"total {total_elapsed/3600:.1f}h | "
                f"ETA {eta_s/3600:.1f}h for {remaining} remaining",
                flush=True
            )
        except Exception as exc:
            print(f"\n  ✗ EP{ep_num:02d} FAILED: {exc}", file=sys.stderr, flush=True)
            print(f"  → Resume with: --from-episode {ep_num}", file=sys.stderr)
            sys.exit(1)

    total_time = time.monotonic() - series_start
    print(f"\n=== Series complete: {len(episodes)} episodes in {total_time/3600:.2f}h ===\n", flush=True)
