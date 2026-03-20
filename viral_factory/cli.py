from __future__ import annotations

import argparse
import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    series_parser.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help="Number of episodes to generate in parallel (default: 1). "
             "5 workers = ~3h for 50 episodes. Raises quota usage proportionally."
    )

    validate_parser = subparsers.add_parser("validate", help="Check all episodes for missing assets and optionally fix them.")
    validate_parser.add_argument(
        "--fix", action="store_true",
        help="Re-run generate on episodes with missing assets."
    )
    validate_parser.add_argument(
        "--workers", type=int, default=4, metavar="N",
        help="Workers for --fix mode (default: 4)."
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

    if args.command == "validate":
        _cmd_validate(config, pipeline, args)
        return

    parser.error("Unknown command.")


def _cmd_validate(config, pipeline: ViralFactoryPipeline, args) -> None:
    """Scan all run directories and report missing assets per episode."""
    runs_dir = Path(config.pipeline.run_root)
    if not runs_dir.exists():
        print("No runs directory found.")
        return

    ep_runs: Dict[int, Path] = {}
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or d.name == "series":
            continue
        parts = d.name.split("-episode-")
        if len(parts) < 2:
            continue
        try:
            ep_num = int(parts[1].split("-")[0])
        except ValueError:
            continue
        ep_runs[ep_num] = d

    if not ep_runs:
        print("No episode runs found.")
        return

    guide = config.series.episode_guide or []
    total_eps = len(guide)
    missing_eps = [ep["episode_number"] for ep in guide if ep["episode_number"] not in ep_runs]

    print(f"\n=== Validate: {len(ep_runs)} episodes found, {total_eps} expected ===\n")
    if missing_eps:
        print(f"Missing episodes (no run dir): {missing_eps}\n")

    incomplete = []
    for ep_num in sorted(ep_runs):
        run_dir = ep_runs[ep_num]
        concept_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name not in ("logs",)]
        if not concept_dirs:
            print(f"  EP{ep_num:02d}: NO concept directory")
            incomplete.append(ep_num)
            continue

        cp = concept_dirs[0]
        media = cp / "media"
        final = cp / "episode-final.mp4"

        vp_file = cp / "video-plan-final.json"
        n_shots = 3
        if vp_file.exists():
            try:
                vp = json.loads(vp_file.read_text())
                n_shots = len(vp.get("shots", []))
            except Exception:
                pass

        missing = []
        if not (cp / "content-pack.json").exists():
            missing.append("content-pack")
        for i in range(n_shots):
            s = f"shot-{i+1:02d}"
            if not (media / f"{s}.mp4").exists():
                missing.append(f"{s}.mp4")
            if not (media / f"{s}-audio.wav").exists():
                missing.append(f"{s}-audio.wav")
            if not (media / f"{s}-ref.png").exists():
                missing.append(f"{s}-ref.png")
        if not final.exists():
            missing.append("episode-final.mp4")
        if not (media / "cover.png").exists():
            missing.append("cover.png")

        if missing:
            print(f"  EP{ep_num:02d}: INCOMPLETE — missing {', '.join(missing)}")
            incomplete.append(ep_num)
        else:
            print(f"  EP{ep_num:02d}: OK")

    print(f"\n{len(ep_runs) - len(incomplete)} complete, {len(incomplete)} incomplete")
    if missing_eps:
        print(f"{len(missing_eps)} episodes never started: {missing_eps}")

    if not args.fix:
        if incomplete or missing_eps:
            print(f"\nRe-run with --fix to regenerate missing assets.")
        return

    to_fix = []
    for ep_num in incomplete:
        if ep_num in ep_runs:
            packs = sorted(ep_runs[ep_num].rglob("content-pack.json"))
            if packs:
                to_fix.append((ep_num, packs[0]))

    if not to_fix and not missing_eps:
        print("Nothing to fix.")
        return

    print(f"\nFixing {len(to_fix)} incomplete + {len(missing_eps)} missing episodes...\n")

    def _fix_one(item):
        ep_num, pack_path = item
        print(f"  → EP{ep_num:02d} regenerating...", flush=True)
        pipeline.generate_assets(pack_path)
        print(f"  ✓ EP{ep_num:02d} fixed", flush=True)

    def _plan_one(ep_num):
        print(f"  → EP{ep_num:02d} planning + generating...", flush=True)
        manifest = pipeline.plan(topic_hint="", episode_number=ep_num)
        for pack in manifest["packs"]:
            pipeline.generate_assets(Path(pack["run_dir"]) / "content-pack.json")
        print(f"  ✓ EP{ep_num:02d} done", flush=True)

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for ep_num, pack_path in to_fix:
            futures[executor.submit(_fix_one, (ep_num, pack_path))] = ep_num
        for ep_num in missing_eps:
            futures[executor.submit(_plan_one, ep_num)] = ep_num
        for future in as_completed(futures):
            ep_num = futures[future]
            try:
                future.result()
            except Exception as exc:
                failed.append(ep_num)
                print(f"  ✗ EP{ep_num:02d} fix failed: {exc}", flush=True)

    print(f"\nFixed: {len(to_fix) + len(missing_eps) - len(failed)} OK, {len(failed)} failed")
    if failed:
        print(f"Still broken: {sorted(failed)}")


def _cmd_run_series(config, pipeline: ViralFactoryPipeline, args) -> None:
    guide = config.series.episode_guide
    if not guide:
        print("ERROR: No episode guide found in config.", file=sys.stderr)
        sys.exit(1)

    total = len(guide)
    from_ep = args.from_episode
    to_ep = args.to_episode if args.to_episode is not None else guide[-1]["episode_number"]
    episodes = [ep for ep in guide if from_ep <= ep["episode_number"] <= to_ep]
    workers = args.workers

    generate = not args.plan_only and any([
        config.assets.generate_images,
        config.assets.generate_audio,
        config.assets.generate_videos,
    ])

    print(f"\n=== الشيف كريم — Full Series Run ===", flush=True)
    print(f"Episodes {from_ep}–{to_ep} ({len(episodes)} of {total} total) | workers={workers}\n", flush=True)

    _print_lock = threading.Lock()
    counters = {"done": 0, "failed": 0}
    series_start = time.monotonic()

    def _log(msg: str) -> None:
        with _print_lock:
            print(msg, flush=True)

    def _run_one(ep: dict) -> int:
        ep_num = ep["episode_number"]
        t0 = time.monotonic()
        _log(f"  → EP{ep_num:02d} start: {ep.get('title_en', '')}")

        # Resume: check for existing run directory with a content-pack
        runs_dir = Path(config.pipeline.run_root)
        existing_packs = sorted(runs_dir.glob(f"*-episode-{ep_num}-of/*/content-pack.json"))
        if existing_packs:
            _log(f"  → EP{ep_num:02d} resuming existing run")
            if generate:
                for cp in existing_packs:
                    pipeline.generate_assets(cp)
        else:
            manifest = pipeline.plan(topic_hint="", episode_number=ep_num)
            if generate:
                for pack in manifest["packs"]:
                    pipeline.generate_assets(Path(pack["run_dir"]) / "content-pack.json")

        elapsed = time.monotonic() - t0
        _log(f"  ✓ EP{ep_num:02d} done in {elapsed/60:.1f}m")
        return ep_num

    failed_episodes: List[int] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_one, ep): ep["episode_number"] for ep in episodes}
        for future in as_completed(futures):
            ep_num = futures[future]
            try:
                future.result()
                counters["done"] += 1
            except Exception as exc:
                counters["failed"] += 1
                failed_episodes.append(ep_num)
                _log(f"  ✗ EP{ep_num:02d} FAILED: {exc}")

            done = counters["done"] + counters["failed"]
            elapsed_total = time.monotonic() - series_start
            remaining = len(episodes) - done
            eta_s = (elapsed_total / done * remaining) if done > 0 else 0
            _log(
                f"  [{done}/{len(episodes)}] "
                f"elapsed {elapsed_total/3600:.1f}h | ETA {eta_s/3600:.1f}h"
            )

    total_time = time.monotonic() - series_start
    print(f"\n=== Done: {counters['done']} OK, {counters['failed']} failed in {total_time/3600:.2f}h ===", flush=True)
    if failed_episodes:
        print(f"Failed episodes: {sorted(failed_episodes)}", file=sys.stderr)
        print(f"Retry failed: --from-episode <N> --to-episode <N> for each", file=sys.stderr)
        sys.exit(1)


def _cmd_validate(config, pipeline: ViralFactoryPipeline, args) -> None:
    runs_dir = Path(config.pipeline.run_root)
    workers = args.workers
    fix = args.fix

    # Find all content packs grouped by episode number
    all_packs = sorted(runs_dir.glob("*-episode-*-of/*/content-pack.json"))
    if not all_packs:
        print("No runs found.")
        return

    # Group by episode number, keep latest run per episode
    ep_packs: Dict[int, Path] = {}
    for cp in all_packs:
        # directory name like "20260319-194220-episode-3-of"
        run_dir_name = cp.parent.parent.name
        try:
            ep_num = int(run_dir_name.split("-episode-")[1].split("-of")[0])
        except (IndexError, ValueError):
            continue
        ep_packs[ep_num] = cp  # sorted order means last = latest

    incomplete = []
    missing_eps = []

    print(f"\n=== Validate: checking {len(ep_packs)} episodes ===\n", flush=True)

    for ep_num in sorted(ep_packs):
        cp = ep_packs[ep_num]
        media_dir = cp.parent / "media"
        pack_data = json.loads(cp.read_text())
        n_shots = len(pack_data.get("video_plan", {}).get("shots", []))
        if n_shots == 0:
            n_shots = 3

        issues = []
        for i in range(1, n_shots + 1):
            if not (media_dir / f"shot-{i:02d}.mp4").exists():
                issues.append(f"shot-{i:02d}.mp4")
            if not (media_dir / f"shot-{i:02d}-audio.wav").exists():
                issues.append(f"shot-{i:02d}-audio.wav")
        if not (cp.parent / "episode-final.mp4").exists():
            issues.append("episode-final.mp4")

        if issues:
            print(f"  EP{ep_num:02d}: MISSING {', '.join(issues)}", flush=True)
            incomplete.append((ep_num, cp))
        else:
            print(f"  EP{ep_num:02d}: OK", flush=True)

    # Check for episodes with no run at all (1-50)
    guide = config.series.episode_guide or []
    for ep in guide:
        n = ep["episode_number"]
        if n not in ep_packs:
            missing_eps.append(n)

    if missing_eps:
        print(f"\n  No runs at all for episodes: {missing_eps}", flush=True)

    if not incomplete and not missing_eps:
        print(f"\nAll episodes complete!", flush=True)
        return

    if not fix:
        print(f"\n{len(incomplete)} incomplete, {len(missing_eps)} missing.", flush=True)
        print(f"Run with --fix to regenerate missing assets.", flush=True)
        return

    # Fix: regenerate missing assets for incomplete episodes
    print(f"\n=== Fixing {len(incomplete)} incomplete episodes (workers={workers}) ===\n", flush=True)
    _print_lock = threading.Lock()
    fixed = 0
    failed = 0

    def _fix_one(ep_num: int, cp: Path) -> None:
        with _print_lock:
            print(f"  [FIX] EP{ep_num:02d} regenerating missing assets…", flush=True)
        pipeline.generate_assets(cp)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fix_one, ep_num, cp): ep_num for ep_num, cp in incomplete}
        for future in as_completed(futures):
            ep_num = futures[future]
            try:
                future.result()
                fixed += 1
                with _print_lock:
                    print(f"  [FIX] EP{ep_num:02d} OK", flush=True)
            except Exception as exc:
                failed += 1
                with _print_lock:
                    print(f"  [FIX] EP{ep_num:02d} FAILED: {exc}", flush=True)

    print(f"\n=== Fix done: {fixed} fixed, {failed} still broken ===", flush=True)
    if missing_eps:
        print(f"Episodes with no runs need run-series: {missing_eps}", flush=True)
