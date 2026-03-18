# 2otbo5ly Viral Factory

`viral_factory` is a `GCP/Vertex AI` content pipeline for pre-launch TikTok growth. It is built for an Egyptian-Arabic cartoon series around `2otbo5ly`, with continuity memory across episodes.

Each run does four things:

1. Searches the web with Google Search grounding.
2. Generates a pool of episode concepts for the current trend window.
3. Refines each script `10` times and each Veo video plan `10` times.
4. Writes captions, overlays, and optional generation requests for `Imagen`, `Veo`, and `Gemini-TTS`.

The current default world is:

- series title: `الشيف اللي اترفد`
- visual mode: `cartoon_hybrid`
- tone: Egyptian social comedy
- food look: stylized but craveable

## Output

Each run creates a folder under [`runs`](/Users/barbary/Documents/New%20project/runs) with:

- `research.json`
- `series-state-before.json`
- `concepts.json`
- one folder per selected concept with:
  - `script-00-initial.json`
  - `script-01.json` through `script-10.json`
  - `script-final.json`
  - `video-plan-00-initial.json`
  - `video-plan-01.json` through `video-plan-10.json`
  - `video-plan-final.json`
  - `captions.json`
  - `content-pack.json`
  - `media-manifest.json` if generation is enabled
- `series-state-after.json`

## Setup

Install the Google Cloud CLI on macOS if `gcloud` is missing:

```bash
brew install --cask gcloud-cli
```

Create the local Python environment:

```bash
./scripts/bootstrap-mac.sh
```

Then create a venv, install dependencies, and authenticate with Vertex AI:

```bash
cd "/Users/barbary/Documents/New project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_CLOUD_VIDEO_LOCATION="us-central1"
```

If you do not want to install `gcloud`, you can use a service account JSON key instead:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_CLOUD_VIDEO_LOCATION="us-central1"
```

If you want generated video output to land in Cloud Storage, update `assets.video_output_gcs_uri` in [`configs/2otbo5ly.json`](/Users/barbary/Documents/New%20project/configs/2otbo5ly.json).

## Run

Plan only:

```bash
python3 -m viral_factory --config configs/2otbo5ly.json plan \
  --topic "Episode 1 of the fired chef cartoon series in Egypt"
```

Plan and generate media:

1. Enable the relevant asset flags in [`configs/2otbo5ly.json`](/Users/barbary/Documents/New%20project/configs/2otbo5ly.json).
2. Run:

```bash
python3 -m viral_factory --config configs/2otbo5ly.json run \
  --topic "Ramadan food struggle episode for the fired chef cartoon series"
```

Generate assets later from an existing content pack:

```bash
python3 -m viral_factory --config configs/2otbo5ly.json generate \
  --pack runs/<timestamp-topic>/<concept-folder>/content-pack.json
```

## Cartoon Series Mode

When `series.enabled` is `true`, the pipeline:

1. Loads a continuity file before ideation.
2. Feeds the series bible and recent episode memory into every prompt.
3. Generates scripts with `episode_summary`, `cliffhanger`, and `continuity_updates`.
4. Generates video plans with `style_guardrails`, `character_sheet_prompt_en`, `location_sheet_prompt_en`, and per-shot `continuity_lock`.
5. Updates the continuity state after each run.

If image generation is enabled in series mode, it also creates:

- `character-sheet.png`
- `location-sheet.png`
- `cover.png`

## Notes

- This repo is GCP-only. It does not use OpenAI APIs.
- Veo prompts are written in English because Google documents that Veo responds more reliably to English prompts.
- Arabic captions and meme text should be added in post, not baked into the generated scene.
- `gcloud` and credentials are not validated inside this repo until you run it locally.

## Verification

```bash
python3 -m compileall viral_factory tests
python3 -m unittest discover -s tests
python3 -m viral_factory --help
```
