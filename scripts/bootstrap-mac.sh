#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cat <<'EOF'

Local environment is ready.

Next:
1. Authenticate ADC if you have not already:
   gcloud auth application-default login
2. Set your real project id and quota project:
   export GOOGLE_CLOUD_PROJECT="your-real-project-id"
   export GOOGLE_CLOUD_LOCATION="global"
   export GOOGLE_CLOUD_VIDEO_LOCATION="us-central1"
   gcloud auth application-default set-quota-project "$GOOGLE_CLOUD_PROJECT"
3. Run the planner:
   python3 -m viral_factory --config configs/2otbo5ly.json plan --topic "Episode 1 of الشيف اللي اترفد"

EOF
