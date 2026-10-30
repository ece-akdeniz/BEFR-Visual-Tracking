#!/usr/bin/env bash
# Build a clean public submission tree (fresh git history).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-/tmp/BEFR-Visual-Tracking}"

rm -rf "$OUT"
mkdir -p "$OUT"

rsync -a \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.pytest_cache' \
  --exclude '.agents' \
  --exclude 'befr_visual_tracking.egg-info' \
  --exclude 'lars-visualization' \
  --exclude 'config' \
  --exclude 'scripts' \
  --exclude 'rviz' \
  --exclude 'camera_model.ipynb' \
  --exclude 'VM_STARTUP.md' \
  --exclude 'presentation_anim (1).pptx' \
  --exclude 'presentation_anim' \
  "$ROOT/README.md" \
  "$ROOT/OFFLINE_RESULTS.md" \
  "$ROOT/requirements.txt" \
  "$ROOT/setup.py" \
  "$ROOT/.gitignore" \
  "$ROOT/befr_visual_tracking" \
  "$ROOT/dashboard" \
  "$ROOT/launch" \
  "$ROOT/tests" \
  "$ROOT/results" \
  "$ROOT/presentation" \
  "$OUT/"

# Matplotlib frame generator used for advanced slides
if [[ -f "$ROOT/presentation_anim/generate_advanced_animation_frames.py" ]]; then
  cp "$ROOT/presentation_anim/generate_advanced_animation_frames.py" \
    "$OUT/presentation/generate_advanced_frames.py"
fi

cat > "$OUT/.gitignore" <<'EOF'
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
.DS_Store
EOF

echo "Submission tree ready at: $OUT"
echo "Create public GitHub repo: BEFR Visual Tracking  (URL slug: BEFR-Visual-Tracking)"
du -sh "$OUT"
