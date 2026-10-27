#!/usr/bin/env bash
# Download CC0-licensed scene images from Pexels for use as I2V first frames.
#
# Pexels license (https://www.pexels.com/license/):
#   - Free, commercial and personal use.
#   - No attribution required (but appreciated).
#   - Cannot sell unaltered copies, cannot imply endorsement, cannot use for misleading content.
#
# This script fetches NON-PEOPLE scene imagery (scenery, animals, objects, abstract). It is NOT
# intended for human-face data. Your Tier-1 experiment doesn't need faces.
#
# Setup:
#   1. Register at https://www.pexels.com/api/  (free, 200 req/hour)
#   2. cp .env.example .env  and fill PEXELS_API_KEY
#   3. bash scripts/04_crawl_pexels.sh
#
# Output: data/pexels_cc0/<query>/*.jpg + data/pexels_cc0/manifest.json
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

# Load .env
if [[ -f .env ]]; then
  set -o allexport
  # shellcheck disable=SC1091
  source .env
  set +o allexport
fi
if [[ -z "${PEXELS_API_KEY:-}" ]]; then
  echo "ERROR: PEXELS_API_KEY not set."
  echo "       1) go to https://www.pexels.com/api/ and get a free key"
  echo "       2) cp .env.example .env and paste the key"
  exit 1
fi

PER_QUERY="${PER_QUERY:-20}"           # images per search term
OUT_DIR="${OUT_DIR:-data/pexels_cc0}"
mkdir -p "$OUT_DIR"

# Curated non-people VBench-I2V-like scene queries. All should map well to I2V motion priors.
QUERIES=(
  "mountain landscape"
  "forest river"
  "ocean waves"
  "desert sand dunes"
  "aurora northern lights"
  "waterfall rainforest"
  "snowy mountain peak"
  "cherry blossom tree"
  "autumn forest path"
  "urban street night"
  "neon city skyline"
  "cafe interior window"
  "library bookshelf"
  "art studio sunlight"
  "cat sitting windowsill"
  "golden retriever running"
  "red fox snow"
  "owl in forest"
  "jellyfish underwater"
  "koi fish pond"
  "butterfly flower garden"
  "hot air balloon sunrise"
  "sailboat open sea"
  "steam locomotive"
  "vintage car countryside"
  "glass sphere refraction"
  "oil paint splash"
  "smoke abstract black"
  "candle flame dark"
  "lava flow"
)

python3 - "$PEXELS_API_KEY" "$OUT_DIR" "$PER_QUERY" "${QUERIES[@]}" <<'PY'
import json, os, sys, time, urllib.request, urllib.parse
from pathlib import Path

api_key, out_root, per_query, *queries = sys.argv[1:]
per_query = int(per_query)
out_root = Path(out_root)
manifest = []

for q in queries:
    q_safe = q.replace(" ", "_")
    q_dir = out_root / q_safe
    q_dir.mkdir(parents=True, exist_ok=True)
    url = ("https://api.pexels.com/v1/search?"
           + urllib.parse.urlencode({
               "query": q,
               "per_page": per_query,
               "orientation": "landscape",
               "size": "medium",
           }))
    req = urllib.request.Request(url, headers={"Authorization": api_key})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        print(f"[pexels] FAIL {q}: {e}")
        continue
    photos = data.get("photos", [])
    print(f"[pexels] {q}: got {len(photos)} photos")
    for i, p in enumerate(photos):
        src = p["src"]["large"]            # ~940x650 jpg, plenty for 512x320 I2V
        fname = q_dir / f"{p['id']}.jpg"
        if fname.exists():
            manifest.append({"query": q, "pexels_id": p["id"], "file": str(fname),
                             "photographer": p.get("photographer"), "url": p["url"]})
            continue
        try:
            urllib.request.urlretrieve(src, str(fname))
            manifest.append({
                "query": q, "pexels_id": p["id"], "file": str(fname),
                "photographer": p.get("photographer"), "url": p["url"],
                "pexels_src": src,
            })
        except Exception as e:
            print(f"[pexels]   skip {p['id']}: {e}")
    time.sleep(0.5)   # polite throttle

(out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
(out_root / "LICENSE_NOTE.md").write_text(
    "# Pexels CC0 image dump\n\n"
    "All images here were downloaded via the Pexels API under the Pexels license\n"
    "(https://www.pexels.com/license/). Free for commercial and personal use, no\n"
    "attribution required. See manifest.json for original URLs and photographer\n"
    "credits.\n"
)
print(f"[pexels] done. {len(manifest)} images in {out_root}")
PY

echo
echo "[done] images in $OUT_DIR"
echo "       use them in Tier 1 by pointing data/prompts/*.json 'image' fields to these paths"
echo "       (OPTIONAL — Tier 1 default is to generate first frames with SD3.5-M, not Pexels)"
