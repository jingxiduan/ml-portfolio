#!/bin/bash
# Rebuild the >100 MB model weights that GitHub cannot store as single files.
# They are kept as 90 MB chunks in backups/weights_chunks/ (with sha256 manifest).
#
#   bash scripts/restore_chunks.sh
#
# Restores into third_party/JoyVASA/pretrained_weights/... (the paths the code uses).
set -euo pipefail
cd "$(dirname "$0")/.."
D=backups/weights_chunks
[ -d "$D" ] || { echo "[err] $D missing" >&2; exit 1; }

python3 - "$D" <<'PY'
import os, sys, glob, collections
d = sys.argv[1]
groups = collections.defaultdict(list)
for p in sorted(glob.glob(os.path.join(d, "*.part*"))):
    base = p[:p.index(".part")]
    groups[base].append(p)
for base, parts in sorted(groups.items()):
    rel = os.path.basename(base).replace("_", "/")
    # original path: third_party/JoyVASA/pretrained_weights/<...>
    rel = rel.replace("third_party/JoyVASA/pretrained_weights/", "")
    out = os.path.join("third_party/JoyVASA/pretrained_weights", rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print("  [skip] %s" % out); continue
    with open(out, "wb") as fo:
        for p in sorted(parts):
            with open(p, "rb") as fi:
                while True:
                    b = fi.read(1 << 22)
                    if not b:
                        break
                    fo.write(b)
    print("  [ok] %s (%.0f MB)" % (out, os.path.getsize(out) / 1048576))
PY
echo "[verify] sha256sum -c $D/MANIFEST.sha256"
(cd "$D" && sha256sum -c MANIFEST.sha256 | tail -3)
