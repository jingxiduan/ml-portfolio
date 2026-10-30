#!/usr/bin/env bash
# Backup helper for GPU windows whose disks are reclaimed on expiry.
# Run it the moment a training / render / data-prep job ends (or the guard stops it).
#
#   tools/backup_push.sh small "<commit message>" <path>...
#       Commit and push small files to the CURRENT branch.
#       Skips: files > 90MB, *.mp4, *.wav/*.flac/*.mp3, .venv, third_party, weights, __pycache__.
#       Skipped big files are listed so you can pack them with `big`.
#
#   tools/backup_push.sh big <tag> <path>...
#       Pack paths into <=90MB tar.gz parts and push them to the orphan branch backup/<tag>,
#       leaving main untouched. Use for mp4, audio subsets, large checkpoints, rendered outputs.
#       Restore: git checkout backup/<tag> && cat <prefix>.part-* | tar xzf -
#
#   tools/backup_push.sh status
#       Show how much unpushed / unbacked data there is.
#
# Auth: whatever `git push` already uses on this machine (SSH key). The script never reads or stores credentials.
set -euo pipefail

MAX_BYTES=$((90 * 1024 * 1024))
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
cmd="${1:-}"; shift || true

push_retry() {  # push_retry <refspec...>
  local n=0
  until git push "$@"; do
    n=$((n + 1))
    [ "$n" -ge 4 ] && { echo "[backup] push failed after $n tries" >&2; return 1; }
    echo "[backup] push failed, retry $n/3 in 15s" >&2
    sleep 15
  done
}

case "$cmd" in
  small)
    msg="${1:?commit message required}"; shift
    [ "$#" -ge 1 ] || { echo "give at least one path" >&2; exit 2; }
    skipped=()
    added=0
    while IFS= read -r -d '' f; do
      case "$f" in
        */.venv/*|*/third_party/*|*/weights/*|*/__pycache__/*|*.mp4|*.wav|*.flac|*.mp3) continue ;;
      esac
      sz=$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")
      if [ "$sz" -gt "$MAX_BYTES" ]; then skipped+=("$f ($((sz / 1048576))MB)"); continue; fi
      git add -f -- "$f"; added=$((added + 1))
    done < <(find "$@" -type f -print0)
    echo "[backup] staged $added files"
    if [ "${#skipped[@]}" -gt 0 ]; then
      echo "[backup] NOT pushed (too big or media) -> use 'big':"
      printf '   %s\n' "${skipped[@]}"
    fi
    if git diff --cached --quiet; then echo "[backup] nothing new to commit"; else git commit -q -m "$msg"; fi
    br="$(git rev-parse --abbrev-ref HEAD)"
    git pull --rebase --autostash -q origin "$br" || true
    push_retry origin "$br"
    echo "[backup] pushed $(git rev-parse --short HEAD) on $br"
    ;;
  big)
    tag="${1:?tag required}"; shift
    [ "$#" -ge 1 ] || { echo "give at least one path" >&2; exit 2; }
    ts="$(date +%Y%m%d-%H%M%S)"
    br="backup/${tag}"
    wt="$(mktemp -d)"
    trap 'git worktree remove --force "$wt" 2>/dev/null || true' EXIT
    git fetch -q origin "$br" 2>/dev/null && have_remote=1 || have_remote=0
    if [ "$have_remote" = 1 ]; then
      git worktree add -q -B "$br" "$wt" "origin/$br"
    else
      git worktree add -q --detach "$wt"
      (cd "$wt" && git checkout -q --orphan "$br" && git rm -rq --cached . >/dev/null 2>&1; find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +)
    fi
    prefix="${tag}_${ts}.tgz.part-"
    tar czf - -- "$@" | split -b 90m - "$wt/$prefix"
    (cd "$wt" && sha256sum "$prefix"* > "SHA256_${tag}_${ts}.txt" 2>/dev/null || shasum -a 256 "$prefix"* > "SHA256_${tag}_${ts}.txt")
    printf 'paths: %s\nrestore: cat %s* | tar xzf -\n' "$*" "$prefix" > "$wt/RESTORE_${tag}_${ts}.txt"
    nparts=$(ls "$wt/$prefix"* | wc -l | tr -d ' ')
    (cd "$wt" && git add -A && git commit -q -m "backup ${tag} ${ts}: $*" && push_retry origin "$br")
    echo "[backup] pushed branch $br, ${nparts} part(s): ${prefix}*"
    ;;
  status)
    echo "unpushed commits on current branch:"; git --no-pager log --oneline '@{u}..' 2>/dev/null || echo "  (no upstream)"
    echo "uncommitted changes:"; git status --short | head -20
    echo "large untracked/ignored outputs (>90MB):"
    find . -path ./.git -prune -o -path '*/.venv' -prune -o -path '*/third_party' -prune -o -path '*/weights' -prune -o \
      -type f -size +90M -print 2>/dev/null | head -20
    ;;
  *)
    sed -n 2,22p "$0"; exit 2 ;;
esac
