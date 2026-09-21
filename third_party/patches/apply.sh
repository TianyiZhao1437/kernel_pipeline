#!/usr/bin/env bash
# Re-apply this repository's patches to the vendored flashinfer-bench tree.
#
# VENDOR.md's pin-update procedure replaces the vendored tree wholesale, which
# discards these patches. Run this afterwards. A rejected hunk is the signal
# that upstream moved the code the patch depends on -- resolve it by rereading
# upstream, not by forcing the patch.
#
#   ./third_party/patches/apply.sh           apply
#   ./third_party/patches/apply.sh --check   dry run, report only
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tree="$(dirname "$here")/flashinfer-bench"
check=""
[ "${1:-}" = "--check" ] && check="--check"

for p in "$here"/[0-9][0-9][0-9]-*.patch; do
    name="$(basename "$p")"
    if git -C "$tree" apply --reverse --check "$p" 2>/dev/null; then
        echo "already applied  $name"
        continue
    fi
    if git -C "$tree" apply $check "$p"; then
        echo "applied          $name"
    else
        echo "FAILED           $name" >&2
        exit 1
    fi
done
