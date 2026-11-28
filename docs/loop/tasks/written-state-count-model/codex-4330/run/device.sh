#!/usr/bin/env bash
# Run on the leased accelerator from the sealed input directory.
set -euo pipefail
umask 077
phase=${1:?fixture, fit or score}
cd /workspace/whole4330
device=./whole-turn-device
image=whole-turn.cubin
case "$phase" in
  fixture)
    "$device" fixture "$image" roots token-classes.u8 fixture
    ;;
  fit)
    test -s fit-verdict.md && grep -Fxq 'TEST FITNESS: FIT' fit-verdict.md
    sha256sum --strict --quiet -c STATIC.sha256
    test ! -e eval.jsonl
    for run in fit1 fit2; do
      "$device" fit "$image" roots token-classes.u8 initial.i16le train.jsonl "$run"
    done
    cmp fit1/weights.i16le fit2/weights.i16le
    ;;
  score)
    sha256sum --strict --quiet -c STATIC.sha256
    sha256sum --strict --quiet -c EVAL.sha256
    cmp fit1/weights.i16le fit2/weights.i16le
    "$device" score "$image" roots token-classes.u8 initial.i16le fit1/weights.i16le eval.jsonl scored
    ;;
  *) exit 2 ;;
esac
