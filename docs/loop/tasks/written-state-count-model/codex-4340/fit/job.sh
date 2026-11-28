#!/usr/bin/env bash
# Invoke as a uor-pod run command on the assigned GPU.
set -euo pipefail
umask 077
export LC_ALL=C
export PATH="/usr/local/cuda/bin:$PATH"
if [[ $# != 5 ]]; then
    echo 'usage: job.sh LAUNCH ARCH_NUMBER INPUT_CAPSULE MANIFEST_DIGEST FRESH_JOB_ROOT' >&2
    exit 2
fi
launch=$(realpath "$1")
inputs=$(realpath "$3")
job=$(realpath -m "$5")
mkdir "$job"
cd "$job"
date -u +%FT%TZ > started-utc
status=0
/usr/bin/time -f '%e' -o job.wall_seconds \
    bash "$launch" "$2" "$inputs" "$job/measured" "$4" > job.stdout 2> job.stderr || status=$?
date -u +%FT%TZ > finished-utc
printf '%s\n' "$status" > job.exit.tmp
mv job.exit.tmp job.exit
exit "$status"
