#!/bin/bash
set -euo pipefail
umask 077
cd /root/j4359
echo 'ea4da3e3ebb3bc8b876f0a8c2f8e91bb79c18fc1102ecc5761453e86dbbda31f  joint-proof' | sha256sum --strict -c -
echo 'd20043ebabcfc33a7b9fafbbf3b65bb75cbd2ae97ecfb9104f7519ff2b4b2107  protected.tsv' | sha256sum --strict -c -
echo '44af3ca6f2fe91c76d32c5e377a2f6d66f4c58381ebc414a1af5885cafcae724  sentinel.tsv' | sha256sum --strict -c -
dst=/workspace/attempt01
mkdir "$dst"
./joint-proof protected.tsv > "$dst/protected-raw.tsv"
./joint-proof --sentinel sentinel.tsv > "$dst/sentinel-raw.tsv"
cd "$dst"
sha256sum protected-raw.tsv sentinel-raw.tsv > SHA256
touch done
