#!/usr/bin/env bash
# #4340's sealed build prefix reads its own digest under its previous filename.
set -euo pipefail
capsule=$(realpath "$1")
cd "$capsule"
printf '%s  INPUTS.sha256\n' 09c93ef41bbe46385f40cb13aa1bf67906ed3b95494ad55ccc8f4545f0be48cf | sha256sum --strict --quiet -c -
sha256sum --strict --quiet -c INPUTS.sha256
sha256sum --strict --quiet -c SOURCE-PINS.sha256
cd source
ln aot-guard.sh launch.sh
cmp aot-guard.sh launch.sh
printf '%s  launch.sh\n' 054a8f0d0ad31c3c48ab972fe4efded449febfbcc811529bc4c6e9d1a1eb3309 | sha256sum --strict --quiet -c -
