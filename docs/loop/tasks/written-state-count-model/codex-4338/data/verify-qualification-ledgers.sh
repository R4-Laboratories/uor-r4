#!/usr/bin/env bash
set -euo pipefail

unit=$(realpath "${1:?private #4338 TRAIN source and decisions root}")
cd "$unit"
sha256sum --quiet --strict -c - <<'APPROVED_LEDGER_SHA256'
585b0d0f264732e7e4cf8d524096384c11f83cf23877ed9b622dd39a704cd9aa  positive-even/decisions.jsonl
78a0b40bda7734d923fc3300a7f08d583fb12ddb8ba6be70357b82aab690edec  positive-odd/decisions.jsonl
19fc9a489e55ab4f9edc72e0d73fcf0931cd3f9060288679d23970bb0d95e3f7  answer-a/decisions.jsonl
b99b9017c8ca5da563e683e122655e59e06c15fb2a01cf57cd319f704c047e66  answer-b/decisions.jsonl
ab810a3526e7f7ce8d4120c77009bce698e00c0e39de10a309757c0fa875fd59  answer-c/decisions.jsonl
APPROVED_LEDGER_SHA256
