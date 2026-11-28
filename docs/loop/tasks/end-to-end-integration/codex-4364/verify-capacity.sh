#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
    echo 'usage: verify-capacity.sh RESTORED_DEVICE_DIRECTORY TRUSTED_RECEIPT_SHA256' >&2
    exit 2
fi
cd "$1"
sha256sum --strict --status -c members.sha256
printf '%s  capacity.json\n' "$2" | sha256sum --strict --status -c -
test "$(cat status)" = 0
uuid=$(awk -F ', ' 'NR == 2 {print $2}' device.csv | sed 's/^GPU-//;s/-//g' | tr '[:upper:]' '[:lower:]')
[[ "$uuid" =~ ^[0-9a-f]{32}$ ]]
jq -e --arg uuid "$uuid" '
  .protocol == "uor-opening-capacity-v1" and .verdict == "PASS" and .gpu_uuid == $uuid
  and .parent_sha256 == "20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589"
  and .historical_cubin_sha256 == "d7594dab97fd490bee2cad35d595a12054f64b2cdebe1eb628ac96ac8f56328a"
  and .cubin_sha256 == "9b8185fd00e3d7896569476b1160102dc0305ddc24fe29d63cb0bd231113d533"
  and .batch_lengths == [980,1018,933,1011] and .opening_labels == [953,957,887,978]
  and .labels_per_arm == 167 and .equal_writer_labels == 167
  and .transitions == 3938 and .workspace_transitions_per_row == 1017
  and .dim == 512 and .layers == 6 and .ffn_branches == 4 and .pkm_sub_keys == 64
  and .pkm_key_dim == 32 and .threads_per_block == 256
  and (.opening_readback | map(.opening)) == .opening_labels
  and .visits_per_site == 4 and .live_coarse_visits > 0 and .live_fine_visits > 0
  and (.active_trace | length) == 40 and (.sham_trace | length) == 40
  and .margin_words == 256 and .active_gain_q15 == 32768 and .sham_gain_q15 == 0
  and .step_before == 3333 and .step_after == 3334 and .updates_per_arm == 1
  and .equal_gradient_words > 0 and .equal_gradient_words == .equal_weight_words
  and .equal_weight_words == .equal_adam_m_words and .equal_weight_words == .equal_adam_v_words
  and .equal_forward_workspace_words_per_comparison == 4 * 19 * 1017 * 512
  and .different_coarse_gradient_words > 0 and .different_tied_embedding_gradient_words > 0
  and .different_treatment_weight_words > 0
  and .original_and_zero_checkpoint_sha256 != .active_checkpoint_sha256
' capacity.json > /dev/null
