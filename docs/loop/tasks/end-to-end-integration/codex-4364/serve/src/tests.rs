#[test]
fn rejects_unbound_training_receipts() -> Result<()> {
    let directory =
        std::env::temp_dir().join(format!("opposed-receipt-test-{}", std::process::id()));
    fresh_dir(&directory)?;
    let path = directory.join("receipt.json");
    let path_str = path.to_str().ok_or("non-UTF8 fixture path")?;
    let training_pin = "b".repeat(64);
    let finite_pin = "c".repeat(64);
    let mut fixture = json!({
        "protocol": "uor-opening-real-single-update-v1", "result": "MEASURED_UPDATE",
        "parent_sha256": PARENT, "roster_sha256": ROSTER, "cubin_sha256": training_pin,
        "finite_receipt_sha256": finite_pin, "step_before":3333, "step_after":3334,
        "updates_per_arm": 1, "labels_per_arm": 167,
        "control": "physical original ABI #4361", "historical_control_cubin_sha256": HISTORICAL_CONTROL_CUBIN,
        "baseline_forward_parity_transitions": 3938, "batch_lengths": [980, 1018, 933, 1011],
        "opening_labels": [953, 957, 887, 978], "margin_words": 256,
        "different_gradient_words": 1, "sham_gain_q15": 0, "active_gain_q15": 32768,
        "sham_words": {"gradient_le_sha256":"1".repeat(64), "weights_le_sha256":"2".repeat(64),
            "adam_m_le_sha256":"3".repeat(64), "adam_v_le_sha256":"4".repeat(64)},
        "active_words": {"gradient_le_sha256":"5".repeat(64), "weights_le_sha256":"6".repeat(64),
            "adam_m_le_sha256":"7".repeat(64), "adam_v_le_sha256":"8".repeat(64)},
        "sham_checkpoint_sha256": "d".repeat(64), "active_checkpoint_sha256":"e".repeat(64),
    });
    let bytes = serde_json::to_vec(&fixture)?;
    let pin = sha256(&bytes);
    assert!(training_receipt(path_str, &pin, &training_pin, &finite_pin).is_err());
    write_new(&path, &bytes)?;
    assert!(training_receipt(path_str, &pin, &training_pin, &finite_pin).is_ok());
    assert!(training_receipt(path_str, &training_pin, &training_pin, &finite_pin).is_err());
    assert!(training_receipt(path_str, &pin, &finite_pin, &finite_pin).is_err());
    assert!(training_receipt(path_str, &pin, &training_pin, &training_pin).is_err());
    fixture
        .as_object_mut()
        .ok_or("invalid receipt fixture")?
        .remove("active_checkpoint_sha256");
    let invalid = serde_json::to_vec(&fixture)?;
    let invalid_path = directory.join("invalid.json");
    write_new(&invalid_path, &invalid)?;
    assert!(training_receipt(
        invalid_path.to_str().ok_or("invalid path")?,
        &sha256(&invalid),
        &training_pin,
        &finite_pin
    )
    .is_err());
    fs::remove_dir_all(directory)?;
    Ok(())
}

#[test]
fn grades_first_wrong_and_eos_separately() {
    let expected = [7, 8, EOS];
    assert_eq!(
        outcome(&expected, &expected, true).verdict,
        "exact_through_eos"
    );
    assert_eq!(
        outcome(&[7, EOS], &expected, true).first_mismatch_index,
        Some(1)
    );
    assert_eq!(
        outcome(&[7, 8, 9, EOS], &expected, true).first_mismatch_index,
        Some(2)
    );
    assert_eq!(
        outcome(&[7, 9, EOS], &expected, true).first_mismatch_index,
        Some(1)
    );
    assert_eq!(outcome(&[7, 8], &expected, false).verdict, "censored");
}

#[test]
fn ranks_ties_and_excludes_masked_slots() -> Result<()> {
    let mut scores = [-12; 64];
    scores[..4].copy_from_slice(&[99, 17, 17, 16]);
    let first = compare_rank(&scores, 1, |i| i != 0)?;
    let second = compare_rank(&scores, 2, |i| i != 0)?;
    assert_eq!(first["rank"], 1);
    assert_eq!(first["gold_minus_rival_q15"], 0);
    assert_eq!(first["gold_elected"], true);
    assert_eq!(first["strict_gold_win"], false);
    assert_eq!(second["rank"], 2);
    assert_eq!(second["rival_slot"], 1);
    assert!(compare_rank(&scores, 0, |i| i != 0).is_err());
    Ok(())
}
