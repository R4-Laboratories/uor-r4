//! Integration tests for the pure-integer `afflom/vRAM` geometric 4 KiB page store.

use hologram_forge_core::vram_store::{
    ffi_create, ffi_destroy, ffi_discard_pages, ffi_read_page, ffi_read_range, ffi_stats_json_bytes,
    ffi_write_page, ffi_write_range, run_from_json, VramStore, ERR_CAPACITY, ERR_INVALID_ARG,
    ERR_INVALID_HANDLE, ERR_OUT_OF_RANGE, PAGE, TAG_CORRECTED, TAG_FILL, TAG_PACKED, TAG_REPEATED,
    TAG_RESIDUAL, TAG_SELECTOR, TAG_SHARED, TAG_SPARSE,
};
use serde_json::json;

const SIDE: usize = 64;

fn prng_page(seed: u32) -> [u8; PAGE] {
    let mut out = [0u8; PAGE];
    let mut s = seed ^ 0x9e37_79b9;
    for b in &mut out {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        *b = (s >> 24) as u8;
    }
    out
}

#[test]
fn all_seven_representations_round_trip_losslessly_and_save_memory() {
    let mut store = VramStore::new(256, 4 * 1024 * 1024).expect("create vram store");

    // 1. Uniform Fill
    let p_fill = [0xA5u8; PAGE];
    assert_eq!(store.write_page(0, &p_fill).unwrap(), TAG_FILL);

    // 2a. Repeated (power-of-two factor)
    let p_rep_pow2: [u8; PAGE] = std::array::from_fn(|i| ((i % 32).wrapping_mul(7)) as u8);
    assert_eq!(store.write_page(1, &p_rep_pow2).unwrap(), TAG_REPEATED);

    // 2b. Repeated (non-power-of-two shifted displacement)
    let pattern: Vec<u8> = (0u8..23).map(|x| x.wrapping_mul(11) ^ 0x5A).collect();
    let p_rep_shift: [u8; PAGE] = std::array::from_fn(|i| pattern[i % pattern.len()]);
    assert_eq!(store.write_page(2, &p_rep_shift).unwrap(), TAG_REPEATED);

    // 3. Sparse (base + bounded exceptions)
    let mut p_sparse = [0x33u8; PAGE];
    for k in 0..10 {
        p_sparse[k * 389 + 5] = (k as u8).wrapping_mul(19) ^ 0x81;
    }
    assert_eq!(store.write_page(3, &p_sparse).unwrap(), TAG_SPARSE);

    // 4. Selector (exact 64x64 two-symbol-per-row with shared binary column mask)
    let p_selector: [u8; PAGE] = std::array::from_fn(|i| {
        let r = (i / SIDE) as u8;
        let c = i % SIDE;
        if c % 5 == 0 || c == 61 {
            r.wrapping_add(17)
        } else {
            240u8.wrapping_sub(r)
        }
    });
    assert_eq!(store.write_page(4, &p_selector).unwrap(), TAG_SELECTOR);

    // 5. Corrected (near-Selector page with few point edits)
    let mut p_corrected = p_selector;
    for idx in [11, 519, 1999, 4094] {
        p_corrected[idx] ^= 0x7E;
    }
    assert_eq!(store.write_page(5, &p_corrected).unwrap(), TAG_CORRECTED);

    // 6. Packed (<= 16 unique values, non-periodic high local entropy defeating Selector/Repeated)
    let mut p_packed = [0u8; PAGE];
    let mut s_pack: u32 = 0x2468_ace1;
    for b in &mut p_packed {
        s_pack ^= s_pack << 13;
        s_pack ^= s_pack >> 17;
        s_pack ^= s_pack << 5;
        *b = (((s_pack >> 24) & 0x0f) as u8).wrapping_mul(13) ^ 0x42;
    }
    assert_eq!(store.write_page(6, &p_packed).unwrap(), TAG_PACKED);

    // 7a. Unique Literal base -> stored in pool as Shared with 1 owner
    let p_lit = prng_page(1);
    assert_eq!(store.write_page(7, &p_lit).unwrap(), TAG_SHARED);

    // 7b. Exact duplicate literal -> reuses Shared base with 0 extra pool allocation
    assert_eq!(store.write_page(8, &p_lit).unwrap(), TAG_SHARED);

    // 7c. Residual -> borrows shared base + sparse byte edits outside fingerprint sample windows
    let mut p_res = p_lit;
    for off in [13usize, 1024, 1500, 3999] {
        p_res[off] ^= 0xFF;
    }
    assert_eq!(store.write_page(9, &p_res).unwrap(), TAG_RESIDUAL);

    let expected = [
        &p_fill,
        &p_rep_pow2,
        &p_rep_shift,
        &p_sparse,
        &p_selector,
        &p_corrected,
        &p_packed,
        &p_lit,
        &p_lit,
        &p_res,
    ];
    let mut out = [0u8; PAGE];
    for (idx, exp) in expected.iter().enumerate() {
        store.read_page(idx, &mut out).expect("read back");
        assert_eq!(&out, *exp, "lossless mismatch at page {idx}");
    }

    let stats = store.stats_json();
    assert_eq!(stats["occupied_pages"], 10);
    assert_eq!(stats["unique_literals"], 1);
    assert_eq!(stats["shared_hits"], 1);
    assert_eq!(stats["residual_hits"], 1);
    assert!(
        stats["savings_permille"].as_u64().unwrap() > 650,
        "expected >65% geometric compression savings, got {stats}"
    );
}

#[test]
fn cross_page_ranges_sparse_directory_and_discard_reclaim_literals() {
    // 256 MiB logical sparse space (`65536` pages across leaves 0 and 3)
    let mut store = VramStore::new(65_536, 1024 * 1024).expect("create sparse store");
    let lit = prng_page(42);
    let _ = store.write_page(0, &lit).unwrap();
    let _ = store.write_page(50_000, &lit).unwrap();

    // Unaligned cross-page write across boundary of pages 10 and 11.
    let mut initial_p10 = [0xAAu8; PAGE];
    initial_p10[4090..].copy_from_slice(&[1, 2, 3, 4, 5, 6]);
    store.write_page(10, &initial_p10).unwrap();
    store.write_page(11, &[0xBBu8; PAGE]).unwrap();

    let patch: Vec<u8> = (0u8..120).map(|x| x.wrapping_mul(3) ^ 0x1D).collect();
    store
        .write_range((10 * PAGE + 4050) as u64, &patch)
        .expect("cross-page write");

    // Unaffected prefix in p10 and suffix in p11 must remain intact.
    let mut p10_after = [0u8; PAGE];
    let mut p11_after = [0u8; PAGE];
    store.read_page(10, &mut p10_after).unwrap();
    store.read_page(11, &mut p11_after).unwrap();
    assert!(p10_after[..4050].iter().all(|&b| b == 0xAA));
    assert!(p11_after[74..].iter().all(|&b| b == 0xBB));

    let mut readback = vec![0u8; patch.len()];
    store
        .read_range((10 * PAGE + 4050) as u64, &mut readback)
        .expect("cross-page read");
    assert_eq!(readback, patch);

    // Discarding page 0 leaves page 50_000 owning the literal; discarding 50_000 frees pool entry.
    assert_eq!(store.discard_pages(0, 1).unwrap(), 1);
    assert_eq!(store.stats_json()["unique_literals"], 1);
    assert_eq!(store.discard_pages(50_000, 1).unwrap(), 1);
    assert_eq!(store.stats_json()["unique_literals"], 0);
    assert_eq!(store.stats_json()["pool_bytes"], 0);

    // Discarded pages read back as zeros.
    let mut zero_check = [0xFFu8; PAGE];
    store.read_page(0, &mut zero_check).unwrap();
    assert!(zero_check.iter().all(|&b| b == 0));
}

#[test]
fn capacity_limit_rolls_back_atomically_and_ffi_guards_handles() {
    let mut store = VramStore::new(16, 2 * PAGE).expect("8 KiB cap store");
    let lit1 = prng_page(100);
    let lit2 = prng_page(200);
    let lit3 = prng_page(300);

    assert_eq!(store.write_page(0, &lit1).unwrap(), TAG_SHARED);
    assert_eq!(store.write_page(1, &lit2).unwrap(), TAG_SHARED);
    // Single-page overflow preserves store untouched.
    assert_eq!(store.write_page(2, &lit3), Err(ERR_CAPACITY));
    assert_eq!(store.stats_json()["occupied_pages"], 2);

    // Cross-page write that fits page 2 (Repeated) and overflows page 3 (unique literal) rolls back page 2!
    let mut two_page_payload = vec![0x5Eu8; PAGE * 2];
    two_page_payload[PAGE..].copy_from_slice(&lit3);
    assert_eq!(
        store.write_range((2 * PAGE) as u64, &two_page_payload),
        Err(ERR_CAPACITY)
    );
    let mut check_p2 = [0xFFu8; PAGE];
    store.read_page(2, &mut check_p2).unwrap();
    assert!(
        check_p2.iter().all(|&b| b == 0),
        "failed cross-page write must roll back partial pages"
    );

    // FFI handle lifecycle & error codes
    let handle = ffi_create(8, 16 * PAGE as i32);
    assert!(handle > 0);
    assert_eq!(ffi_write_page(handle, 0, &[0x11u8; PAGE]), i32::from(TAG_FILL));
    assert_eq!(ffi_write_page(handle, 99, &[0x11u8; PAGE]), ERR_OUT_OF_RANGE);
    assert_eq!(ffi_write_page(handle, 0, &[0u8; 10]), ERR_INVALID_ARG);
    let mut out = [0u8; PAGE];
    assert_eq!(ffi_read_page(handle, 0, &mut out), 0);
    assert!(out.iter().all(|&b| b == 0x11));
    assert_eq!(ffi_discard_pages(handle, 0, 1), 1);
    assert!(ffi_stats_json_bytes(handle).is_some());
    assert_eq!(ffi_destroy(handle), 0);
    assert_eq!(ffi_destroy(handle), ERR_INVALID_HANDLE);
    assert_eq!(ffi_read_range(handle, 0, &mut out), ERR_INVALID_HANDLE);
    assert_eq!(ffi_write_range(handle, 0, &out), ERR_INVALID_HANDLE);

    // JSON self_test executes cleanly and verifies all representation tags.
    let st = run_from_json(&json!({ "action": "self_test" }));
    assert_eq!(st["ok"], true, "vram self_test failed: {st}");
    assert_eq!(st["tags"], st["expected_tags"]);
}

#[test]
fn sole_owner_overwrite_in_place_range_write_and_json_handle_guards() {
    // Tight 1-page capacity: sole-owner literal at page 0 can be overwritten cleanly by Fill(0)
    // and mutated in-place via write_range without 4107-byte Residual anti-compression.
    let mut tight = VramStore::new(4, PAGE).expect("tight store");
    let lit1 = prng_page(555);
    assert_eq!(tight.write_page(0, &lit1).unwrap(), TAG_SHARED);
    assert_eq!(tight.write_page(0, &[0u8; PAGE]).unwrap(), TAG_FILL);
    assert_eq!(tight.write_page(0, &lit1).unwrap(), TAG_SHARED);

    tight
        .write_range(10, &[0xFFu8])
        .expect("sole-owner in-place edit fits within 1-page capacity");
    let mut read_modified = [0u8; PAGE];
    tight.read_page(0, &mut read_modified).unwrap();
    let mut exp_modified = lit1;
    exp_modified[10] = 0xFF;
    assert_eq!(read_modified, exp_modified);
    assert_eq!(tight.stats_json()["residual_hits"], 0);

    // Failed write_range restores writes/shared_hits/residual_hits as well as page contents.
    let mut two_cap = VramStore::new(8, 2 * PAGE).expect("two page cap");
    assert_eq!(two_cap.write_page(0, &lit1).unwrap(), TAG_SHARED);
    let writes_before = two_cap.stats_json()["writes"].as_u64().unwrap();
    let mut partial_plus_overflow = Vec::with_capacity(3 * PAGE);
    partial_plus_overflow.extend_from_slice(&lit1); // shared_hit on p1
    partial_plus_overflow.extend_from_slice(&prng_page(777)); // unique -> ok on p2 (backing = 2*PAGE)
    partial_plus_overflow.extend_from_slice(&prng_page(888)); // overflows on p3!
    assert_eq!(
        two_cap.write_range(PAGE as u64, &partial_plus_overflow),
        Err(ERR_CAPACITY)
    );
    assert_eq!(two_cap.stats_json()["writes"].as_u64().unwrap(), writes_before);
    assert_eq!(two_cap.stats_json()["shared_hits"].as_u64().unwrap(), 0);
    assert_eq!(two_cap.stats_json()["occupied_pages"].as_u64().unwrap(), 1);

    // Handle slot generation rejects stale handles and enforces MAX_STORES=16 ERR_CAPACITY.
    let mut handles = Vec::new();
    for _ in 0..16 {
        let h = ffi_create(8, 16 * PAGE as i32);
        assert!(h > 0, "expected valid handle, got {h}");
        handles.push(h);
    }
    assert_eq!(ffi_create(8, 16 * PAGE as i32), ERR_CAPACITY);
    let recycled_target = handles[0];
    assert_eq!(ffi_destroy(recycled_target), 0);
    let fresh_handle = ffi_create(8, 16 * PAGE as i32);
    assert!(fresh_handle > 0 && fresh_handle != recycled_target);
    assert_eq!(ffi_write_page(recycled_target, 0, &[0x22u8; PAGE]), ERR_INVALID_HANDLE);
    assert_eq!(ffi_destroy(fresh_handle), 0);
    for h in handles.into_iter().skip(1) {
        assert_eq!(ffi_destroy(h), 0);
    }

    // Reserved upper bits on handle (bits 29..30) must be rejected cleanly.
    assert_eq!(ffi_write_page(0x2000_0001, 0, &[0x22u8; PAGE]), ERR_INVALID_HANDLE);

    // Multi-page write_range where page 0 drops sole-owner `lit_a` into rollback_dead
    // and page 1 writes a near-duplicate of `lit_a` must skip `Residual(rollback_dead)`
    // so page 1 encodes cleanly as Shared (`1 + PAGE <= PAGE + 1`) rather than failing ERR_CAPACITY
    // or shadowing another live page literal.
    let mut dead_skip_store = VramStore::new(8, PAGE + 1).expect("tight dead-skip store");
    let lit_a = prng_page(909);
    assert_eq!(dead_skip_store.write_page(0, &lit_a).unwrap(), TAG_SHARED);
    let mut lit_a_mut = lit_a;
    lit_a_mut[100] ^= 0x55;
    let mut range_payload = Vec::with_capacity(2 * PAGE);
    range_payload.extend_from_slice(&[0u8; PAGE]);
    range_payload.extend_from_slice(&lit_a_mut);
    dead_skip_store
        .write_range(0, &range_payload)
        .expect("multi-page range must not fit Residual against rollback_dead literal");
    assert_eq!(dead_skip_store.stats_json()["residual_hits"], 0);
    let mut verify_p1 = [0u8; PAGE];
    dead_skip_store.read_page(1, &mut verify_p1).unwrap();
    assert_eq!(verify_p1, lit_a_mut);

    // JSON overflow / invalid-type casts must be rejected cleanly.
    let bad_create = run_from_json(&json!({
        "action": "create",
        "max_pages": -1,
    }));
    assert_eq!(bad_create["ok"], false);
    let bad_len = run_from_json(&json!({
        "action": "read_hex",
        "handle": 1,
        "offset": 0,
        "len": 4_294_967_297u64,
    }));
    assert_eq!(bad_len["ok"], false);
    let bad_handle = run_from_json(&json!({
        "action": "stats",
        "handle": 4_294_967_297u64,
    }));
    assert_eq!(bad_handle["ok"], false);
}
