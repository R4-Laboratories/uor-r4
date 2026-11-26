use serde_json::json;

use crate::{equal_rows, packet, signed};

#[test]
fn a_changed_token_address_fails_even_when_bytes_match() {
    let original = json!({"row":0,"noun":{"bytes":[3,6],"tokens":[8,9]}});
    let changed = json!({"row":0,"noun":{"bytes":[3,6],"tokens":[8,10]}});
    assert!(equal_rows(&[original], &[changed], "token identity").is_err());
}

#[test]
fn typed_state_update_saturates_and_retains_unrelated_bytes() {
    let mut base = vec![0; 13_336];
    base[..8].copy_from_slice(b"UORSTT01");
    base[8..12].copy_from_slice(&1u32.to_le_bytes());
    base[6160 - 4..6160].copy_from_slice(&1u32.to_le_bytes());
    base[6160..6162].copy_from_slice(&(i16::MAX - 1).to_le_bytes());
    base[6162..6164].copy_from_slice(&(i16::MIN + 1).to_le_bytes());
    let state = packet::state(&base).expect("finite state");
    let mut votes = signed::Votes::default();
    votes[0][0][0] = 2;
    votes[0][0][1] = -3;
    let (candidate, changed) = signed::proposal(&base, &state, &votes, 3).expect("finite proposal");
    assert_eq!(changed, 2);
    assert_eq!(&candidate[..6160], &base[..6160]);
    assert_eq!(&candidate[6164..], &base[6164..]);
    assert_eq!(&candidate[6160..6162], &i16::MAX.to_le_bytes());
    assert_eq!(&candidate[6162..6164], &i16::MIN.to_le_bytes());
    let (inactive, changed) =
        signed::proposal(&base, &state, &votes, 4).expect("inactive proposal");
    assert_eq!(changed, 0);
    assert_eq!(inactive, base);
}
