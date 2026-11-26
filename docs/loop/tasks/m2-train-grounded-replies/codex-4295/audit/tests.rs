use crate::{same, source};
use serde_json::json;

#[test]
fn rejects_silent_address_substitution_and_missing_members() {
    let expected = json!({"bytes":[5,9],"tokens":[1,2]});
    assert!(same(
        &expected,
        &json!({"bytes":[10,14],"tokens":[2,3]}),
        "address"
    )
    .is_err());
    assert!(same(&expected, &json!({"bytes":[5,9]}), "address").is_err());
    assert!(same(
        &expected,
        &json!({"bytes":[5,9],"tokens":[1,2,3]}),
        "address"
    )
    .is_err());
    assert!(same(&expected, &expected, "address").is_ok());
}

#[test]
fn rejects_stated_surface_outside_utf8_boundary() {
    assert!(source::decorate(json!({"bytes":[1,2],"tokens":[0,1]}), "é").is_err());
    assert!(source::decorate(json!({"bytes":[0,2],"tokens":[0,1]}), "é").is_ok());
}
