//! Integration tests for the Lean-certified `verdant_studio` engine & virtual-FS driver.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[test]
fn vendored_verdant_studio_source_matches_provenance_sha256() {
    let lib_src = include_str!("../src/verdant_studio.rs");
    let prov: Value =
        serde_json::from_str(include_str!("../src/verdant_provenance.json")).expect("provenance JSON");
    let actual_sha = format!("{:x}", Sha256::digest(lib_src.as_bytes()));
    assert_eq!(
        prov["files"]["src/lib.rs"].as_str(),
        Some(actual_sha.as_str()),
        "vendored verdant_studio.rs must match provenance.json byte-for-byte"
    );
}

#[test]
fn check_next_agenda_admit_decision_and_encoding_work_over_virtual_fs() {
    // 1. Fresh auto-initialized project passes `check`, `next`, `agenda`.
    let check_empty = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["check"],
        "files": {},
        "auto_init": true
    }));
    assert_eq!(check_empty["ok"], true, "check empty failed: {check_empty}");
    assert!(check_empty["written_files"].as_object().unwrap().is_empty());

    let next_res = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["next"],
        "files": {},
        "auto_init": true
    }));
    assert_eq!(next_res["ok"], true, "next failed: {next_res}");

    let agenda_res = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["agenda"],
        "files": {},
        "auto_init": true
    }));
    assert_eq!(agenda_res["ok"], true, "agenda failed: {agenda_res}");

    // 2. Admit a stakeholder requirement and a construct citing it, then verify `check`.
    let admit_lines = [
        json!({
            "id": "REQ-UI-01",
            "statement": "The studio editor previews valid scenes live.",
            "stakeholder": "casey",
            "evidence": { "source": "studio" }
        })
        .to_string(),
        json!({
            "construct": "PreviewScene",
            "req": "REQ-UI-01"
        })
        .to_string(),
    ]
    .join("\n");

    let admit_res = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["admit"],
        "inline_input": admit_lines,
        "files": {},
        "auto_init": true
    }));
    assert_eq!(admit_res["ok"], true, "admit failed: {admit_res}");
    let written = admit_res["written_files"].as_object().expect("written files map");
    assert!(written.contains_key("model/requirements.json"));
    assert!(written.contains_key("model/admissions.json"));
    assert!(written.contains_key("model/register-lock.json"));

    // Subsequent `check` over the written workspace passes, and duplicate admit is refused.
    let recheck = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["check"],
        "files": written,
        "auto_init": true
    }));
    assert_eq!(recheck["ok"], true, "recheck failed: {recheck}");

    let dup_admit = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["admit"],
        "inline_input": admit_lines,
        "files": written,
        "auto_init": true
    }));
    assert_eq!(dup_admit["ok"], false, "duplicate admit must be refused");

    // 3. Record an owner decision and refuse duplicate timestamps.
    let dec_line = json!({
        "agent": "antigravity",
        "at": "2026-11-25T02:00:00Z",
        "quote": "Admit PreviewScene under REQ-UI-01",
        "requirements": ["REQ-UI-01"]
    })
    .to_string();
    let dec_res = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["record-decision"],
        "inline_input": dec_line,
        "files": written,
        "auto_init": true
    }));
    assert_eq!(dec_res["ok"], true, "record-decision failed: {dec_res}");
    let after_dec = dec_res["written_files"].as_object().expect("decision written");
    assert!(after_dec.contains_key("model/owner-decisions.json"));

    // Merge mutated files into workspace state.
    let mut ws = written.clone();
    for (k, v) in after_dec {
        ws.insert(k.clone(), v.clone());
    }
    let dup_dec = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["record-decision"],
        "inline_input": dec_line,
        "files": ws,
        "auto_init": true
    }));
    assert_eq!(dup_dec["ok"], false, "duplicate owner decision must be refused");

    // 4. Propose a JSON Schema 2020-12 encoding -> `.lex.tex`.
    let schema = json!({
        "title": "SceneNode",
        "type": "object",
        "properties": {
            "id": { "type": "string" },
            "visible": { "type": "boolean" }
        },
        "required": ["id", "visible"]
    })
    .to_string();
    let enc_res = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["propose-encoding", "SceneNode", "req", "REQ-UI-01"],
        "inline_input": schema,
        "files": ws,
        "auto_init": true
    }));
    assert_eq!(enc_res["ok"], true, "propose-encoding failed: {enc_res}");
    let enc_written = enc_res["written_files"].as_object().expect("encoding written");
    assert!(
        enc_written.contains_key("app/src/Catalog/SceneNode.lex.tex"),
        "expected Catalog/SceneNode.lex.tex in {enc_written:?}"
    );
}

#[test]
fn path_traversal_and_outside_workspace_paths_are_refused() {
    let traversal = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["check"],
        "files": { "../etc/passwd": "root:x:0:0" },
        "auto_init": false
    }));
    assert_eq!(traversal["ok"], false);

    let arg_traversal = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["admit", "../../etc/passwd"],
        "files": {},
        "auto_init": true
    }));
    assert_eq!(arg_traversal["ok"], false);
    assert!(
        !arg_traversal["stderr"].as_str().unwrap_or("").contains("could not write"),
        "traversal must not misreport as write failure: {arg_traversal}"
    );
}

#[test]
fn review_regressions_missing_files_bounds_and_stage_jsonl_work_cleanly() {
    // Missing files report genuine read/missing diagnostics rather than 'could not write',
    // and auto_init: false passes cleanly when workspace registers exist without authorities.toml.
    let missing_admit = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["admit", "missing.jsonl"],
        "files": {},
        "auto_init": true
    }));
    assert_eq!(missing_admit["ok"], false);
    assert!(
        !missing_admit["stderr"].as_str().unwrap_or("").contains("could not write"),
        "missing file misreported as write failure: {missing_admit}"
    );

    let seed = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["check"],
        "files": {},
        "auto_init": false
    }));
    assert!(
        !seed["stderr"].as_str().unwrap_or("").contains("could not write"),
        "auto_init: false misreported read miss as write failure: {seed}"
    );

    // Non-string args and invalid path chars (':' / control chars) are rejected cleanly.
    let non_str_arg = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["propose-encoding", 123, "req", "R1"],
        "files": {},
        "auto_init": true
    }));
    assert_eq!(non_str_arg["ok"], false);

    let colon_arg = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["stage-bindings"],
        "inline_input": json!({
            "id": "STD:1",
            "domain": "ui",
            "name": "Bad colon id"
        }).to_string(),
        "files": {},
        "auto_init": true
    }));
    assert_eq!(colon_arg["ok"], false, "colon in path must be refused: {colon_arg}");

    // Oversized args / deeply nested JSON / XML do not overflow stack.
    let long_arg = "a".repeat(10_000);
    let long_arg_res = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["read-xml", long_arg],
        "files": {},
        "auto_init": true
    }));
    assert_eq!(long_arg_res["ok"], false);

    let deep_json = format!("{}{}", "[".repeat(5000), "]".repeat(5000));
    let deep_json_res = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["propose-encoding", "Big", "req", "R1"],
        "inline_input": deep_json,
        "files": {},
        "auto_init": true
    }));
    assert_eq!(deep_json_res["ok"], false);

    let deep_xml = "<a>".repeat(5000);
    let deep_xml_res = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["read-xml"],
        "inline_input": deep_xml,
        "files": {},
        "auto_init": true
    }));
    assert_eq!(deep_xml_res["ok"], false);

    // stage-admissions & stage-decisions work with inline_input via .verdant-input.jsonl.
    let stage_adm = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["stage-admissions"],
        "inline_input": json!({
            "id": "REQ-S1",
            "statement": "Staged admission check",
            "stakeholder": "casey",
            "evidence": { "source": "test" }
        }).to_string(),
        "files": {},
        "auto_init": true
    }));
    assert_eq!(stage_adm["ok"], true, "stage-admissions failed: {stage_adm}");

    let stage_dec = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["stage-decisions"],
        "inline_input": json!({
            "agent": "antigravity",
            "at": "2026-11-25T03:00:00Z",
            "quote": "Staged decision check",
            "requirements": ["REQ-S1"]
        }).to_string(),
        "files": {},
        "auto_init": true
    }));
    assert_eq!(stage_dec["ok"], true, "stage-decisions failed: {stage_dec}");

    // Unrelated React/TSX files with >400 commas/tags do not trip JSON/XML complexity bounds.
    let tsx_props = (0..450)
        .map(|i| format!("<div key={{{i}}} onClick={{() => [1,2,{i}]}} />,"))
        .collect::<Vec<_>>()
        .join("\n");
    let tsx_ws = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["check"],
        "files": {
            "src/components/ui/sidebar.tsx": format!("export const Sidebar = () => <>{tsx_props}</>;")
        },
        "auto_init": true
    }));
    assert_eq!(tsx_ws["ok"], true, "TSX workspace check failed: {tsx_ws}");
}

#[test]
fn r2_regressions_lines_xml_attrs_bom_csv_and_stage_mappings_work_cleanly() {
    // Newline bombs on JSONL/TOML are caught cleanly before stack/OOM crash.
    let newline_admit = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["admit"],
        "inline_input": "\n".repeat(10_000),
        "files": {},
        "auto_init": true
    }));
    assert_eq!(newline_admit["ok"], false);

    let blank_stage_map = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["stage-mappings"],
        "inline_input": " \n".repeat(9_000),
        "files": {},
        "auto_init": true
    }));
    assert_eq!(blank_stage_map["ok"], false);

    let toml_lines = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["check"],
        "files": { "model/authorities.toml": "\n".repeat(20_000) },
        "auto_init": true
    }));
    assert_eq!(toml_lines["ok"], false);

    // Huge single-tag XML attribute list & UTF-8 BOM XML nesting are caught cleanly.
    let mut attrs = String::new();
    for i in 0..300 {
        attrs.push_str(&format!(" a{i}=\"1\""));
    }
    let attr_xml = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["read-xml"],
        "inline_input": format!("<r{attrs}/>"),
        "files": {},
        "auto_init": true
    }));
    assert_eq!(attr_xml["ok"], false);

    let bom_xml = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["read-xml", "evil.txt"],
        "files": { "evil.txt": format!("\u{feff}{}", "<a>".repeat(5000)) },
        "auto_init": true
    }));
    assert_eq!(bom_xml["ok"], false);

    // Valid standards-index CSV >= 60 rows (540+ commas) is accepted cleanly by ingest-index.
    let mut csv_rows = vec!["id,domain,name,owner,phases,scope,adoption,adoption_code,source".to_owned()];
    for i in 0..65 {
        csv_rows.push(format!("STD-{i:03},ui,Rule {i},casey,p1,studio,adopted,A,doc"));
    }
    let csv_res = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["ingest-index"],
        "inline_input": csv_rows.join("\n"),
        "files": {},
        "auto_init": true
    }));
    assert_eq!(csv_res["ok"], true, "ingest-index 65-row CSV failed: {csv_res}");

    // stage-mappings works via .verdant-input.jsonl with a valid SKOS mapping record.
    let map_res = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["stage-mappings"],
        "inline_input": json!({
            "subject": "urn:std:1",
            "relation": "exactMatch",
            "object": "urn:studio:ui",
            "scheme": "std",
            "evidence": "Standard doc ref"
        }).to_string(),
        "files": {},
        "auto_init": true
    }));
    assert_eq!(map_res["ok"], true, "stage-mappings failed: {map_res}");
}

#[test]
fn r3_regressions_toml_not_json_and_large_register_roundtrip_work_cleanly() {
    // Valid model/authorities.toml starting with `[[authority]]` must not be misclassified as JSON.
    let mut auth_toml = String::new();
    for i in 0..60 {
        auth_toml.push_str(&format!(
            "[[authority]]\nid = \"AUTH-{i:03}\"\nrealized_by = [\"a\",\"b\",\"c\",\"d\",\"e\",\"f\",\"g\",\"h\",\"i\"]\n"
        ));
    }
    let auth_check = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["check"],
        "files": { "model/authorities.toml": auth_toml },
        "auto_init": true
    }));
    assert!(
        !auth_check["stderr"].as_str().unwrap_or("").contains("structural item count"),
        "TOML must not trigger JSON structural limit: {auth_check}"
    );

    // Admitting 70 requirements writes model/requirements.json that cleanly reloads on subsequent `check`.
    let req_lines: String = (0..70)
        .map(|i| {
            json!({
                "id": format!("REQ-B-{i:03}"),
                "statement": format!("Requirement {i} holds."),
                "stakeholder": "casey",
                "evidence": { "source": "studio" }
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    let big_admit = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["admit"],
        "inline_input": req_lines,
        "files": {},
        "auto_init": true
    }));
    assert_eq!(big_admit["ok"], true, "70-req admit failed: {big_admit}");
    let written = big_admit["written_files"].as_object().expect("written files");
    let recheck = hologram_forge_core::verdant_driver::run_from_json(&json!({
        "args": ["check"],
        "files": written,
        "auto_init": true
    }));
    assert_eq!(recheck["ok"], true, "recheck after 70-req admit failed: {recheck}");
}
