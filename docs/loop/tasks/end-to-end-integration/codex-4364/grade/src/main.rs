use serde_json::{json, Value as V};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};
use uor_r4_tokenizer::ByteBpeTokenizer;

const BASE: &str = "20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589";
const TRAIN: &str = "700c0bd3e78bd2dd8c0644d0abaea3e265538205eec20f69b9da0ccddca1ba92";
const TOKENIZER: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const ARMS: [&str; 3] = ["frozen", "sham", "active"];
const CAP: usize = 96;
const EOS: u16 = 1;

fn sha(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}
fn read(root: &Path, name: &str) -> Vec<u8> {
    fs::read(root.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}
fn doc(root: &Path, name: &str) -> V {
    serde_json::from_slice(&read(root, name)).unwrap()
}
fn jsonl(data: &[u8]) -> Vec<V> {
    std::str::from_utf8(data)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}
fn n(v: &V) -> i64 {
    v.as_i64().unwrap()
}
fn ids(v: &V) -> Vec<u16> {
    serde_json::from_value(v.clone()).unwrap()
}
fn hash_ids(ids: &[u16]) -> String {
    sha(&ids
        .iter()
        .flat_map(|id| id.to_le_bytes())
        .collect::<Vec<_>>())
}
fn hash_pin(pin: &V) {
    let pin = pin.as_str().unwrap();
    assert!(
        pin.len() == 64
            && pin
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    );
}
fn permutation(bytes: &[u8], step: u64) -> Vec<u16> {
    assert_eq!(&bytes[..8], b"UORBFY03");
    for (offset, expected) in [(8, 1), (12, 4096), (16, 512), (20, 6)] {
        assert_eq!(
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()),
            expected
        );
    }
    assert_eq!(u64::from_le_bytes(bytes[44..52].try_into().unwrap()), step);
    let end = bytes.len() - 8;
    let checksum = bytes[..end].iter().fold(0xcbf29ce484222325u64, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(0x100000001b3)
    });
    assert_eq!(bytes[end..], checksum.to_le_bytes());
    let perm: Vec<u16> = bytes[60..8252]
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    let mut sorted = perm.clone();
    sorted.sort_unstable();
    assert!(sorted.into_iter().eq(0..4096));
    perm
}
fn weights(bytes: &[u8]) -> &[u8] {
    &bytes[8252..bytes.len() - 8]
}
fn word_changes(before: &[u8], after: &[u8]) -> V {
    assert_eq!(before.len(), after.len());
    let mut counts = [0usize; 3];
    for (i, (a, b)) in before
        .chunks_exact(2)
        .zip(after.chunks_exact(2))
        .enumerate()
    {
        if a != b {
            let block = if i < 4096 * 512 {
                0
            } else if i < (4096 + 64) * 512 {
                1
            } else {
                2
            };
            counts[block] += 1;
        }
    }
    json!({"total":counts.iter().sum::<usize>(), "tied_embeddings":counts[0], "coarse_centroids":counts[1], "other":counts[2]})
}
fn rank(raw: &[i64], excluded: Option<usize>, gold: usize) -> (V, V) {
    assert_eq!(raw.len(), 64);
    let mut ranking: Vec<usize> = (0..64).filter(|i| Some(*i) != excluded).collect();
    ranking.sort_unstable_by_key(|&i| (std::cmp::Reverse(raw[i]), i));
    let [winner, runner, ..] = ranking[..] else {
        unreachable!()
    };
    let equal: Vec<usize> = ranking
        .iter()
        .copied()
        .take_while(|&i| raw[i] == raw[winner])
        .collect();
    let election = json!({"winner":winner,"runner_up":runner, "winner_score":raw[winner],"runner_up_score":raw[runner],
        "margin":raw[winner]-raw[runner],"equal_maxima":equal});
    let rival = if winner == gold { runner } else { winner };
    let signed = raw[gold] - raw[rival];
    let place = ranking.iter().position(|&i| i == gold).unwrap() + 1;
    (
        json!({"gold_slot":gold,"gold_score_q15":raw[gold],"rank":place,"rival_slot":rival,"rival_score_q15":raw[rival],
        "gold_minus_rival_q15":signed,"strict_gold_win":signed>0,"gold_elected":place==1,"election":election}),
        election,
    )
}
fn token(tok: &ByteBpeTokenizer, id: Option<u16>) -> V {
    id.map_or(V::Null, |id| {
        let b = tok.decode_bytes(&[u32::from(id)]);
        let utf8 = String::from_utf8(b.clone()).ok();
        let hex: String = b.iter().map(|b| format!("{b:02x}")).collect();
        json!({"id":id, "utf8":utf8, "bytes_hex":hex})
    })
}
fn grade(
    row: &V,
    gold: &V,
    partner: &V,
    perm: &[u16],
    tok: &ByteBpeTokenizer,
    checkpoint: &str,
) -> V {
    let id = n(&gold["id"]);
    assert_eq!(row["id"], id);
    assert_eq!(row["pair"], (id + 1) / 2);
    assert_eq!(row["partner_id"], partner["id"]);
    assert_eq!(row["checkpoint_sha256"], checkpoint);
    assert_eq!(
        row["checkpoint_step"],
        if row["arm"] == "frozen" { 3333 } else { 3334 }
    );
    assert_eq!(row["prefix_u16le_sha256"], hash_ids(&ids(&gold["prefix"])));
    let expected = ids(&gold["expected"]);
    let emitted = ids(&row["emitted"]);
    assert!((2..CAP).contains(&expected.len()) && expected.last() == Some(&EOS));
    assert!(expected[..expected.len() - 1]
        .iter()
        .all(|id| (2..4096).contains(id)));
    assert_eq!(row["gold_length_including_eos"], expected.len());
    assert!(!emitted.is_empty() && emitted.iter().all(|id| (1..4096).contains(id)));
    let eos = row["stop"] == "eos";
    if eos {
        assert!(emitted.len() <= CAP && emitted.last() == Some(&EOS));
        assert!(!emitted[..emitted.len() - 1].contains(&EOS));
    } else {
        assert_eq!(row["stop"], "cap");
        assert_eq!(emitted.len(), CAP);
        assert!(!emitted.contains(&EOS));
    }
    let exact = eos && emitted == expected;
    let first_wrong =
        (0..expected.len().max(emitted.len())).find(|&i| expected.get(i) != emitted.get(i));
    let kind = first_wrong.map(|i| match (emitted.get(i), expected.get(i)) {
        (Some(&EOS), Some(_)) => "early_eos",
        (Some(_), Some(&EOS)) => "late_or_missing_eos",
        _ => "wrong_id",
    });
    assert_eq!(row["first_wrong_offset"], json!(first_wrong));
    assert_eq!(row["first_wrong_kind"], json!(kind));
    assert_eq!(row["exact_through_eos"], exact);
    assert_eq!(
        row["grade"],
        if exact {
            "exact_through_eos"
        } else if first_wrong.is_some() {
            "first_wrong"
        } else {
            "censored"
        }
    );
    let actual = emitted[0];
    let donor = ids(&partner["expected"])[0];
    let gold = expected[0];
    assert_ne!(gold, donor);
    assert_eq!(row["actual_opening_id"], actual);
    assert_eq!(row["gold_opening_id"], gold);
    assert_eq!(row["partner_opening_id"], donor);
    assert_eq!(row["gold_first_win"], actual == gold);
    let slot = |id: u16| usize::from(perm[usize::from(id)]);
    let (a, g, d, bos) = (slot(actual), slot(gold), slot(donor), slot(0));
    let coarse_raw: Vec<i64> = serde_json::from_value(row["raw_coarse_q15"].clone()).unwrap();
    let gold_raw: Vec<i64> = serde_json::from_value(row["raw_gold_fine_q15"].clone()).unwrap();
    let elected_raw: Vec<i64> =
        serde_json::from_value(row["raw_elected_fine_q15"].clone()).unwrap();
    let (coarse, _) = rank(&coarse_raw, None, g / 64);
    let (fine, _) = rank(&gold_raw, (g / 64 == bos / 64).then_some(bos % 64), g % 64);
    let (_, elected) = rank(
        &elected_raw,
        (a / 64 == bos / 64).then_some(bos % 64),
        a % 64,
    );
    assert_eq!(coarse, row["coarse"]);
    assert_eq!(fine, row["fine_gold_cell"]);
    assert_eq!(elected, row["elected_fine"]);
    assert_eq!(coarse["election"]["winner"], a / 64);
    assert_eq!(elected["winner"], a % 64);
    assert_eq!(row["elected_coarse_cell"], a / 64);
    assert_eq!(row["gold_cell_admitted"], g / 64 == a / 64);
    if g / 64 == a / 64 {
        assert_eq!(gold_raw, elected_raw);
    }
    let (stage, gap) = if g / 64 == d / 64 {
        ("fine", gold_raw[g % 64] - gold_raw[d % 64])
    } else {
        ("coarse", coarse_raw[g / 64] - coarse_raw[d / 64])
    };
    assert_eq!(row["source_comparison_stage"], stage);
    assert_eq!(row["source_gold_minus_partner_q15"], gap);
    assert_eq!(row["source_gold_strict_win"], gap > 0);
    assert_eq!(row["blind_sham_match"], true);
    assert_eq!(row["diagnostic_transition_match"], true);
    hash_pin(&row["native_rust_transition_sha256"]);
    let wrong = first_wrong.map(|i| {
        json!({"offset":i,"kind":kind,"expected":token(tok, expected.get(i).copied()),
        "emitted":token(tok, emitted.get(i).copied())})
    });
    json!({"arm":row["arm"],"id":id,"pair":row["pair"], "stop":row["stop"],"emitted_count":emitted.len(),
        "expected_count":expected.len(),"exact":exact,"first_wrong":wrong,"first_win":actual==gold,
        "actual_opening_id":actual,"gold_cell_admitted":g/64==a/64,"coarse":coarse,
        "fine_gold_cell":fine,"elected_fine":elected,"source_comparison_stage":stage,"source_gap":gap})
}
fn deltas(a: &V, b: &V, raw_a: &V, raw_b: &V) -> V {
    let metric = |name: &str| {
        json!({"rank": n(&b[name]["rank"])-n(&a[name]["rank"]),
            "gap_q15":n(&b[name]["gold_minus_rival_q15"])-n(&a[name]["gold_minus_rival_q15"]),
            "gold_score_q15":n(&b[name]["gold_score_q15"])-n(&a[name]["gold_score_q15"]),
            "rival_before":a[name]["rival_slot"],"rival_after":b[name]["rival_slot"]})
    };
    let a_ids = ids(&raw_a["emitted"]);
    let b_ids = ids(&raw_b["emitted"]);
    let i = (0..a_ids.len().max(b_ids.len())).find(|&i| a_ids.get(i) != b_ids.get(i));
    json!({"coarse":metric("coarse"),"conditional_fine":metric("fine_gold_cell"),
        "source_gap_q15":n(&b["source_gap"])-n(&a["source_gap"]),
        "rollout_first_difference_offset":i,"opening_changed":a["actual_opening_id"]!=b["actual_opening_id"],
        "stop_changed":a["stop"]!=b["stop"]})
}
fn main() {
    let arg = std::env::args().nth(1).expect("input directory");
    let root = Path::new(&arg);
    let tape_bytes = read(root, "train.jsonl");
    assert_eq!(sha(&tape_bytes), TRAIN);
    let tape = jsonl(&tape_bytes);
    assert_eq!(tape.len(), 4);
    assert_eq!(
        tape.iter()
            .map(|r| ids(&r["expected"]).len())
            .sum::<usize>(),
        167
    );
    let roles_bytes = read(root, "role-and-source.jsonl");
    assert_eq!(
        sha(&roles_bytes),
        "f8fe9561f79dc41ffb6d9c52423ce215f533e7482aa48a833f391611b3656630"
    );
    let roles = jsonl(&roles_bytes);
    assert_eq!(roles.len(), 4);
    for (i, role) in roles.iter().enumerate() {
        assert_eq!(role["id"], i + 1);
        assert_eq!(role["source_graph_pair"], if i < 2 { 33 } else { 120 });
    }
    let tok_bytes = read(root, "tokenizer.json");
    assert_eq!(sha(&tok_bytes), TOKENIZER);
    let tok = ByteBpeTokenizer::from_tokenizer_json_bytes(&tok_bytes).unwrap();
    let report = doc(root, "training.json");
    assert_eq!(report["protocol"], "uor-opening-real-single-update-v1");
    assert_eq!(report["parent_sha256"], BASE);
    assert_eq!(report["roster_sha256"], TRAIN);
    for (key, value) in [
        ("labels_per_arm", 167),
        ("updates_per_arm", 1),
        ("step_before", 3333),
        ("step_after", 3334),
        ("active_gain_q15", 32768),
        ("sham_gain_q15", 0),
        ("margin_words", 256),
        ("baseline_forward_parity_transitions", 3938),
    ] {
        assert_eq!(report[key], value);
    }
    let batch: Vec<usize> = tape
        .iter()
        .map(|r| ids(&r["prefix"]).len() + ids(&r["expected"]).len())
        .collect();
    assert_eq!(report["batch_lengths"], json!(batch));
    let finite_bytes = read(root, "finite.json");
    let finite: V = serde_json::from_slice(&finite_bytes).unwrap();
    assert_eq!(finite["protocol"], "uor-opening-finite-v1");
    assert_eq!(finite["verdict"], "PASS");
    assert_eq!(finite["dim"], 512);
    assert_eq!(finite["registered_constants"]["margin"], 256);
    assert_eq!(finite["registered_constants"]["gain"], 32768);
    assert!(finite["cases"]
        .as_array()
        .unwrap()
        .contains(&json!("sham and Rust full-step parity")));
    let opening_indices: Vec<_> = tape.iter().map(|r| ids(&r["prefix"]).len()).collect();
    assert_eq!(report["opening_labels"], json!(opening_indices));
    let trace: Vec<i64> =
        serde_json::from_value(report["device_active_opening_trace"].clone()).unwrap();
    assert_eq!(trace.len(), 40);
    for (trace, row) in trace
        .chunks_exact(10)
        .zip(report["ordinary_before"].as_array().unwrap())
    {
        for (offset, stage) in [(0, "coarse"), (5, "fine")] {
            let rival = if stage == "coarse" {
                "gold_coarse_rival_cell"
            } else {
                "gold_fine_rival_slot"
            };
            let gap = n(&row[format!("gold_{stage}_gap_q15")]);
            assert_eq!(trace[offset], n(&row[rival]));
            assert_eq!(trace[offset + 1], gap);
            assert_eq!(trace[offset + 2], (256 - gap).max(0));
            assert_eq!(trace[offset + 3], if gap < 256 { 32768 } else { 0 });
            assert_eq!(trace[offset + 4], 0);
        }
    }
    assert_eq!(report["finite_receipt_sha256"], sha(&finite_bytes));
    assert_eq!(finite["cubin_sha256"], report["cubin_sha256"]);
    assert_eq!(
        finite["historical_cubin_sha256"],
        report["historical_control_cubin_sha256"]
    );
    assert_eq!(finite["gpu_uuid"], report["gpu_uuid"]);
    let checkpoints = [
        read(root, "base.uorbfy03"),
        read(root, "sham.uorbfy03"),
        read(root, "active.uorbfy03"),
    ];
    let fingerprints: Vec<_> = checkpoints.iter().map(|b| sha(b)).collect();
    assert_eq!(fingerprints[0], BASE);
    let perm = permutation(&checkpoints[0], 3333);
    for i in 1..3 {
        assert_eq!(perm, permutation(&checkpoints[i], 3334));
        assert_eq!(checkpoints[0][..44], checkpoints[i][..44]);
        assert_eq!(checkpoints[0][52..8252], checkpoints[i][52..8252]);
        let arm = ARMS[i];
        assert_eq!(report[format!("{arm}_checkpoint_sha256")], fingerprints[i]);
        let words = &report[format!("{arm}_words")];
        for name in ["gradient", "weights", "adam_m", "adam_v"] {
            hash_pin(&words[format!("{name}_le_sha256")]);
        }
        assert_eq!(words["weights_le_sha256"], sha(weights(&checkpoints[i])));
    }
    let word_deltas = json!({"frozen_to_sham":word_changes(weights(&checkpoints[0]),weights(&checkpoints[1])),
        "frozen_to_active":word_changes(weights(&checkpoints[0]),weights(&checkpoints[2])),
        "sham_to_active":word_changes(weights(&checkpoints[1]),weights(&checkpoints[2]))});
    let gradient_change = n(&report["different_gradient_words"]);
    assert!((0..=(weights(&checkpoints[0]).len() / 2) as i64).contains(&gradient_change));
    assert_eq!(
        report["sham_words"]["gradient_le_sha256"] != report["active_words"]["gradient_le_sha256"],
        gradient_change > 0
    );
    assert_eq!(
        report["result"],
        if gradient_change > 0 && n(&word_deltas["sham_to_active"]["total"]) > 0 {
            "MEASURED_UPDATE"
        } else {
            "NO_EFFECTIVE_WEIGHT_CONTRAST"
        }
    );
    if std::env::args().any(|arg| arg == "--receipt-only") {
        println!(
            "{}",
            json!({"weights":word_deltas,"gradient_words_changed":gradient_change,"status":report["result"]})
        );
        return;
    }
    let raw_bytes = read(root, "rows.jsonl");
    let raw = jsonl(&raw_bytes);
    let readout = doc(root, "readout.json");
    assert_eq!(readout["schema"], "uor-physical-opposed-opening-readout/1");
    assert_eq!(readout["verdict"], "COMPLETE");
    assert_eq!(readout["cap"], CAP);
    assert_eq!(readout["train_sha256"], TRAIN);
    assert_eq!(readout["rows_sha256"], sha(&raw_bytes));
    assert_eq!(
        readout["training_receipt_sha256"],
        sha(&read(root, "training.json"))
    );
    assert_eq!(readout["training_cubin_sha256"], report["cubin_sha256"]);
    assert_eq!(
        readout["finite_receipt_sha256"],
        report["finite_receipt_sha256"]
    );
    assert_eq!(
        readout["serving_cubin_sha256"],
        "8f5ab22af0074c38c582b17b1f92b5e5dff906b2bb366dccdadb99df7e23c713"
    );
    assert_eq!(
        readout["historical_fixture_sha256"],
        "dcc3019546fd1fe6e272aa71a64d694db2754b946bc82638f64b19671e47d04d"
    );
    assert_eq!(raw.len(), 12);
    assert_eq!(readout["rows"], raw.len());
    let mut scores = Vec::new();
    let mut pairs = Vec::new();
    let mut arms = Vec::new();
    let plant = std::env::args().any(|arg| arg == "--plant-bad-emission");
    for (a, arm) in ARMS.iter().enumerate() {
        let arm_start = scores.len();
        for (i, row) in raw[4 * a..4 * a + 4].iter().enumerate() {
            let mut row = row.clone();
            if plant && a == 0 && i == 0 {
                let first = n(&row["emitted"][0]);
                row["emitted"][0] = json!(if first == 2 { 3 } else { 2 });
            }
            assert_eq!(row["arm"], *arm);
            let s = grade(&row, &tape[i], &tape[i ^ 1], &perm, &tok, &fingerprints[a]);
            let receipt_row = &report[[
                "ordinary_before",
                "ordinary_after_sham",
                "ordinary_after_active",
            ][a]][i];
            for (report_key, expected) in [
                ("id", &s["id"]),
                ("gold_original_id", &row["gold_opening_id"]),
                ("elected_original_id", &s["actual_opening_id"]),
                ("gold_cell", &s["coarse"]["gold_slot"]),
                ("elected_cell", &s["coarse"]["election"]["winner"]),
                ("gold_coarse_rival_cell", &s["coarse"]["rival_slot"]),
                ("gold_coarse_gap_q15", &s["coarse"]["gold_minus_rival_q15"]),
                ("gold_fine_rival_slot", &s["fine_gold_cell"]["rival_slot"]),
                (
                    "gold_fine_gap_q15",
                    &s["fine_gold_cell"]["gold_minus_rival_q15"],
                ),
                (
                    "elected_cell_fine_rival_slot",
                    &s["elected_fine"]["runner_up"],
                ),
                ("elected_cell_fine_gap_q15", &s["elected_fine"]["margin"]),
            ] {
                assert_eq!(
                    &receipt_row[report_key], expected,
                    "receipt arm {arm} row {i} field {report_key}"
                );
            }
            scores.push(s);
        }
        let first_pair = pairs.len();
        for duo in scores[arm_start..].chunks_exact(2) {
            let count = |field| duo.iter().filter(|s| s[field] == true).count();
            let wins = count("first_win");
            let strict = duo.iter().filter(|s| n(&s["source_gap"]) > 0).count();
            pairs.push(json!({"arm":arm,"pair":duo[0]["pair"],"source_comparison_stage":duo[0]["source_comparison_stage"],
                "ordinary_first_wins":wins,"both_members_first_win":wins==2,"strict_source_score_wins":strict,
                "both_members_strict_source_win":strict==2,"source_ties":duo.iter().filter(|s| s["source_gap"]==0).count(),
                "exact_through_eos":count("exact")}));
        }
        let arm_scores = &scores[arm_start..];
        let count = |field| arm_scores.iter().filter(|s| s[field] == true).count();
        let two = pairs[first_pair..]
            .iter()
            .filter(|p| p["both_members_first_win"] == true)
            .count();
        let ends = arm_scores.iter().filter(|s| s["stop"] == "eos").count();
        let fine = arm_scores
            .iter()
            .filter(|s| s["fine_gold_cell"]["gold_elected"] == true)
            .count();
        let strict_fine = arm_scores
            .iter()
            .filter(|s| s["fine_gold_cell"]["strict_gold_win"] == true)
            .count();
        let strict_coarse = arm_scores
            .iter()
            .filter(|s| s["coarse"]["strict_gold_win"] == true)
            .count();
        arms.push(
            json!({"arm":arm,"both_members_first_win":two,"first_wins":count("first_win"),
            "gold_cell_admissions":count("gold_cell_admitted"),"strict_gold_cells":strict_coarse,
            "conditional_gold_fine_wins":fine,"strict_conditional_gold_fine_wins":strict_fine,
            "eos":ends,"cap":4-ends,"exact_through_eos":count("exact")}),
        );
    }
    assert_eq!(readout["pairs"], json!(pairs));
    let comparison: Vec<_> = (0..4)
        .map(|i| {
            json!({"id":i+1,
                "frozen_to_sham":deltas(&scores[i],&scores[i+4],&raw[i],&raw[i+4]),
                "frozen_to_active":deltas(&scores[i],&scores[i+8],&raw[i],&raw[i+8]),
                "sham_to_active":deltas(&scores[i+4],&scores[i+8],&raw[i+4],&raw[i+8])
            })
        })
        .collect();
    let cold_path = root.join("cold");
    let cold_checked = if cold_path.join("report.json").exists() {
        let cold = doc(&cold_path, "report.json");
        assert_eq!(cold["schema"], "uor-cold-opposed-opening-parity/1");
        assert_eq!(cold["verdict"], "COMPLETE");
        assert_eq!(cold["checked"], 12);
        assert_eq!(cold["gpu_rows_sha256"], sha(&raw_bytes));
        assert_eq!(
            cold["training_receipt_sha256"],
            sha(&read(root, "training.json"))
        );
        assert_eq!(cold["tokenizer_sha256"], TOKENIZER);
        assert_eq!(cold["train_sha256"], TRAIN);
        assert_eq!(cold["role_source_sha256"], sha(&roles_bytes));
        let entries = cold["rows"].as_array().unwrap();
        assert_eq!(entries.len(), 12);
        for (index, row) in raw.iter().enumerate() {
            let id = index % 4 + 1;
            let arm = ARMS[index / 4];
            let trace = doc(&cold_path, &format!("{arm}/{id}.trace.json"));
            let manifest = read(&cold_path, &format!("{arm}/manifest.json"));
            let stdout = read(&cold_path, &format!("{arm}/{id}.response.bytes"));
            let messages = doc(&cold_path, &format!("{arm}/{id}.messages.json"));
            assert_eq!(messages, roles[id - 1]["role_messages"]);
            assert_eq!(trace["prefix"], tape[id - 1]["prefix"]);
            assert_eq!(trace["emitted"], row["emitted"]);
            assert_eq!(trace["stop"], row["stop"]);
            let mut emitted: Vec<u32> = serde_json::from_value(trace["emitted"].clone()).unwrap();
            if trace["stop"] == "eos" {
                assert_eq!(emitted.pop(), Some(u32::from(EOS)));
            }
            assert_eq!(stdout, tok.decode_bytes(&emitted));
            assert_eq!(
                entries[index],
                json!({"arm":arm,"id":id,"checkpoint_sha256":row["checkpoint_sha256"],
                "manifest_sha256":sha(&manifest),"prefix_u16le_sha256":row["prefix_u16le_sha256"],
                "emitted_count":emitted.len(),"stop":trace["stop"],"response_bytes_sha256":sha(&stdout),
                "prefix_ids_equal":true,"raw_ids_equal":true,"stop_equal":true,"stdout_equal":true})
            );
        }
        12
    } else {
        0
    };
    let better = n(&arms[2]["both_members_first_win"]) > n(&arms[1]["both_members_first_win"]);
    println!("{}", serde_json::to_string_pretty(&json!({"public":{"rows_sha256":sha(&raw_bytes),
        "training_receipt_sha256":readout["training_receipt_sha256"],"weight_words_changed":word_deltas,
        "recorded_gradient_words_changed":gradient_change,"arms":arms,"pairs":pairs,"cold_checked":cold_checked,
        "primary":if better {"extra TRAIN pair fully correct"} else {"no demonstrated opening correction at this dose"}},
        "private":{"documents":scores,"arm_deltas":comparison}})).unwrap());
}
