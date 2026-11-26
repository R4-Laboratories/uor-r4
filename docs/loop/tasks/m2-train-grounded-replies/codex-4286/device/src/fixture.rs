use crate::{
    packet::{self, Address, Edge, Input, Label, Row, Site, Span},
    rank::{self, Endpoint},
    require, save, Result,
};
use std::{
    fs,
    io::{BufRead, BufReader},
    path::Path,
};
use uor_r4_integer::role_action::instruction::learned::Point;

fn scalar(value: i16) -> Point {
    let mut point = [[0; 32]; 3];
    point[0][0] = value;
    point
}

pub fn write(path: &Path) -> Result<()> {
    fs::create_dir(path)?;
    let mut fit = Vec::new();
    packet::write_word(&mut fit, 1)?;
    for _ in 0..9 {
        for value in [0, 8, 2, 0, 0, 0, 0] {
            packet::write_word(&mut fit, 1)?;
            packet::write_word(&mut fit, 0)?;
            packet::write_point(&mut fit, scalar(value))?;
        }
    }
    save(&path.join("fit.words"), &fit)?;
    let mut arms = Vec::new();
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for (i, values) in [
        [0, 8, 2, 1, 0, 7],
        [0, 8, 4, 1, 2, 7],
        [4, 4, 2, 24, 24, 24],
    ]
    .into_iter()
    .enumerate()
    {
        let sites: Vec<_> = [10, 30, 60, 20, 70, 32]
            .into_iter()
            .zip(values)
            .map(|(p, v)| Site {
                address: Address {
                    bytes: Span {
                        start: p * 3,
                        end: p * 3 + 2,
                    },
                    tokens: Span {
                        start: p,
                        end: p + 1,
                    },
                },
                point: scalar(v),
            })
            .collect();
        let [a, e, h] = [sites[0].address, sites[1].address, sites[2].address];
        let neutral = sites[3].address;
        let row = Row {
            sites,
            tuples: vec![
                [0, 1, 2],
                [1, 0, 2],
                [3, 1, 2],
                [1, 3, 2],
                [0, 1, 4],
                [1, 0, 4],
                [3, 5, 2],
            ],
        };
        // Adjacent to the near neutral on row zero; overlap on subsequent rows.
        let plus = if i == 0 {
            Span { start: 19, end: 20 }
        } else {
            a.tokens
        };
        let minus = Span { start: 29, end: 30 };
        let correct = [
            Edge {
                governor: plus,
                noun: a.tokens,
            },
            Edge {
                governor: minus,
                noun: e.tokens,
            },
        ];
        let exchange = [
            Edge {
                noun: e.tokens,
                ..correct[0]
            },
            Edge {
                noun: a.tokens,
                ..correct[1]
            },
        ];
        arms.push([
            Input::Native,
            Input::Governors([plus, minus]),
            Input::Nouns([a.tokens, e.tokens]),
            Input::Links {
                governors: [plus, minus],
                edges: correct,
            },
            Input::Links {
                governors: [plus, minus],
                edges: exchange,
            },
        ]);
        labels.push(Label {
            case_id: format!("finite/{i}"),
            pair_id: format!("finite/{}", i / 2),
            family: "finite".into(),
            distance: "near".into(),
            admission: "take".into(),
            prompt_sha256: "synthetic".into(),
            gold: [a, e, h],
            neutral,
        });
        rows.push(row);
    }
    let prepared = path.join("prepared");
    packet::store(&prepared, &rows, &labels, &arms)?;
    let control_inputs = arms
        .iter()
        .map(|arm| linkage_controls(arm[3]))
        .collect::<Result<Vec<_>>>()?;
    packet::write_arm(
        &prepared,
        "wrong_governor",
        control_inputs.iter().map(|x| x[0]),
        false,
    )?;
    packet::write_arm(
        &prepared,
        "flipped_governor_keys",
        control_inputs.iter().map(|x| x[1]),
        false,
    )
}

fn linkage_controls(input: Input) -> Result<[Input; 2]> {
    let Input::Links {
        governors: [plus, minus],
        edges,
    } = input
    else {
        return Err("fixture requires an addressed link arm".into());
    };
    let mut wrong_key = edges;
    // Change only one edge's source span, even preserving its starting address.
    wrong_key[0].governor.end += 1;
    Ok([
        Input::Links {
            governors: [plus, minus],
            edges: wrong_key,
        },
        Input::Links {
            governors: [minus, plus],
            edges,
        },
    ])
}

fn check_linkage_controls(
    prepared: &Path,
    results: &Path,
    fit: &uor_r4_integer::role_action::instruction::learned::Learned,
) -> Result<()> {
    let mut geometry = BufReader::new(fs::File::open(prepared.join("geometry.bin"))?);
    let rows = packet::read_geometry(&mut geometry)?;
    let read_arm = |name: &str| {
        let mut input = BufReader::new(fs::File::open(prepared.join(format!("{name}.bin")))?);
        packet::read_arm(&mut input, 3, rows.len())
    };
    let intact = read_arm("correct_link")?;
    let wrong = read_arm("wrong_governor")?;
    let flipped = read_arm("flipped_governor_keys")?;
    let mut physical_wrong = BufReader::new(fs::File::open(results.join("wrong_governor.rank"))?);
    let mut physical_flip =
        BufReader::new(fs::File::open(results.join("flipped_governor_keys.rank"))?);
    require(
        packet::start(&mut physical_wrong)? == rows.len()
            && packet::start(&mut physical_flip)? == rows.len(),
        "fixture linkage row count changed",
    )?;
    for (i, row) in rows.iter().enumerate() {
        require(
            [wrong[i], flipped[i]] == linkage_controls(intact[i])?,
            "fixture linkage controls changed more than the registered keys",
        )?;
        let h = row.addresses(row.tuples[0])[2];
        let terms = rank::point_terms(row, fit)?;
        let (abstention, fixed_heading) =
            rank::check_output(&mut physical_wrong, row, &terms, wrong[i], h)?;
        let (flip, _) = rank::check_output(&mut physical_flip, row, &terms, flipped[i], h)?;
        for pick in [abstention, fixed_heading] {
            require(
                pick.eligible == 0
                    && pick.ties == 0
                    && pick.ordinal.is_none()
                    && pick.bank_cost.is_none()
                    && pick.selection.is_none()
                    && pick.tokens.is_none(),
                "an unmatched governor key must abstain even with both noun sites present",
            )?;
        }
        require(
            flip.ordinal == Some(if i == 1 { 5 } else { 1 }),
            "flipped signed keys did not exchange the selected nouns",
        )?;
    }
    packet::end(&mut physical_wrong)?;
    packet::end(&mut physical_flip)
}

pub fn check(path: &Path) -> Result<()> {
    let results = path.join("scored");
    let prepared = path.join("prepared");
    let fit = rank::pooled(&fs::read(path.join("fit.words"))?)?;
    rank::grade(&prepared, &fit, &results)?;
    check_linkage_controls(&prepared, &results, &fit)?;
    let mut endpoints = BufReader::new(fs::File::open(results.join("endpoints.jsonl"))?).lines();
    let expectations = [
        [
            (Some(0), 0, 1, 7),
            (Some(2), 1, 1, 7),
            (Some(0), 0, 1, 4),
            (Some(0), 0, 1, 2),
            (Some(1), 16, 1, 2),
        ],
        [
            (Some(4), 0, 1, 7),
            (Some(4), 0, 1, 7),
            (Some(4), 0, 1, 4),
            (Some(4), 0, 1, 2),
            (Some(5), 16, 1, 2),
        ],
        [
            (None, 8, 2, 7),
            (Some(0), 8, 1, 7),
            (None, 8, 2, 4),
            (Some(0), 8, 1, 2),
            (Some(1), 8, 1, 2),
        ],
    ];
    for (i, expected) in expectations.into_iter().enumerate() {
        let record: Endpoint =
            serde_json::from_str(&endpoints.next().ok_or("fixture endpoint missing")??)?;
        let p = &record.arms;
        for (pick, expected) in [
            &p.native,
            &p.governor_only,
            &p.nouns_only,
            &p.correct_link,
            &p.exchanged_link,
        ]
        .into_iter()
        .zip(expected)
        {
            require(
                (pick.ordinal, pick.bank_cost, pick.ties, pick.eligible)
                    == (expected.0, Some(expected.1), expected.2, expected.3),
                &format!("unexpected finite score and winner at row {i}: {pick:?}"),
            )?;
        }
        require(
            p.governor_only.governor_distance == Some(0),
            "fixture zero gap was not treated as zero",
        )?;
        require(
            record.conditional_gold_heading.correct_link.ordinal == Some(0)
                && record.conditional_gold_heading.exchanged_link.ordinal == Some(1),
            "gold-heading diagnostic failed to retain the hand-checked linked orientations",
        )?;
    }
    require(endpoints.next().is_none(), "extra finite endpoint")?;
    println!("Three finite rows passed native parity with exact, exchanged, neutral, wrong-heading and tied winners. An unmatched governor key abstained on all rows, and flipped signed keys exactly exchanged both nouns.");
    Ok(())
}
