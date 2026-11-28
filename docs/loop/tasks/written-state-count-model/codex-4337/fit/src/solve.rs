use crate::wire::{BOUND, Keys, MARGIN, Result, Span, WORDS, Words, invalid};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    io::{BufRead, BufReader, Write as _},
    path::Path,
    process::{Command, Stdio},
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    pub case_id: String,
    pub target: usize,
    pub rival: usize,
    pub target_span: Option<Span>,
    pub rival_span: Option<Span>,
    pub keys: Vec<(u16, i8)>,
}

impl Comparison {
    pub fn new(
        case_id: &str,
        (target, target_span, left): (usize, Option<Span>, &Keys),
        (rival, rival_span, right): (usize, Option<Span>, &Keys),
    ) -> Self {
        let mut keys = Vec::new();
        for (&a, &b) in left.iter().zip(right) {
            match a.cmp(&b) {
                std::cmp::Ordering::Less => keys.extend([(a, 1), (b, -1)]),
                std::cmp::Ordering::Greater => keys.extend([(b, -1), (a, 1)]),
                std::cmp::Ordering::Equal => {}
            }
        }
        Self {
            case_id: case_id.into(),
            target,
            rival,
            target_span,
            rival_span,
            keys,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Term {
    pub row: Comparison,
    pub multiplier: i64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Verdict {
    Feasible {
        words: Vec<i16>,
    },
    Impossible {
        terms: Vec<Term>,
        minimum: String,
        box_maximum: String,
    },
    Unknown {
        reason: String,
    },
}

pub fn minimum(words: &Words, rows: &[Comparison]) -> Option<i32> {
    rows.iter()
        .map(|row| {
            row.keys
                .iter()
                .map(|&(k, v)| i32::from(words[k as usize]) * i32::from(v))
                .sum()
        })
        .min()
}

fn valid_words(words: &[i16], rows: &[Comparison]) -> bool {
    let Ok(words) = <&Words>::try_from(words) else {
        return false;
    };
    words.iter().all(|w| (-BOUND..=BOUND).contains(w))
        && minimum(words, rows).is_some_and(|m| m >= MARGIN)
}

fn sparsify(words: &mut [i16], rows: &[Comparison]) {
    let mut use_sites = vec![Vec::new(); WORDS];
    let mut margins = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let mut margin = 0;
        for &(key, sign) in &row.keys {
            let key = usize::from(key);
            let sign = i32::from(sign);
            use_sites[key].push((i, sign));
            margin += sign * i32::from(words[key]);
        }
        margins.push(margin);
    }
    let mut order: Vec<_> = (0..WORDS).collect();
    order.sort_by_key(|&key| (use_sites[key].len(), key));
    for key in order {
        let current = i32::from(words[key]);
        if current == 0 {
            continue;
        }
        // Move toward zero as far as every original rival permits.
        let max_move = use_sites[key]
            .iter()
            .filter(|&&(_, sign)| sign == current.signum())
            .map(|&(i, _)| margins[i] - MARGIN)
            .min()
            .unwrap_or(current.abs());
        let changed = current.signum() * max_move.min(current.abs());
        words[key] = (current - changed) as i16;
        for &(i, sign) in &use_sites[key] {
            margins[i] -= sign * changed;
        }
    }
}

fn contradiction(terms: impl IntoIterator<Item = (Comparison, i64)>) -> Option<Verdict> {
    let mut weights = [0_i128; WORDS];
    let mut minimum = 0_i128;
    let mut nonzero = Vec::new();
    for (row, multiplier) in terms {
        if multiplier < 0 {
            return None;
        }
        if multiplier == 0 {
            continue;
        }
        minimum += i128::from(MARGIN) * i128::from(multiplier);
        for &(key, sign) in &row.keys {
            *weights.get_mut(usize::from(key))? += i128::from(multiplier) * i128::from(sign);
        }
        nonzero.push(Term { row, multiplier });
    }
    let box_maximum = i128::from(BOUND) * weights.iter().map(|v| v.abs()).sum::<i128>();
    (minimum > box_maximum).then_some(Verdict::Impossible {
        terms: nonzero,
        minimum: minimum.to_string(),
        box_maximum: box_maximum.to_string(),
    })
}

pub fn verify(verdict: &Verdict, rows: &[Comparison]) -> Result<()> {
    match verdict {
        Verdict::Feasible { words } if valid_words(words, rows) => Ok(()),
        Verdict::Impossible {
            terms,
            minimum,
            box_maximum,
        } => {
            let index: BTreeMap<_, _> = rows
                .iter()
                .map(|row| ((row.case_id.as_str(), row.target, row.rival), row))
                .collect();
            if terms.iter().any(|term| {
                index
                    .get(&(term.row.case_id.as_str(), term.row.target, term.row.rival))
                    .copied()
                    != Some(&term.row)
            }) {
                return Err(invalid("contradiction does not match original comparisons"));
            }
            let checked =
                contradiction(terms.iter().map(|term| (term.row.clone(), term.multiplier)));
            match checked {
                Some(Verdict::Impossible {
                    minimum: m,
                    box_maximum: b,
                    ..
                }) if m == *minimum && b == *box_maximum => Ok(()),
                _ => Err(invalid("contradiction did not replay")),
            }
        }
        Verdict::Unknown { .. } => Ok(()),
        _ => Err(invalid("training witness failed exact replay")),
    }
}

pub struct Tools<'a> {
    pub z3: &'a Path,
    pub highs: &'a Path,
    pub scratch: &'a Path,
}

fn smt<C: AsRef<str>>(z3: &Path, script: &str, names: &[C]) -> Result<(String, Vec<i64>)> {
    let mut child = Command::new(z3)
        .args(["-in", "-smt2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let result = (|| {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| invalid("missing Z3 stdin"))?;
        let mut stdout = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| invalid("missing Z3 stdout"))?,
        );
        writeln!(stdin, "{script}\n(check-sat)")?;
        stdin.flush()?;
        let mut status = String::new();
        stdout.read_line(&mut status)?;
        let status = status.trim().to_owned();
        if status == "sat" {
            let identifiers = names
                .iter()
                .map(AsRef::as_ref)
                .collect::<Vec<_>>()
                .join(" ");
            writeln!(stdin, "(get-value ({identifiers}))")?;
        }
        writeln!(stdin, "(exit)")?;
        drop(stdin);
        let mut output = String::new();
        std::io::Read::read_to_string(&mut stdout, &mut output)?;
        let mut model = Vec::new();
        match status.as_str() {
            "sat" => {
                let mut fields = output
                    .split(|c: char| c.is_ascii_whitespace() || c == '(' || c == ')')
                    .filter(|field| !field.is_empty());
                for name in names {
                    if fields.next() != Some(name.as_ref()) {
                        return Err(invalid("unexpected Z3 model identifier"));
                    }
                    model.push(
                        fields
                            .next()
                            .ok_or_else(|| invalid("missing Z3 model value"))?
                            .parse()?,
                    );
                }
                if fields.next().is_some() {
                    return Err(invalid("unexpected trailing Z3 model fields"));
                }
            }
            "unsat" | "unknown" => {}
            _ => return Err(invalid(format!("unexpected Z3 status {status:?}"))),
        }
        Ok((status, model))
    })();
    if !child.wait()?.success() {
        return Err(invalid("Z3 process failed"));
    }
    result
}

fn active_keys(rows: &[&Comparison]) -> BTreeSet<u16> {
    rows.iter()
        .flat_map(|row| row.keys.iter().map(|&(key, _)| key))
        .collect()
}

fn linear_smt(terms: impl IntoIterator<Item = (String, i8)>) -> String {
    let parts: Vec<_> = terms
        .into_iter()
        .map(|(name, sign)| {
            if sign > 0 {
                name
            } else {
                format!("(- {name})")
            }
        })
        .collect();
    match parts.as_slice() {
        [] => "0".into(),
        [single] => single.clone(),
        _ => format!("(+ {})", parts.join(" ")),
    }
}

fn exact_integer(tools: &Tools<'_>, rows: &[&Comparison]) -> Result<(String, Vec<i16>)> {
    let active = active_keys(rows);
    let names: Vec<_> = active.iter().map(|key| format!("w{key}")).collect();
    let mut script = String::from("(set-logic QF_LIA)\n(set-option :timeout 20000)\n");
    for name in &names {
        writeln!(
            script,
            "(declare-const {name} Int)\n(assert (<= 0 {name} {}))",
            2 * BOUND
        )?;
    }
    for row in rows {
        let lhs = linear_smt(
            row.keys
                .iter()
                .map(|&(key, sign)| (format!("w{key}"), sign)),
        );
        // Each retained key is paired with the opposite sign in the same bank.
        writeln!(script, "(assert (>= {lhs} {MARGIN}))")?;
    }
    let (status, model) = smt(tools.z3, &script, &names)?;
    let mut words = vec![0; WORDS];
    for (key, value) in active.into_iter().zip(model) {
        let word = i16::try_from(value - i64::from(BOUND))?;
        words[usize::from(key)] = word;
    }
    Ok((status, words))
}

fn linear_lp(terms: impl IntoIterator<Item = (String, i8)>) -> String {
    terms
        .into_iter()
        .map(|(name, sign)| {
            if sign > 0 {
                format!(" + {name}")
            } else {
                format!(" - {name}")
            }
        })
        .collect()
}

fn hint(tools: &Tools<'_>, lp: &str, kind: &str) -> Result<BTreeMap<String, String>> {
    let path = tools.scratch.join(format!("{kind}.lp"));
    let solution = tools.scratch.join(format!("{kind}.sol"));
    fs::write(&path, lp)?;
    let output = Command::new(tools.highs)
        .current_dir(tools.scratch)
        .arg(&path)
        .arg("--solution_file")
        .arg(&solution)
        .args(["--parallel", "off", "--time_limit", "10"])
        .output()?;
    fs::write(tools.scratch.join(format!("{kind}.stdout")), &output.stdout)?;
    fs::write(tools.scratch.join(format!("{kind}.stderr")), &output.stderr)?;
    if !output.status.success() || !solution.is_file() {
        return Ok(BTreeMap::new());
    }
    let data = fs::read_to_string(solution)?;
    Ok(data
        .lines()
        .skip_while(|line| !line.starts_with("# Columns "))
        .skip(1)
        .take_while(|line| !line.starts_with('#'))
        .filter_map(|line| line.split_once(' '))
        .map(|(name, value)| (name.into(), value.trim().into()))
        .collect())
}

fn primal_hint(
    tools: &Tools<'_>,
    unique: &[&Comparison],
    original: &[Comparison],
) -> Result<Option<Verdict>> {
    let active = active_keys(unique);
    let mut lp = String::from("Minimize\n objective: 0 w");
    writeln!(
        lp,
        "{}\nSubject To",
        active.first().ok_or_else(|| invalid("empty active keys"))?
    )?;
    for (i, row) in unique.iter().enumerate() {
        let lhs = linear_lp(
            row.keys
                .iter()
                .map(|&(key, sign)| (format!("w{key}"), sign)),
        );
        // A continuous hint with 17 points of slack survives rounding 32 terms to integers.
        writeln!(lp, " r{i}:{lhs} >= 17")?;
    }
    writeln!(lp, "Bounds")?;
    for key in &active {
        writeln!(lp, " {} <= w{key} <= {BOUND}", -BOUND)?;
    }
    writeln!(lp, "End")?;
    let hints = hint(tools, &lp, "primal")?;
    let mut words = vec![0; WORDS];
    for key in active {
        let value = hints
            .get(&format!("w{key}"))
            .and_then(|v| v.parse::<f64>().ok());
        let Some(value) = value.filter(|v| v.is_finite() && v.abs() <= f64::from(BOUND)) else {
            return Ok(None);
        };
        words[usize::from(key)] = value.round() as i16;
    }
    if !valid_words(&words, original) {
        return Ok(None);
    }
    sparsify(&mut words, original);
    Ok(Some(Verdict::Feasible { words }))
}

fn dual_hint(tools: &Tools<'_>, rows: &[&Comparison]) -> Result<Option<Verdict>> {
    let mut by_key = BTreeMap::<u16, Vec<(usize, i8)>>::new();
    for (i, row) in rows.iter().enumerate() {
        for &(key, sign) in &row.keys {
            by_key.entry(key).or_default().push((i, sign));
        }
    }
    let sum = linear_lp((0..rows.len()).map(|i| (format!("l{i}"), 1)));
    let mut lp = format!("Maximize\n obj:{sum}\nSubject To\n cap:{sum} <= 1\n");
    // A normalized nonnegative cancellation nominates support for an exact dual.
    for (key, terms) in &by_key {
        let lhs = linear_lp(terms.iter().map(|&(i, sign)| (format!("l{i}"), sign)));
        writeln!(lp, " k{key}:{lhs} = 0")?;
    }
    writeln!(lp, "End")?;
    let hints = hint(tools, &lp, "dual")?;
    let support: Vec<_> = rows
        .iter()
        .enumerate()
        .filter_map(|(i, &row)| {
            let value = hints.get(&format!("l{i}"))?.parse::<f64>().ok()?;
            (value > 1e-10).then_some((row, value))
        })
        .collect();
    if support.is_empty() {
        return Ok(None);
    }
    let approximate = || {
        support
            .iter()
            .map(|&(row, value)| (row.clone(), (value * 1_000_000_000.).round() as i64))
    };
    if let Some(result) = contradiction(approximate()) {
        return Ok(Some(result));
    }

    let names: Vec<_> = (0..support.len()).map(|i| format!("l{i}")).collect();
    let mut script = String::from("(set-logic QF_LIA)\n(set-option :timeout 20000)\n");
    let mut by_key = BTreeMap::<u16, Vec<(String, i8)>>::new();
    for ((row, _), name) in support.iter().zip(&names) {
        writeln!(script, "(declare-const {name} Int)\n(assert (>= {name} 1))")?;
        for &(key, sign) in &row.keys {
            by_key.entry(key).or_default().push((name.clone(), sign));
        }
    }
    for terms in by_key.into_values() {
        writeln!(script, "(assert (= {} 0))", linear_smt(terms))?;
    }
    let (status, model) = smt(tools.z3, &script, &names)?;
    Ok((status == "sat")
        .then(|| {
            contradiction(
                support
                    .iter()
                    .zip(model)
                    .map(|(&(row, _), value)| (row.clone(), value)),
            )
        })
        .flatten())
}

pub fn solve(tools: &Tools<'_>, rows: &[Comparison]) -> Result<Verdict> {
    if rows.is_empty() {
        return Ok(Verdict::Unknown {
            reason: "no registered comparisons".into(),
        });
    }
    let mut unique = BTreeMap::new();
    for row in rows {
        if row.keys.is_empty() {
            return contradiction([(row.clone(), 1)])
                .ok_or_else(|| invalid("invalid zero-vector proof"));
        }
        let opposite: Vec<_> = row.keys.iter().map(|&(k, sign)| (k, -sign)).collect();
        if let Some(&other) = unique.get(&opposite) {
            return contradiction([(row.clone(), 1), (Comparison::clone(other), 1)])
                .ok_or_else(|| invalid("invalid two-comparison proof"));
        }
        unique.entry(row.keys.clone()).or_insert(row);
    }
    let unique: Vec<_> = unique.into_values().collect();
    if let Some(witness) = primal_hint(tools, &unique, rows)? {
        return Ok(witness);
    }
    if let Some(proof) = dual_hint(tools, &unique)? {
        return Ok(proof);
    }
    let (status, mut words) = exact_integer(tools, &unique)?;
    if status == "sat" && valid_words(&words, rows) {
        sparsify(&mut words, rows);
        return Ok(Verdict::Feasible { words });
    }
    if status == "sat" {
        return Err(invalid("Z3 witness did not pass exact replay"));
    }
    Ok(Verdict::Unknown {
        reason: format!("Z3 {status}, no checked witness or integer contradiction"),
    })
}
