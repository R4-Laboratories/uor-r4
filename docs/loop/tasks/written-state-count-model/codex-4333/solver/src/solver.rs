use serde::Serialize;
use std::{
    collections::BTreeMap,
    fmt::Write as _,
    io::{BufRead, BufReader, Write as _},
    process::{Command, Stdio},
};

pub const CAPACITY: usize = 32;
pub const BOUND: i64 = 4096;
type Coefficients = [i64; CAPACITY];

#[derive(Clone, Serialize)]
pub struct Inequality {
    pub case_id: String,
    pub winner: Option<usize>,
    pub loser: Option<usize>,
    pub coefficients: Coefficients,
    pub minimum: i64,
}

#[derive(Serialize)]
pub struct Term {
    inequality: Inequality,
    multiplier: String,
}

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Verdict {
    Feasible {
        words: Coefficients,
    },
    Impossible {
        terms: Vec<Term>,
        weighted_minimum: String,
        box_maximum: String,
    },
    Unknown {
        reason: String,
    },
}

pub fn meets(words: &Coefficients, inequalities: &[Inequality]) -> bool {
    words.iter().all(|w| (-BOUND..=BOUND).contains(w))
        && inequalities.iter().all(|row| {
            row.coefficients
                .iter()
                .zip(words)
                .map(|(a, w)| a * w)
                .sum::<i64>()
                >= row.minimum
        })
}

fn certify(rows: &[(&Inequality, i128)]) -> Option<Verdict> {
    let mut weighted = [0_i128; CAPACITY];
    let mut lower = 0_i128;
    let mut terms = Vec::new();
    for (row, weight) in rows {
        if *weight < 0 {
            return None;
        }
        if *weight == 0 {
            continue;
        }
        for (total, coefficient) in weighted.iter_mut().zip(row.coefficients) {
            *total = total.checked_add(weight.checked_mul(i128::from(coefficient))?)?;
        }
        lower = lower.checked_add(weight.checked_mul(i128::from(row.minimum))?)?;
        terms.push(Term {
            inequality: (*row).clone(),
            multiplier: weight.to_string(),
        });
    }
    let l1 = weighted
        .iter()
        .try_fold(0_i128, |sum, n| sum.checked_add(n.checked_abs()?))?;
    let upper = l1.checked_mul(i128::from(BOUND))?;
    (lower > upper).then(|| Verdict::Impossible {
        terms,
        weighted_minimum: lower.to_string(),
        box_maximum: upper.to_string(),
    })
}

fn numeral(value: i64) -> String {
    if value < 0 {
        format!("(- {})", value.unsigned_abs())
    } else {
        value.to_string()
    }
}

fn linear(terms: impl Iterator<Item = (i64, String)>) -> String {
    let terms: Vec<_> = terms
        .filter_map(|(coefficient, variable)| match coefficient {
            0 => None,
            1 => Some(variable),
            -1 => Some(format!("(- {variable})")),
            _ => Some(format!("(* {} {variable})", numeral(coefficient))),
        })
        .collect();
    match terms.len() {
        0 => "0".into(),
        1 => terms[0].clone(),
        _ => format!("(+ {})", terms.join(" ")),
    }
}

fn run_z3(
    z3: &str,
    script: &str,
    value_names: &str,
    want_core: bool,
) -> Result<(String, String), Box<dyn std::error::Error>> {
    let mut child = Command::new(z3)
        .args(["-in", "-smt2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let result = (|| -> Result<_, Box<dyn std::error::Error>> {
        let mut stdin = child.stdin.take().ok_or("missing Z3 stdin")?;
        let mut stdout = BufReader::new(child.stdout.take().ok_or("missing Z3 stdout")?);
        write!(stdin, "{script}\n(check-sat)\n")?;
        stdin.flush()?;
        let mut status = String::new();
        stdout.read_line(&mut status)?;
        let status = status.trim().to_owned();
        match status.as_str() {
            "sat" => writeln!(stdin, "(get-value ({value_names}))")?,
            "unsat" if want_core => writeln!(stdin, "(get-unsat-core)")?,
            "unknown" => writeln!(stdin, "(get-info :reason-unknown)")?,
            "unsat" => {}
            _ => return Err(format!("unexpected Z3 status {status:?}").into()),
        }
        writeln!(stdin, "(exit)")?;
        drop(stdin);
        let mut payload = String::new();
        std::io::Read::read_to_string(&mut stdout, &mut payload)?;
        Ok((status, payload))
    })();
    if !child.wait()?.success() {
        return Err("Z3 process failed".into());
    }
    result
}

#[derive(Debug)]
enum SExpr<'a> {
    Atom(&'a str),
    List(Vec<SExpr<'a>>),
}

fn sexpr<'a>(tokens: &mut std::slice::Iter<'_, &'a str>) -> Option<SExpr<'a>> {
    match *tokens.next()? {
        "(" => {
            let mut items = Vec::new();
            loop {
                if tokens.as_slice().first() == Some(&")") {
                    tokens.next();
                    return Some(SExpr::List(items));
                }
                items.push(sexpr(tokens)?);
            }
        }
        ")" => None,
        atom => Some(SExpr::Atom(atom)),
    }
}

fn parse_list<'a>(text: &'a str) -> Option<Vec<SExpr<'a>>> {
    let mut tokens = Vec::new();
    let mut start = 0;
    for (i, ch) in text.char_indices() {
        if ch.is_ascii_whitespace() || ch == '(' || ch == ')' {
            if i > start {
                tokens.push(&text[start..i]);
            }
            if ch == '(' || ch == ')' {
                tokens.push(&text[i..i + 1]);
            }
            start = i + 1;
        }
    }
    if start < text.len() {
        tokens.push(&text[start..]);
    }
    let mut tokens = tokens.iter();
    match sexpr(&mut tokens)? {
        SExpr::List(items) if tokens.next().is_none() => Some(items),
        _ => None,
    }
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn rational(expr: &SExpr<'_>) -> Option<(i128, i128)> {
    match expr {
        SExpr::Atom(value) => {
            if let Some((integer, fraction)) = value.split_once('.') {
                let denominator = 10_i128.checked_pow(fraction.len().try_into().ok()?)?;
                let whole: i128 = integer.parse().ok()?;
                let fractional: i128 = fraction.parse().ok()?;
                let signed = if integer.starts_with('-') {
                    fractional.checked_neg()?
                } else {
                    fractional
                };
                Some((
                    whole.checked_mul(denominator)?.checked_add(signed)?,
                    denominator,
                ))
            } else {
                Some((value.parse().ok()?, 1))
            }
        }
        SExpr::List(items) => match items.as_slice() {
            [SExpr::Atom("-"), item] => {
                let (n, d) = rational(item)?;
                Some((n.checked_neg()?, d))
            }
            [SExpr::Atom("/"), a, b] => {
                let (an, ad) = rational(a)?;
                let (bn, bd) = rational(b)?;
                Some((an.checked_mul(bd)?, ad.checked_mul(bn)?))
            }
            _ => None,
        },
    }
}

fn values(text: &str, prefix: char, count: usize) -> Option<Vec<(i128, i128)>> {
    let mut values = vec![None; count];
    for node in parse_list(text)? {
        let SExpr::List(pair) = node else { return None };
        let [SExpr::Atom(name), value] = pair.as_slice() else {
            return None;
        };
        let index: usize = name.strip_prefix(prefix)?.parse().ok()?;
        if values.get_mut(index)?.replace(rational(value)?).is_some() {
            return None;
        }
    }
    values.into_iter().collect()
}

fn dual(z3: &str, core: &[&Inequality]) -> Option<Verdict> {
    let mut script = String::from("(set-logic QF_LRA)\n(set-option :timeout 15000)\n");
    let lambda: Vec<_> = (0..core.len()).map(|i| format!("l{i}")).collect();
    for name in &lambda {
        writeln!(
            script,
            "(declare-fun {name} () Real)\n(assert (>= {name} 0))"
        )
        .ok()?;
    }
    let mut surplus = Vec::new();
    for i in 0..CAPACITY {
        let combined = linear(
            core.iter()
                .zip(&lambda)
                .map(|(row, name)| (row.coefficients[i], name.clone())),
        );
        if combined == "0" {
            continue;
        }
        let bound = format!("b{i}");
        writeln!(
            script,
            "(declare-fun {bound} () Real)\n(assert (>= {bound} {combined}))\n(assert (>= {bound} (- {combined})))"
        )
        .ok()?;
        surplus.push((-BOUND, bound));
    }
    surplus.extend(
        core.iter()
            .zip(&lambda)
            .map(|(row, name)| (row.minimum, name.clone())),
    );
    writeln!(script, "(assert (>= {} 1))", linear(surplus.into_iter())).ok()?;
    let (status, payload) = run_z3(z3, &script, &lambda.join(" "), false).ok()?;
    if status != "sat" {
        return None;
    }
    let fractions = values(&payload, 'l', core.len())?;
    let lcm = fractions.iter().try_fold(1_i128, |lcm, &(n, d)| {
        if n < 0 || d <= 0 {
            return None;
        }
        let reduced = d / gcd(n, d);
        lcm.checked_mul(reduced / gcd(lcm, reduced))
    })?;
    let weighted = core
        .iter()
        .zip(fractions)
        .map(|(&row, (n, d))| {
            let factor = gcd(n, d);
            Some((row, (n / factor).checked_mul(lcm / (d / factor))?))
        })
        .collect::<Option<Vec<_>>>()?;
    certify(&weighted)
}

fn solve_nontrivial(z3: &str, rows: &[Inequality]) -> Result<Verdict, Box<dyn std::error::Error>> {
    // The tightest row for each coefficient vector implies all weaker duplicates.
    let mut tightest = BTreeMap::<Coefficients, &Inequality>::new();
    for row in rows {
        if let Some(other) = tightest.get(&row.coefficients.map(|a| -a))
            && let Some(conflict) = certify(&[(row, 1), (*other, 1)])
        {
            return Ok(conflict);
        }
        let strongest = tightest.entry(row.coefficients).or_insert(row);
        if row.minimum > strongest.minimum {
            *strongest = row;
        }
    }
    let unique: Vec<_> = tightest.into_values().collect();
    if let (Ok(highs), Some(directory)) = (
        std::env::var("SIGNED_GATE_HIGHS"),
        std::env::var_os("SIGNED_GATE_SCRATCH"),
    ) {
        let directory = std::path::Path::new(&directory);
        if let Some(certificate) = approximate_dual(z3, &highs, directory, &unique) {
            return Ok(certificate);
        }
        if let Some(witness) = approximate_integer(&highs, directory, rows, &unique) {
            return Ok(witness);
        }
    }
    if let Some(verdict) = solve_primal(z3, rows, &unique, "QF_LRA", "Real")? {
        return Ok(verdict);
    }
    Ok(
        solve_primal(z3, rows, &unique, "QF_LIA", "Int")?.unwrap_or(Verdict::Unknown {
            reason: "no checked witness or box certificate".into(),
        }),
    )
}

fn solve_primal(
    z3: &str,
    rows: &[Inequality],
    unique: &[&Inequality],
    logic: &str,
    sort: &str,
) -> Result<Option<Verdict>, Box<dyn std::error::Error>> {
    let names: Vec<_> = (0..CAPACITY).map(|i| format!("w{i}")).collect();
    let mut script = format!(
        "(set-logic {logic})\n(set-option :timeout 15000)\n(set-option :produce-unsat-cores true)\n",
    );
    for name in &names {
        writeln!(
            script,
            "(declare-fun {name} () {sort})\n(assert (and (>= {name} {}) (<= {name} {BOUND})))",
            numeral(-BOUND)
        )?;
    }
    for (i, row) in unique.iter().enumerate() {
        let lhs = linear(
            row.coefficients
                .iter()
                .zip(&names)
                .map(|(&a, name)| (a, name.clone())),
        );
        writeln!(
            script,
            "(assert (! (>= {lhs} {}) :named r{i}))",
            numeral(row.minimum)
        )?;
    }
    let (status, payload) = run_z3(z3, &script, &names.join(" "), true)?;
    match status.as_str() {
        "sat" => {
            let mut words = [0_i64; CAPACITY];
            let values = values(&payload, 'w', CAPACITY);
            let Some(values) = values else {
                if sort == "Real" {
                    return Ok(None);
                }
                return Err("invalid Z3 integer model".into());
            };
            for (word, (n, d)) in words.iter_mut().zip(values) {
                if d <= 0 || (sort == "Int" && d != 1) {
                    return Err("invalid rational in Z3 model".into());
                }
                // Try the rounded real solution first. Only its exact integer replay counts.
                let Some(rounded) = n.checked_abs().and_then(|m| m.checked_add(d / 2)) else {
                    return Ok(None);
                };
                *word = ((rounded / d) * n.signum()).try_into()?;
            }
            if !meets(&words, rows) {
                if sort == "Real" {
                    return Ok(None);
                }
                return Err("Z3 integer witness violates an original inequality or bound".into());
            }
            Ok(Some(Verdict::Feasible { words }))
        }
        "unsat" => {
            let core = parse_list(&payload).ok_or("invalid Z3 core")?;
            let source: Vec<_> = core
                .iter()
                .map(|node| {
                    let SExpr::Atom(name) = node else { return None };
                    let i: usize = name.strip_prefix('r')?.parse().ok()?;
                    unique.get(i).copied()
                })
                .collect::<Option<_>>()
                .ok_or("invalid core reference")?;
            if let Some(certificate) = dual(z3, &source) {
                return Ok(Some(certificate));
            }
            if sort == "Real" {
                return Ok(None);
            }
            Ok(Some(Verdict::Unknown {
                reason: format!(
                    "integer solver reported unsat on {} inequalities, but no exact box certificate was obtained",
                    source.len()
                ),
            }))
        }
        _ if sort == "Real" => Ok(None),
        _ => Ok(Some(Verdict::Unknown {
            reason: payload.trim().into(),
        })),
    }
}

pub fn solve(
    z3: &str,
    rows: &[Inequality],
    zero_weight: bool,
) -> Result<Verdict, Box<dyn std::error::Error>> {
    let zero = [0; CAPACITY];
    if meets(&zero, rows) {
        return Ok(Verdict::Feasible { words: zero });
    }
    for row in rows {
        if let Some(conflict) = certify(&[(row, 1)]) {
            return Ok(conflict);
        }
    }
    if zero_weight {
        return Err("zero-weight calculation retained a nonzero coefficient".into());
    }
    solve_nontrivial(z3, rows)
}

fn lp_terms(terms: impl Iterator<Item = (i64, String)>) -> String {
    let terms: Vec<_> = terms
        .filter(|(a, _)| *a != 0)
        .map(|(a, name)| {
            let sign = if a < 0 { "-" } else { "+" };
            format!("{sign} {} {name}", a.unsigned_abs())
        })
        .collect();
    if terms.is_empty() {
        "0".into()
    } else {
        terms.join(" ")
    }
}

fn highs_hint(
    highs: &str,
    directory: &std::path::Path,
    model: &str,
    prefix: char,
    count: usize,
) -> Option<Vec<String>> {
    let base = format!("site-gate-{}", std::process::id());
    let lp = directory.join(format!("{base}.lp"));
    let sol = directory.join(format!("{base}.sol"));
    if sol.exists() {
        std::fs::remove_file(&sol).ok()?;
    }
    std::fs::write(&lp, model).ok()?;
    let result = Command::new(highs)
        .arg(&lp)
        .arg("--solution_file")
        .arg(&sol)
        .args(["--parallel", "off", "--time_limit", "5"])
        .output()
        .ok()?;
    if !result.status.success() {
        return None;
    }
    let contents = std::fs::read_to_string(sol).ok()?;
    let mut values = vec![None; count];
    let columns = contents
        .lines()
        .skip_while(|line| !line.starts_with("# Columns "));
    for line in columns.skip(1).take_while(|line| !line.starts_with('#')) {
        let Some((name, value)) = line.split_once(' ') else {
            continue;
        };
        let Some(index) = name.strip_prefix(prefix) else {
            continue;
        };
        let i: usize = index.parse().ok()?;
        values.get_mut(i)?.replace(value.trim().into());
    }
    values.into_iter().collect()
}

fn approximate_dual(
    z3: &str,
    highs: &str,
    directory: &std::path::Path,
    rows: &[&Inequality],
) -> Option<Verdict> {
    let names: Vec<_> = (0..rows.len()).map(|i| format!("l{i}")).collect();
    let mut objective: Vec<_> = rows
        .iter()
        .zip(&names)
        .map(|(r, name)| (r.minimum, name.clone()))
        .collect();
    objective.extend((0..CAPACITY).map(|i| (-BOUND, format!("t{i}"))));
    let mut lp = format!(
        "Maximize\n obj: {}\nSubject To\n cap: {} <= 1\n",
        lp_terms(objective.into_iter()),
        lp_terms(names.iter().map(|name| (1, name.clone())))
    );
    for i in 0..CAPACITY {
        for sign in [-1, 1] {
            let side = if sign < 0 { "neg" } else { "pos" };
            let lhs = lp_terms(
                rows.iter()
                    .zip(&names)
                    .map(|(r, name)| (sign * r.coefficients[i], name.clone()))
                    .chain(std::iter::once((-1, format!("t{i}")))),
            );
            writeln!(lp, " c{i}_{side}: {lhs} <= 0").ok()?;
        }
    }
    writeln!(lp, "End").ok()?;
    // Approximate values nominate a small subset only. Exact arithmetic supplies the proof.
    let hint = highs_hint(highs, directory, &lp, 'l', rows.len())?;
    let support: Vec<_> = rows
        .iter()
        .zip(hint)
        .filter(|(_, value)| value.parse::<f64>().is_ok_and(|v| v > 1e-9))
        .map(|(&row, _)| row)
        .collect();
    if support.is_empty() || support.len() > CAPACITY + 1 {
        return None;
    }
    dual(z3, &support)
}

fn approximate_integer(
    highs: &str,
    directory: &std::path::Path,
    rows: &[Inequality],
    unique: &[&Inequality],
) -> Option<Verdict> {
    let names: Vec<_> = (0..CAPACITY).map(|i| format!("w{i}")).collect();
    let mut lp = String::from("Minimize\n obj: 0 w0\nSubject To\n");
    for (i, row) in unique.iter().enumerate() {
        if row.coefficients.iter().all(|&c| c == 0) {
            continue;
        }
        let lhs = lp_terms(
            row.coefficients
                .iter()
                .zip(&names)
                .map(|(&a, name)| (a, name.clone())),
        );
        writeln!(lp, " c{i}: {lhs} >= {}", row.minimum).ok()?;
    }
    writeln!(lp, "Bounds").ok()?;
    for name in &names {
        writeln!(lp, " {} <= {name} <= {BOUND}", -BOUND).ok()?;
    }
    writeln!(lp, "General\n {}\nEnd", names.join(" ")).ok()?;
    let hinted = highs_hint(highs, directory, &lp, 'w', CAPACITY)?;
    let mut words = [0; CAPACITY];
    for (word, hint) in words.iter_mut().zip(hinted) {
        let value = hint.parse::<f64>().ok()?;
        if !value.is_finite() {
            return None;
        }
        *word = value.round() as i64;
    }
    meets(&words, rows).then_some(Verdict::Feasible { words })
}
