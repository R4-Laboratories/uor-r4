#![forbid(unsafe_code)]
#[path = "packet.rs"]
#[allow(dead_code)]
mod packet;
use serde_json::{json, Value};
use std::{env, fs, io::Write, path::Path};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn check(ok: bool, message: &str) -> Result<()> {
    if !ok {
        return Err(message.into());
    }
    Ok(())
}
fn records(path: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .map(|l| Ok(serde_json::from_str(l)?))
        .collect()
}
fn index(j: &Value, name: &str) -> Result<usize> {
    Ok(j[name]
        .as_u64()
        .ok_or_else(|| format!("invalid {name}"))?
        .try_into()?)
}
fn value(j: &Value, name: &str) -> Result<i64> {
    j[name]
        .as_i64()
        .ok_or_else(|| format!("invalid {name}").into())
}
fn run(
    text: &Path,
    mask: &Path,
    parent: &Path,
    independent: &Path,
    physical: &Path,
    receipt: &Path,
) -> Result<()> {
    let tape = packet::text(&fs::read(text)?)?;
    let near: Vec<usize> = fs::read_to_string(mask)?
        .lines()
        .map(|s| match s {
            "0" => Ok(0),
            "1" => Ok(1),
            _ => Err("invalid distance mask"),
        })
        .collect::<std::result::Result<_, _>>()?;
    let edges = records(independent)?;
    let votes = records(&physical.join("votes.jsonl"))?;
    let quality = records(&physical.join("quality.jsonl"))?;
    let summaries = records(&physical.join("vote-summary.jsonl"))?;
    check(
        tape.len() == near.len()
            && edges.len() == tape.len() * 2
            && quality.len() == 9
            && summaries.len() == 8,
        "incomplete input",
    )?;
    let mut expected = fs::read(parent)?;
    let mut vindex = 0;
    let mut result = Vec::new();
    for step in 0..=8 {
        check(
            expected == fs::read(physical.join(format!("step-{step}.state")))?,
            "physical checkpoint bytes disagree",
        )?;
        let banks = packet::state(&expected)?;
        let mut separate: [Vec<[[i32; 2]; packet::EDGE]>; 2] =
            std::array::from_fn(|s| vec![[[0; 2]; packet::EDGE]; banks[s].transport.len()]);
        let mut strata = [[0i64; 4]; 2]; // hinge, wins, losses, eligible
        for (i, edge) in edges.iter().enumerate() {
            let row = i / 2;
            let sign = i % 2;
            check(
                index(edge, "row")? == row && index(edge, "sign")? == sign,
                "signed identity changed",
            )?;
            let Some(true) = edge["eligible_for_fit"].as_bool() else {
                continue;
            };
            let read = &edge["history"][step];
            let margin = value(read, "margin")?;
            let side = near[row];
            let scores = &mut strata[side];
            scores[0] += (1 - margin).max(0);
            scores[1] += i64::from(margin > 0);
            scores[2] += i64::from(margin < 0);
            scores[3] += 1;
            if step == 8 || margin > 0 {
                continue;
            }
            let info = &tape[row];
            let governor = index(edge, "governor_word")?;
            for (site, center, weight) in [
                (index(edge, "gold_site")?, index(read, "gold_center")?, 1),
                (index(read, "rival_site")?, index(read, "rival_center")?, -1),
            ] {
                let coord = info
                    .edges
                    .get(governor * info.sites.len() + site)
                    .ok_or("bad site")?;
                let bank = banks[sign].transport.get(center).ok_or("bad center")?;
                for (d, (&sample, &state)) in coord.iter().zip(bank).enumerate() {
                    separate[sign][center][d][side] +=
                        weight * (i32::from(sample) - i32::from(state)).signum();
                }
            }
        }
        let q = &quality[step];
        check(index(q, "step")? == step, "bad quality step")?;
        for (side, name) in ["far", "near"].iter().enumerate() {
            for (n, k) in ["hinge", "strict_wins", "strict_losses", "eligible"]
                .iter()
                .enumerate()
            {
                check(
                    value(&q[name], k)? == strata[side][n],
                    "stratum quality mismatch",
                )?;
            }
        }
        for (n, k) in ["hinge", "strict_wins", "strict_losses"].iter().enumerate() {
            check(
                value(q, k)? == strata[0][n] + strata[1][n],
                "total quality mismatch",
            )?;
        }
        if step == 8 {
            break;
        }
        let mut counts = [0usize; 6]; // opposed, cancelled, agree, near_only, far_only, zero
        let mut moved = 0;
        let mut blocked_by_near = 0;
        let mut blocked_by_far = 0;
        for (s, slots) in separate.iter().enumerate() {
            for (slot, axes) in slots.iter().enumerate() {
                for (axis, &[far, near]) in axes.iter().enumerate() {
                    let v = votes.get(vindex).ok_or("missing coordinate")?;
                    vindex += 1;
                    check(
                        [
                            index(v, "step")?,
                            index(v, "sign")?,
                            index(v, "slot")?,
                            index(v, "axis")?,
                        ] == [step, s, slot, axis]
                            && value(v, "near")? == i64::from(near)
                            && value(v, "far")? == i64::from(far),
                        "opposing vote evidence mismatch",
                    )?;
                    match (far.signum(), near.signum()) {
                        (0, 0) => counts[5] += 1,
                        (0, _) => counts[3] += 1,
                        (_, 0) => counts[4] += 1,
                        (f, n) if f == n => counts[2] += 1,
                        _ => {
                            counts[0] += 1;
                            counts[1] += usize::from(far + near == 0);
                            blocked_by_near += usize::from((far + near).signum() != far.signum());
                            blocked_by_far += usize::from((far + near).signum() != near.signum());
                        }
                    }
                    let offset = 8 + 6664 * s + 6152 + 2 * (slot * packet::EDGE + axis);
                    let old = banks[s].transport[slot][axis];
                    let next = old.saturating_add((far + near).signum() as i16);
                    expected[offset..offset + 2].copy_from_slice(&next.to_le_bytes());
                    moved += usize::from(next != old);
                }
            }
        }
        let summary = &summaries[step];
        check(index(summary, "step")? == step, "bad vote summary step")?;
        for (i, k) in [
            "opposed",
            "cancelled",
            "agree",
            "near_only",
            "far_only",
            "zero",
        ]
        .iter()
        .enumerate()
        {
            check(index(summary, k)? == counts[i], "bad conflict summary")?;
        }
        result.push(json!({"step":step,"opposed":counts[0],"cancelled":counts[1],"agree":counts[2],"near_only":counts[3],
          "far_only":counts[4],"inactive":counts[5],"moved":moved,"far_direction_blocked_or_reversed":blocked_by_near,
          "near_direction_blocked_or_reversed":blocked_by_far}));
    }
    check(vindex == votes.len(), "unexpected additional coordinates")?;
    let mut file = fs::File::create_new(receipt)?;
    serde_json::to_writer_pretty(
        &mut file,
        &json!({"rows":tape.len(),"signed_edges":edges.len(),
      "checkpoint_bytes_verified":9*expected.len(),"individual_coordinate_records_verified":vindex,
      "quality_snapshots_verified":9,"vote_steps":result,"disagreements":0}),
    )?;
    writeln!(file)?;
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = env::args_os()
        .skip(1)
        .map(std::path::PathBuf::from)
        .collect();
    let [text, mask, parent, independent, physical, receipt] = args.as_slice() else {
        return Err(
            "usage: replay TEXT MASK PARENT INDEPENDENT_SIGNED_READS PHYSICAL_DIR NEW_RECEIPT"
                .into(),
        );
    };
    run(text, mask, parent, independent, physical, receipt)
}
