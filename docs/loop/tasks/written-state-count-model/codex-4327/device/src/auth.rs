use super::*;

#[derive(Serialize)]
pub(super) struct Authentication<'a> {
    train_index: usize,
    pair_id: &'a str,
    side: u32,
    supplied_opening: Step,
    supplied_period: Step,
    pub(super) supervised_b: Option<TraceStep>,
    post_b_prefix: Option<Vec<u32>>,
}

pub(super) fn authenticate<'a>(
    replay: &Replayer,
    rows: &[&'a PanelRow],
    scores: &CudaSlice<i32>,
) -> Result<Vec<Authentication<'a>>> {
    let openings = replay.released_openings(rows, scores)?;
    let eligible: Vec<_> = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.requested() > 0)
        .collect();
    let inputs = eligible
        .iter()
        .map(|&(index, _)| &openings[index].prefix)
        .collect::<Vec<_>>();
    let writers = replay.device.events(&inputs, 1)?;
    let host_scores = replay.device.stream.clone_dtoh(scores)?;
    let mut records = vec![0; eligible.len() * COMMIT_STRIDE];
    let mut decisions = Vec::with_capacity(eligible.len());
    for ((&(index, row), writer), record) in eligible
        .iter()
        .zip(&writers)
        .zip(records.chunks_exact_mut(COMMIT_STRIDE))
    {
        native_only(writer)?;
        let address = Address {
            pos: row.positions[1],
            id: row.ids[1],
        };
        let decision = Decision {
            stage: 0,
            row: u32::try_from(index)?,
            fence: u32::try_from(row.boundaries()?.0)?,
            original: address,
            eligible: Some(Gate::First),
            intervention: Some(Gate::First),
            requested_count: 1,
            learned_score: Some(host_scores[index * STAGES]),
        };
        if !decision.prepare(
            record,
            writer.native_proposal,
            &openings[index].prefix.prefix,
            address,
        ) {
            return Err(format!("TRAIN original B address refused on row {index}").into());
        }
        decisions.push(decision);
    }
    let commits = replay
        .kernel
        .select(&replay.device.stream, &records, scores)?;
    let mut proof = BTreeMap::new();
    for ((((index, _), decision), writer), commit) in eligible
        .into_iter()
        .zip(decisions)
        .zip(writers)
        .zip(commits)
    {
        let address = decision.original;
        if commit
            != (Commit {
                id: address.id,
                source: 1,
                next: 1,
                refusal: 0,
            })
        {
            return Err(
                format!("GPU did not establish an authenticated B on TRAIN row {index}").into(),
            );
        }
        let next = openings[index].prefix.self_feed(commit.id);
        let trace = decision.trace(writer, commit, address, true)?;
        proof.insert(index, (trace, next.prefix));
    }
    Ok(rows
        .iter()
        .zip(openings)
        .enumerate()
        .map(|(train_index, (row, opening))| {
            let (supervised_b, post_b_prefix) = match proof.remove(&train_index) {
                Some((trace, prefix)) => (Some(trace), Some(prefix)),
                None => (None, None),
            };
            Authentication {
                train_index,
                pair_id: &row.pair_id,
                side: row.side,
                supplied_opening: opening.opening,
                supplied_period: opening.period,
                supervised_b,
                post_b_prefix,
            }
        })
        .collect())
}
