impl IntegerDiscreteModel {
    /// Study-only tracing on staged artifact code. Exhaust original exact-ID
    /// matches with the original query-site projection and D21 word bounds.
    #[doc(hidden)]
    pub fn delimiter_relaxation_study(
        &self,
        ids: &[u32],
        user: RoleSpan,
        cut: usize,
    ) -> Result<serde_json::Value> {
        use crate::span_selector::{bounds, START};
        use serde_json::json;

        paired_copy::Policy::validate_ids(ids)?;
        if !(0 < user.start && user.start < cut && cut < user.end && user.end <= ids.len()) {
            return Err(invalid("invalid study boundary"));
        }
        let (policy, paired) = self
            .paired_policy()
            .ok_or_else(|| invalid("study needs the original D21 binder"))?;
        if self.requested_role_framing().is_none()
            || self.contextual_source_arm() != Some(SourceArm::WideRecurrence)
        {
            return Err(invalid("study needs the original D26 recurrence carrier"));
        }
        let ordered_sites = self.ordered_role_sites()?;
        let query_projection = ordered_sites.prompt(ids, 1..ids.len())?;
        let source_sites = RoleSites::from_span_classes(&self.codebook_roots, &policy.span.classes);
        let source_projection = source_sites.prompt(ids, 1..ids.len())?;
        let mut exact_slices = 0;
        let mut start_losses = 0;
        let mut span_losses = 0;
        let mut gap_losses = 0;
        let mut removed_delimiter = 0;
        let mut native = 0;
        let mut traces = Vec::new();

        for query in query_projection
            .candidates()
            .filter(|q| q.start >= cut && q.end <= user.end)
        {
            let spelling = &ids[query.start..query.end];
            for start in user.start..cut {
                let end = start + spelling.len();
                if end > cut || ids[start..end] != *spelling {
                    continue;
                }
                exact_slices += 1;
                if policy.span.classes[ids[start] as usize] != START {
                    start_losses += 1;
                    continue;
                }
                let (_, last) = bounds(ids, &policy.span.classes, start);
                if last + 1 != end {
                    span_losses += 1;
                    continue;
                }
                if start >= query.start.saturating_sub(1) || last >= query.start {
                    gap_losses += 1;
                    continue;
                }

                let fact = RoleSpan { start, end };
                let delimiter = paired.delimiters.contains(&ids[start - 1]);
                let native_match = paired
                    .assertion_sites(policy, ids, query)
                    .any(|(f, l)| f == start && l + 1 == end);
                if native_match != delimiter {
                    return Err(invalid(
                        "study differed from the original D21 assertion iterator",
                    ));
                }
                let (fence, selection, candidates) = self
                    .contextual_source_observed(ids, fact, query)?
                    .ok_or_else(|| invalid("missing study carrier"))?;
                if delimiter {
                    native += 1;
                    let (strict, strict_candidates) =
                        self.contextual_source_trace(ids, fact, query)?;
                    if strict != selection || strict_candidates != candidates {
                        return Err(invalid("study trace differs from public native trace"));
                    }
                } else {
                    removed_delimiter += 1;
                    match self.contextual_source_trace(ids, fact, query) {
                        Err(err) if err.to_string().contains("original D21 bound assertion") => {}
                        _ => {
                            return Err(invalid(
                                "public trace did not reject the delimiter relaxation",
                            ))
                        }
                    }
                }
                let lexical: Vec<_> = source_projection
                    .candidates()
                    .filter(|span| {
                        fence.is_some_and(|stop| span.start >= end && span.end <= stop as usize)
                    })
                    .map(|span| [span.start, span.end])
                    .collect();
                traces.push(json!({
                    "query": [query.start, query.end], "fact": [start, end], "preceding_delimiter": delimiter,
                    "fence": fence, "lexical_spans": lexical, "selection": selection,
                    "candidates": candidates.iter().map(|(candidate, score)| json!({
                        "range": candidate.completed_source_range,
                        "eligible": score.eligible,
                        "margin": score.margin
                    })).collect::<Vec<_>>(),
                }));
            }
        }
        if exact_slices != start_losses + span_losses + gap_losses + native + removed_delimiter {
            return Err(invalid("incomplete exact-ID slice census"));
        }
        Ok(json!({
            "exact_id_slices": exact_slices, "start_losses": start_losses,
            "word_bound_losses": span_losses, "binder_gap_losses": gap_losses,
            "native_assertions": native, "delimiter_only_assertions": removed_delimiter,
            "traces": traces,
        }))
    }
}
