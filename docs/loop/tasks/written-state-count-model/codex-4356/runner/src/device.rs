// Reuse the sealed CUDA capture, original election, and checkpoint scorer.
include!("../../../codex-4355/runner/src/device.rs");

const CENTER_CUBIN: &[u8] = include_bytes!(env!("CENTERED_ELECTION_CUBIN"));

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct BankShift {
    pub score_q48: i64,
    pub offsets: [i32; 4],
    pub changed_words: u32,
    pub l1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct SparsePrecheck {
    pub before: Decision,
    pub centered: Decision,
    pub before_ceiling: Decision,
    pub centered_ceiling: Decision,
    pub shift_failures: u32,
    pub tied_aliases: u32,
    pub blocking_aliases: u32,
    #[serde(skip)]
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct CenteredVisit {
    pub learning: ElectionVisit,
    pub before_center: Decision,
    pub gauge: BankShift,
    pub gold_before_q48: i64,
    pub gold_centered_q48: i64,
    pub clipped_learning_words: u32,
    pub shift_failure: u32,
}

// SAFETY: these repr(C) structs match recenter.cu and admit all-zero bits.
// SparsePrecheck and its retained election ABI explicitly reserve alignment bytes.
unsafe impl DeviceRepr for BankShift {}
unsafe impl ValidAsZeroBits for BankShift {}
unsafe impl DeviceRepr for SparsePrecheck {}
unsafe impl ValidAsZeroBits for SparsePrecheck {}
unsafe impl DeviceRepr for CenteredVisit {}
unsafe impl ValidAsZeroBits for CenteredVisit {}
const _: [(); 32] = [(); size_of::<BankShift>()];
const _: [(); 112] = [(); size_of::<SparsePrecheck>()];
const _: [(); 152] = [(); size_of::<CenteredVisit>()];

#[derive(Serialize)]
pub struct FrozenSparse {
    #[serde(serialize_with = "serialize_weights")]
    pub weights: [i16; WEIGHT_WORDS],
    pub gauge: BankShift,
    pub rows: Vec<SparsePrecheck>,
}

#[derive(Serialize)]
pub struct CenterEpoch {
    pub stats: ElectionStats,
    #[serde(serialize_with = "serialize_weights")]
    pub weights: [i16; WEIGHT_WORDS],
    pub visits: Vec<CenteredVisit>,
    pub checkpoint: Vec<Decision>,
    /// Synchronized fit launch wall time, excluding uploads and readback.
    pub elapsed_ns: u128,
}

#[derive(Serialize)]
pub struct CenterFit {
    #[serde(serialize_with = "serialize_weights")]
    pub weights: [i16; WEIGHT_WORDS],
    pub epochs: Vec<CenterEpoch>,
}

fn structural_mask() -> u64 {
    p4_lexical_bfs()
        .iter()
        .fold(0, |mask, &slot| mask | (1u64 << slot))
}

fn structural_word(word: usize, slots: u64) -> bool {
    slots & (1u64 << (word % 64)) != 0
}

fn validate_center_support(support: &Support, slots: u64) -> Result<()> {
    if slots.count_ones() != 26 || support.geometry.iter().any(|&mask| mask & !slots != 0) {
        return Err(fail("invalid structural lexical bank support"));
    }
    // Gpu::map additionally requires ordered support to match geometry.
    Ok(())
}

fn shift_receipt(offsets: [i32; 4], slots: u64) -> BankShift {
    let width = slots.count_ones();
    BankShift {
        score_q48: offsets.into_iter().map(i64::from).sum::<i64>() * SCORE_UNIT_Q48,
        offsets,
        changed_words: width * offsets.iter().filter(|&&offset| offset != 0).count() as u32,
        l1: width * offsets.into_iter().map(i32::unsigned_abs).sum::<u32>(),
    }
}

fn validate_frozen_shift(
    before: &[i16; WEIGHT_WORDS],
    after: &[i16; WEIGHT_WORDS],
    gauge: BankShift,
    slots: u64,
) -> Result<()> {
    let mut offsets = [0; 4];
    for (bank, values) in before.chunks_exact(64).enumerate() {
        let mut supported = values
            .iter()
            .enumerate()
            .filter_map(|(slot, &value)| structural_word(slot, slots).then_some(i32::from(value)));
        let first = supported
            .next()
            .ok_or_else(|| fail("empty structural bank"))?;
        let (low, high) = supported.fold((first, first), |(low, high), value| {
            (low.min(value), high.max(value))
        });
        let midpoint = (i32::from(i16::MIN) + i32::from(i16::MAX) - low - high) / 2;
        offsets[bank] = midpoint;
        for (slot, &value) in values.iter().enumerate() {
            let expected = i32::from(value)
                + if structural_word(slot, slots) {
                    midpoint
                } else {
                    0
                };
            if i32::from(after[bank * 64 + slot]) != expected {
                return Err(fail(format!("invalid translated bank {bank} slot {slot}")));
            }
        }
    }
    if gauge != shift_receipt(offsets, slots) {
        return Err(fail(
            "frozen bank shift receipt disagrees with translated words",
        ));
    }
    Ok(())
}

fn validate_center_visits(
    visits: &[CenteredVisit],
    targets: &[u16],
    policy: ElectionPolicy,
    support: &Support,
    slots: u64,
    enabled: bool,
) -> Result<ElectionStats> {
    for (visit, &target) in visits.iter().zip(targets) {
        validate_decision(visit.before_center, target)?;
        let gauge = visit.gauge;
        if visit.shift_failure != 0
            || visit.before_center != visit.learning.before
            || visit.gold_centered_q48.checked_sub(visit.gold_before_q48) != Some(gauge.score_q48)
            || gauge
                .offsets
                .iter()
                .any(|&offset| !(-32767..=32767).contains(&offset))
            || gauge != shift_receipt(gauge.offsets, slots)
            || (!enabled && gauge != BankShift::default())
            || visit.clipped_learning_words > visit.learning.saturated_words
        {
            return Err(fail(format!(
                "invalid centered visit for token {target}: {visit:?}"
            )));
        }
    }
    let learning = visits
        .iter()
        .map(|visit| visit.learning)
        .collect::<Vec<_>>();
    summarize_visits(&learning, targets, policy, support)
}

impl Gpu {
    fn center_module(&self) -> Result<Arc<cudarc::driver::CudaModule>> {
        Ok(self
            .stream
            .context()
            .load_module(Ptx::from_binary(CENTER_CUBIN.to_vec()))?)
    }

    pub fn center_digest(&self) -> String {
        hex::encode(Sha256::digest(CENTER_CUBIN))
    }

    pub fn fit_center(
        &self,
        capture: &Capture,
        targets: &[u16],
        orders: &[Vec<u32>],
        mode: Mode,
        support: &Support,
        policy: ElectionPolicy,
        enabled: bool,
    ) -> Result<CenterFit> {
        self.fit_center_from(
            capture,
            targets,
            orders,
            mode,
            support,
            policy,
            enabled,
            &[0; WEIGHT_WORDS],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn fit_center_from(
        &self,
        capture: &Capture,
        targets: &[u16],
        orders: &[Vec<u32>],
        mode: Mode,
        support: &Support,
        policy: ElectionPolicy,
        enabled: bool,
        starting_weights: &[i16; WEIGHT_WORDS],
    ) -> Result<CenterFit> {
        if orders.is_empty() || targets.is_empty() {
            return Err(fail("centered election requires visit orders and targets"));
        }
        for order in orders {
            let mut seen = vec![false; targets.len()];
            if order.len() != targets.len()
                || order.iter().any(|&row| {
                    !seen
                        .get_mut(row as usize)
                        .is_some_and(|slot| !std::mem::replace(slot, true))
                })
            {
                return Err(fail("centered election epoch must visit each row once"));
            }
        }
        let slots = structural_mask();
        validate_center_support(support, slots)?;
        let device_targets = self.validated_targets(capture, targets)?;
        let map = self.map(support)?;
        let module = self.center_module()?;
        let fit = module.load_function("centered_election_fit")?;
        let mut weights = self.stream.clone_htod(starting_weights)?;
        let mut visits = self.stream.alloc_zeros::<CenteredVisit>(targets.len())?;
        let n = u32::try_from(targets.len())?;
        let mode = mode as u32;
        let device_policy = policy as u32;
        let device_enabled = u32::from(enabled);
        let mut epochs = Vec::with_capacity(orders.len());
        for order in orders {
            let device_order = self.stream.clone_htod(order)?;
            self.stream.synchronize()?;
            let mut launch = self.stream.launch_builder(&fit);
            launch
                .arg(&n)
                .arg(&device_order)
                .arg(&device_targets)
                .arg(&capture.device_features)
                .arg(&capture.priors)
                .arg(&self.trunk)
                .arg(&mode)
                .arg(&map)
                .arg(&device_policy)
                .arg(&device_enabled)
                .arg(&mut weights)
                .arg(&mut visits);
            let start = std::time::Instant::now();
            // SAFETY: each of the validated rows occurs once in the uploaded
            // order. The single 128-thread block owns the persistent words.
            unsafe { launch.launch(grid(1))? };
            self.stream.synchronize()?;
            let elapsed_ns = start.elapsed().as_nanos();
            let epoch_words = self.read_weights(&weights)?;
            if epoch_words.iter().enumerate().any(|(word, &value)| {
                let allowed = if enabled {
                    slots
                } else {
                    support.geometry[word / 64]
                };
                !structural_word(word, allowed) && value != starting_weights[word]
            }) {
                return Err(fail("centered election wrote outside its bank support"));
            }
            let visits = self.stream.clone_dtoh(&visits)?;
            let stats = validate_center_visits(&visits, targets, policy, support, slots, enabled)?;
            let (checkpoint, _) = self.scan_checkpoint(
                capture,
                targets,
                &device_targets,
                mode,
                &map,
                &weights,
                false,
            )?;
            let last_row = order[order.len() - 1] as usize;
            if checkpoint[last_row] != visits[last_row].learning.after {
                return Err(fail(
                    "centered last visit disagrees with the independent checkpoint",
                ));
            }
            epochs.push(CenterEpoch {
                stats,
                weights: epoch_words,
                visits,
                checkpoint,
                elapsed_ns,
            });
        }
        let weights = epochs
            .last()
            .ok_or_else(|| fail("missing centered epochs"))?
            .weights;
        Ok(CenterFit { weights, epochs })
    }

    pub fn frozen_sparse(
        &self,
        capture: &Capture,
        targets: &[u16],
        mode: Mode,
        support: &Support,
        weights: &[i16; WEIGHT_WORDS],
    ) -> Result<FrozenSparse> {
        let slots = structural_mask();
        validate_center_support(support, slots)?;
        let device_targets = self.validated_targets(capture, targets)?;
        let map = self.map(support)?;
        let module = self.center_module()?;
        let center = module.load_function("lexical_center_banks")?;
        let precheck = module.load_function("lexical_sparse_precheck")?;
        let before = self.stream.clone_htod(weights)?;
        let mut after = self.stream.alloc_zeros::<i16>(WEIGHT_WORDS)?;
        let mut gauge = self.stream.alloc_zeros::<BankShift>(1)?;
        let mut output = self.stream.alloc_zeros::<SparsePrecheck>(targets.len())?;
        let mut launch = self.stream.launch_builder(&center);
        launch.arg(&before).arg(&mut after).arg(&mut gauge);
        // SAFETY: one block reads and writes a complete 256-word state and
        // writes the single provided shift receipt.
        unsafe { launch.launch(grid(1))? };
        let mode = mode as u32;
        let mut launch = self.stream.launch_builder(&precheck);
        launch
            .arg(&device_targets)
            .arg(&capture.device_features)
            .arg(&capture.priors)
            .arg(&self.trunk)
            .arg(&mode)
            .arg(&map)
            .arg(&before)
            .arg(&after)
            .arg(&gauge)
            .arg(&mut output);
        // SAFETY: one block per validated row, with disjoint output entries
        // and read-only bank words and receipt from the preceding launch.
        unsafe { launch.launch(grid(u32::try_from(targets.len())?))? };
        let gauge = self.stream.clone_dtoh(&gauge)?[0];
        let shifted = self.read_weights(&after)?;
        validate_frozen_shift(weights, &shifted, gauge, slots)?;
        let rows = self.stream.clone_dtoh(&output)?;
        let (reference, _) = self.scan_checkpoint(
            capture,
            targets,
            &device_targets,
            mode,
            &map,
            &before,
            false,
        )?;
        for ((row, &target), &decision) in rows.iter().zip(targets).zip(&reference) {
            for actual in [
                row.before,
                row.centered,
                row.before_ceiling,
                row.centered_ceiling,
            ] {
                validate_decision(actual, target)?;
            }
            if row.before != decision
                || row.before != row.centered
                || row.shift_failures != 0
                || row.reserved != 0
                || row.tied_aliases > row.blocking_aliases
                || row.blocking_aliases >= PRIOR_WORDS as u32
                || row.before_ceiling.margin_q48 < row.before.margin_q48
                || row.centered_ceiling.margin_q48 < row.centered.margin_q48
                || row.before_ceiling.rank > row.before.rank
                || row.centered_ceiling.rank > row.centered.rank
                || (row.blocking_aliases > 0
                    && (row.before_ceiling.margin_q48 > 0 || row.centered_ceiling.margin_q48 > 0))
            {
                return Err(fail(format!(
                    "invalid frozen sparse comparison for token {target}: {row:?}"
                )));
            }
        }
        Ok(FrozenSparse {
            weights: shifted,
            gauge,
            rows,
        })
    }
}

fn finite_frozen_shifts(gpu: &Gpu, capture: &Capture, support: &Support) -> Result<()> {
    let mask = structural_mask();
    let low = mask.trailing_zeros() as usize;
    let high = (mask & (mask - 1)).trailing_zeros() as usize;
    let targets = [9];
    for sign in [-1, 1] {
        let constants = [100 * sign, -200 * sign, 305 * sign, -405 * sign];
        let mut midpoint = [24_719; WEIGHT_WORDS];
        let mut shifted = midpoint;
        for (bank, constant) in constants.into_iter().enumerate() {
            for slot in 0..64 {
                if structural_word(slot, mask) {
                    let reference = if slot == low {
                        -9
                    } else if slot == high {
                        8
                    } else {
                        0
                    };
                    midpoint[64 * bank + slot] = reference;
                    shifted[64 * bank + slot] = reference + constant;
                }
            }
        }
        let expected = shift_receipt(constants.map(|value| -i32::from(value)), mask);
        for mode in [Mode::Geometry, Mode::Ordered] {
            let result = gpu.frozen_sparse(capture, &targets, mode, support, &shifted)?;
            let idempotent =
                gpu.frozen_sparse(capture, &targets, mode, support, &result.weights)?;
            let mut original = gpu.witness(capture, &targets, mode, support, &shifted)?[0];
            let centered = gpu.witness(capture, &targets, mode, support, &result.weights)?[0];
            original.winner_score_q48 += expected.score_q48;
            if result.weights != midpoint
                || result.gauge != expected
                || original != centered
                || idempotent.weights != midpoint
                || idempotent.gauge != BankShift::default()
            {
                return Err(fail(format!(
                    "finite full-score shift disagreed under {mode:?}"
                )));
            }
        }
    }
    let mut extrema = [0; WEIGHT_WORDS];
    for bank in 0..4 {
        extrema[64 * bank + low] = i16::MIN;
        extrema[64 * bank + high] = i16::MAX;
    }
    let frozen = gpu.frozen_sparse(capture, &targets, Mode::Geometry, support, &extrema)?;
    if frozen.weights != extrema
        || frozen.gauge != BankShift::default()
        || frozen.rows[0].tied_aliases == 0
        || frozen.rows[0].centered_ceiling.margin_q48 != 0
    {
        return Err(fail(
            "finite extrema or impossible original-ID ties changed",
        ));
    }
    Ok(())
}

fn finite_off_equality(new: &CenterFit, reference: &ElectionFit) -> Result<()> {
    if new.weights != reference.weights || new.epochs.len() != reference.epochs.len() {
        return Err(fail("finite instrumented-off final state changed"));
    }
    for (epoch, old) in new.epochs.iter().zip(&reference.epochs) {
        if epoch.weights != old.weights
            || epoch.checkpoint != old.checkpoint
            || epoch
                .visits
                .iter()
                .map(|visit| visit.learning)
                .ne(old.visits.iter().copied())
        {
            return Err(fail("finite instrumented-off epoch changed"));
        }
    }
    Ok(())
}

pub fn finite_center() -> Result<()> {
    let unit = SCORE_UNIT_Q48 as u64;
    let base = 100 * unit;
    let prefix = vec![0, 7];
    let trunk = fixture_trunk(&[(7, 0x10), (8, 0x11), (10, 0x11), (12, 0x01)]);
    let gpu = Gpu::new(&trunk, p4_lexical_bfs())?;
    if gpu.center_digest() == gpu.election_digest() {
        return Err(fail("centered and sealed election modules must differ"));
    }
    let ties = gpu.capture(std::slice::from_ref(&prefix), &vec![1; PRIOR_WORDS])?;
    let support = gpu.support(&ties)?;
    finite_frozen_shifts(&gpu, &ties, &support)?;

    let carry = Carry::from_packed(ties.features()[0].carry_code as u16);
    let gold = carry.addresses([0; 4])?.map(usize::from);
    let rival = carry.addresses([1, 0, 0, 0])?.map(usize::from);
    let geometry = Mode::Geometry;
    let targets = [9];
    let orders = [vec![0]];
    let mask = structural_mask();
    let displaced = std::array::from_fn(|word| if structural_word(word, mask) { 215 } else { 37 });
    let strictly_winning =
        gpu.capture(std::slice::from_ref(&prefix), &fixture_priors(&[(9, base)]))?;
    let undisplaced = std::array::from_fn(|word| if structural_word(word, mask) { 0 } else { 37 });
    for (capture, outcome) in [(&ties, INFEASIBLE), (&strictly_winning, ALREADY_STRICT)] {
        let fit = gpu.fit_center_from(
            capture,
            &targets,
            &[vec![0], vec![0]],
            geometry,
            &support,
            ElectionPolicy::GoldOnly,
            true,
            &displaced,
        )?;
        let first = fit.epochs[0].visits[0];
        let repeat = fit.epochs[1].visits[0];
        if first.learning.outcome != outcome
            || first.gauge != shift_receipt([-215; 4], mask)
            || fit.epochs[0].weights != undisplaced
            || repeat.learning != first.learning
            || repeat.gauge != BankShift::default()
            || fit.weights != undisplaced
        {
            return Err(fail("finite no-learning visit failed to center its banks"));
        }
    }
    let clipped_case = gpu.capture(
        std::slice::from_ref(&prefix),
        &fixture_priors(&[(9, base), (12, base + unit)]),
    )?;
    let mut at_limit = [0; WEIGHT_WORDS];
    at_limit[gold[0]] = i16::MAX;
    at_limit[rival[0]] = i16::MAX;
    // Gold in bank 1 arrives exactly at the endpoint for the old full update.
    at_limit[gold[1]] = i16::MAX - 2;
    let sparse = gpu.frozen_sparse(&clipped_case, &targets, geometry, &support, &at_limit)?;
    let row = sparse.rows[0];
    if row.before_ceiling.margin_q48 != -SCORE_UNIT_Q48 || row.centered_ceiling.margin_q48 <= 0 {
        return Err(fail(
            "finite sparse headroom did not unlock a strict margin",
        ));
    }
    let opposed = gpu.capture(
        &[prefix.clone(), prefix.clone()],
        &fixture_priors(&[(9, base), (8, base)]).repeat(2),
    )?;
    let opposed_orders = [vec![0, 1], vec![1, 0]];
    for policy in [ElectionPolicy::GoldOnly, ElectionPolicy::Full] {
        for mode in [Mode::Geometry, Mode::Ordered] {
            let old = gpu.fit_all(&opposed, &[9, 8], &opposed_orders, mode, &support, policy)?;
            let off = gpu.fit_center(
                &opposed,
                &[9, 8],
                &opposed_orders,
                mode,
                &support,
                policy,
                false,
            )?;
            finite_off_equality(&off, &old)?;
        }
        for deficit in [2 * unit - 1, 2 * unit] {
            let capture = gpu.capture(
                std::slice::from_ref(&prefix),
                &fixture_priors(&[(9, base), (12, base + deficit)]),
            )?;
            let old = gpu.fit_all_from(
                &capture, &targets, &orders, geometry, &support, policy, &displaced,
            )?;
            let off = gpu.fit_center_from(
                &capture, &targets, &orders, geometry, &support, policy, false, &displaced,
            )?;
            let on = gpu.fit_center_from(
                &capture, &targets, &orders, geometry, &support, policy, true, &displaced,
            )?;
            finite_off_equality(&off, &old)?;
            let visit = on.epochs[0].visits[0];
            let gain = if policy == ElectionPolicy::Full { 2 } else { 1 };
            let radius = 2 / gain + u32::from(deficit == 2 * unit);
            let margin = i64::from(radius * gain) * SCORE_UNIT_Q48 - i64::try_from(deficit)?;
            if visit.learning != off.epochs[0].visits[0].learning
                || visit.learning.before.rival != 12
                || visit.learning.after.rival != 12
                || visit.learning.radius != radius
                || visit.learning.after.margin_q48 != margin
                || visit.clipped_learning_words != 0
                || visit.gauge.offsets != [-215; 4]
                || on.epochs[0].checkpoint != off.epochs[0].checkpoint
                || on.weights.iter().enumerate().any(|(word, &value)| {
                    let offset = if structural_word(word, mask) { -215 } else { 0 };
                    i32::from(value) != i32::from(off.weights[word]) + offset
                })
            {
                return Err(fail(format!("finite untranslated learning changed raw-Q48 contrasts under {policy:?}: {visit:?}")));
            }
        }
        let old = gpu.fit_all_from(
            &clipped_case,
            &targets,
            &orders,
            geometry,
            &support,
            policy,
            &at_limit,
        )?;
        let off = gpu.fit_center_from(
            &clipped_case,
            &targets,
            &orders,
            geometry,
            &support,
            policy,
            false,
            &at_limit,
        )?;
        let on = gpu.fit_center_from(
            &clipped_case,
            &targets,
            &orders,
            geometry,
            &support,
            policy,
            true,
            &at_limit,
        )?;
        finite_off_equality(&off, &old)?;
        let original = off.epochs[0].visits[0];
        let centered = on.epochs[0].visits[0];
        let expected = match policy {
            ElectionPolicy::GoldOnly => {
                original.learning.outcome == INFEASIBLE
                    && original.clipped_learning_words == 0
                    && centered.learning.radius == 2
            }
            ElectionPolicy::Full => {
                original.learning.outcome == UPDATED
                    && original.learning.saturated_words == 2
                    && original.clipped_learning_words == 1
                    && original.learning.radius == 2
                    && centered.learning.radius == 1
            }
        };
        if !expected
            || centered.learning.outcome != UPDATED
            || centered.learning.before != original.learning.before
            || centered.learning.after.margin_q48 != SCORE_UNIT_Q48
            || centered.clipped_learning_words != 0
        {
            return Err(fail(format!("finite causal clipping control failed under {policy:?}: off {original:?}, on {centered:?}")));
        }
    }
    crate::conflicts::finite_device()
}

include!("constraints.inc.rs");
