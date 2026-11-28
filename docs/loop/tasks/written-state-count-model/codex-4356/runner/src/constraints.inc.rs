#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct AddressObservation {
    pub prior: u64,
    pub gold: u32,
    pub rival: u32,
}
unsafe impl DeviceRepr for AddressObservation {}
unsafe impl ValidAsZeroBits for AddressObservation {}
const _: [(); 16] = [(); size_of::<AddressObservation>()];

impl Gpu {
    pub(crate) fn constraint_block(
        &self,
        capture: &Capture,
        targets: &[u16],
        support: &Support,
        first: usize,
        count: usize,
    ) -> Result<Vec<AddressObservation>> {
        if count == 0
            || first
                .checked_add(count)
                .filter(|&end| end <= targets.len())
                .is_none()
        {
            return Err(fail("invalid constraint capture block"));
        }
        let module = self.center_module()?;
        let function = module.load_function("lexical_observe_constraints")?;
        let labels = self.validated_targets(capture, targets)?;
        let map = self.map(support)?;
        let start = u32::try_from(first)?;
        let mut observations = self
            .stream
            .alloc_zeros::<AddressObservation>(count * PRIOR_WORDS)?;
        let mut launch = self.stream.launch_builder(&function);
        launch
            .arg(&labels)
            .arg(&capture.device_features)
            .arg(&capture.priors)
            .arg(&self.trunk)
            .arg(&map)
            .arg(&start)
            .arg(&mut observations);
        // SAFETY: the validated row range and one block per row cover exactly
        // count * PRIOR_WORDS outputs. Each lane owns distinct original IDs.
        unsafe {
            launch.launch(grid(u32::try_from(count)?))?;
        }
        Ok(self.stream.clone_dtoh(&observations)?)
    }
}

pub(crate) fn constraint_fixture() -> Result<(Vec<AddressObservation>, Vec<AddressObservation>)> {
    let gpu = Gpu::new(&fixture_trunk(&[(7, 0x10), (8, 0x11)]), p4_lexical_bfs())?;
    let unit = SCORE_UNIT_Q48 as u64;
    let priors = [
        fixture_priors(&[(8, 10 * unit), (7, 11 * unit)]),
        fixture_priors(&[(7, 8 * unit), (8, 8 * unit)]),
    ]
    .concat();
    let c = gpu.capture(&[vec![0, 7], vec![0, 7]], &priors)?;
    let s = gpu.support(&c)?;
    let mut blocks = gpu.constraint_block(&c, &[8, 7], &s, 0, 2)?;
    let right = blocks.split_off(PRIOR_WORDS);
    let carry = Carry::from_packed(c.features()[0].carry_code as u16);
    let packed = |root| -> Result<u32> {
        let addresses = carry.addresses(root)?;
        Ok(addresses
            .iter()
            .enumerate()
            .fold(0, |n, (bank, &a)| n | ((u32::from(a) % 64) << (6 * bank))))
    };
    if blocks[6].gold != packed([1, 1, 0, 0])?
        || blocks[6].rival != packed([0, 1, 0, 0])?
        || blocks[6].prior != 11 * unit
        || right[7].prior != 8 * unit
    {
        return Err(fail(
            "constraint device observation did not match native address/known prior",
        ));
    }
    Ok((blocks, right))
}
