#![forbid(unsafe_code)]

const WIDTH: usize = 64;
const ACT_CLAMP: i32 = 1 << 22;

#[derive(Debug, PartialEq, Eq)]
pub enum MarginError {
    InadmissibleTarget,
    InvalidConfiguration,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Margin {
    pub rival: usize,
    pub gap_q15: i64,
    pub loss_q15: i64,
    pub derivative_q15: i32,
}

pub fn strongest_rival_margin(
    scores: &[i32; WIDTH],
    eligible: u64,
    target: usize,
    margin_q15: i32,
    gain_q15: i32,
) -> Result<Option<Margin>, MarginError> {
    if target >= WIDTH || eligible & (1 << target) == 0 {
        return Err(MarginError::InadmissibleTarget);
    }
    if !(1..=ACT_CLAMP).contains(&margin_q15) || !(1..=ACT_CLAMP).contains(&gain_q15) {
        return Err(MarginError::InvalidConfiguration);
    }
    let mut rival = None;
    for slot in 0..WIDTH {
        if slot == target || eligible & (1u64 << slot) == 0 {
            continue;
        }
        if rival.is_none_or(|previous| scores[slot] > scores[previous]) {
            rival = Some(slot);
        }
    }
    Ok(rival.map(|rival| {
        let gap_q15 = i64::from(scores[target]) - i64::from(scores[rival]);
        let deficit = i64::from(margin_q15) - gap_q15;
        Margin {
            rival,
            gap_q15,
            loss_q15: (deficit.max(0) * i64::from(gain_q15) + (1 << 14)) >> 15,
            derivative_q15: if deficit > 0 { gain_q15 } else { 0 },
        }
    }))
}

pub fn add_to_legacy(
    gradients: &mut [i32; WIDTH],
    target: usize,
    margin: &Margin,
) -> Result<u8, MarginError> {
    if target >= WIDTH || margin.rival >= WIDTH || target == margin.rival {
        return Err(MarginError::InadmissibleTarget);
    }
    let gain = i64::from(margin.derivative_q15);
    let mut clipped = 0;
    for (slot, delta) in [(target, -gain), (margin.rival, gain)] {
        let merged = i64::from(gradients[slot]) + delta;
        let bounded = merged.clamp(-i64::from(ACT_CLAMP), i64::from(ACT_CLAMP));
        clipped += u8::from(bounded != merged);
        gradients[slot] = bounded as i32;
    }
    Ok(clipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_follow_serving_slot_order_and_ignore_ineligible_scores() {
        let mut scores = [0; WIDTH];
        scores[0] = ACT_CLAMP;
        let rank = strongest_rival_margin(&scores, 0b1110, 3, 1, 32768)
            .unwrap()
            .unwrap();
        assert_eq!(
            rank,
            Margin {
                rival: 1,
                gap_q15: 0,
                loss_q15: 1,
                derivative_q15: 32768
            }
        );
        let mut dz = [0; WIDTH];
        assert_eq!(add_to_legacy(&mut dz, 3, &rank), Ok(0));
        assert_eq!(&dz[..4], &[0, 32768, 0, -32768]);
    }

    #[test]
    fn range_boundary_and_reciprocal_direction() {
        let mut a = [0; WIDTH];
        a[63] = i32::MAX;
        a[0] = i32::MIN;
        let mask = (1 << 63) | 1;
        let rank = strongest_rival_margin(&a, mask, 0, 1, 32768)
            .unwrap()
            .unwrap();
        assert_eq!(rank.gap_q15, -4_294_967_295);
        assert_eq!(rank.loss_q15, 4_294_967_296);
        let mut dz = [-ACT_CLAMP; WIDTH];
        assert_eq!(add_to_legacy(&mut dz, 0, &rank), Ok(1));

        let mut b = [0; WIDTH];
        b[63] = 1;
        let rank = strongest_rival_margin(&b, mask, 63, 1, 32768)
            .unwrap()
            .unwrap();
        assert_eq!(rank.gap_q15, 1);
        assert_eq!(rank.loss_q15, 0);
        assert_eq!(rank.derivative_q15, 0);
    }

    #[test]
    fn missing_rival_and_invalid_inputs_have_distinct_outcomes() {
        let scores = [0; WIDTH];
        assert_eq!(strongest_rival_margin(&scores, 1, 0, 1, 1), Ok(None));
        assert_eq!(
            strongest_rival_margin(&scores, 1, 64, 1, 1),
            Err(MarginError::InadmissibleTarget)
        );
        assert_eq!(
            strongest_rival_margin(&scores, 1, 1, 1, 1),
            Err(MarginError::InadmissibleTarget)
        );
        assert_eq!(
            strongest_rival_margin(&scores, 1, 0, 0, 1),
            Err(MarginError::InvalidConfiguration)
        );
    }

    #[test]
    fn loss_rounding_can_hide_a_live_gradient() {
        let rank = strongest_rival_margin(&[0; WIDTH], 3, 0, 1, 1)
            .unwrap()
            .unwrap();
        assert_eq!(rank.loss_q15, 0);
        assert_eq!(rank.derivative_q15, 1);
    }
}
