const S4_SOURCE_MAGIC: &[u8; 8] = b"UORS4G01";
const PICK_CLASSES: usize = 5;
const S4_ROLES: usize = 3;
const ROLE_ROWS: usize = PICK_CLASSES * S4_ROLES;
const MAX_WEIGHT: i8 = 64;
const SOURCE_BODY_BYTES: usize = 8 + 32 + 32 + S4_SOURCE_WEIGHT_COUNT;
pub const S4_SOURCE_WEIGHT_COUNT: usize = ROLE_ROWS + PICK_CLASSES;
pub const S4_SOURCE_GATE_BYTES: usize = SOURCE_BODY_BYTES + 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum S4SourceMode {
    Active,
    Disabled,
    WrongAddress,
    RoleBlind,
    ShuffledRole,
}

impl S4SourceMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Disabled => "disabled",
            Self::WrongAddress => "wrong-address",
            Self::RoleBlind => "role-blind",
            Self::ShuffledRole => "shuffled-role",
        }
    }
}

/// Signed Q8 fractions in five Pick classes crossed with S4 roles 1 through 3,
/// followed by five independently fitted role-blind rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct S4SourceGate {
    p4_state_sha256: [u8; 32],
    train_sha256: [u8; 32],
    weights: [i8; S4_SOURCE_WEIGHT_COUNT],
}

impl S4SourceGate {
    pub fn new(
        p4_state_sha256: [u8; 32],
        train_sha256: [u8; 32],
        weights: [i8; S4_SOURCE_WEIGHT_COUNT],
    ) -> Result<Self> {
        if weights
            .iter()
            .any(|w| !(-MAX_WEIGHT..=MAX_WEIGHT).contains(w))
        {
            return Err(invalid("S4 source gate weight exceeds signed Q8 cap"));
        }
        Ok(Self {
            p4_state_sha256,
            train_sha256,
            weights,
        })
    }

    pub fn p4_state_sha256(&self) -> [u8; 32] {
        self.p4_state_sha256
    }

    pub fn train_sha256(&self) -> [u8; 32] {
        self.train_sha256
    }

    /// Byte-only serialization, including a checksum over the identity and learned state.
    pub fn encode_bytes(&self) -> [u8; S4_SOURCE_GATE_BYTES] {
        let mut bytes = [0; S4_SOURCE_GATE_BYTES];
        bytes[..8].copy_from_slice(S4_SOURCE_MAGIC);
        bytes[8..40].copy_from_slice(&self.p4_state_sha256);
        bytes[40..72].copy_from_slice(&self.train_sha256);
        for (output, &weight) in bytes[72..SOURCE_BODY_BYTES].iter_mut().zip(&self.weights) {
            *output = weight.to_le_bytes()[0];
        }
        let digest = Sha256::digest(&bytes[..SOURCE_BODY_BYTES]);
        bytes[SOURCE_BODY_BYTES..].copy_from_slice(&digest);
        bytes
    }

    pub fn from_bytes(bytes: &[u8], expected_p4_state_sha256: [u8; 32]) -> Result<Self> {
        if bytes.len() != S4_SOURCE_GATE_BYTES || &bytes[..8] != S4_SOURCE_MAGIC {
            return Err(invalid("invalid UORS4G01 S4 source gate format"));
        }
        if bytes[8..40] != expected_p4_state_sha256 {
            return Err(invalid("S4 source gate P4 state SHA-256 mismatch"));
        }
        if Sha256::digest(&bytes[..SOURCE_BODY_BYTES])[..] != bytes[SOURCE_BODY_BYTES..] {
            return Err(invalid("S4 source gate checksum mismatch"));
        }
        let mut train_sha256 = [0; 32];
        train_sha256.copy_from_slice(&bytes[40..72]);
        let mut weights = [0; S4_SOURCE_WEIGHT_COUNT];
        for (weight, &input) in weights.iter_mut().zip(&bytes[72..SOURCE_BODY_BYTES]) {
            *weight = i8::from_le_bytes([input]);
        }
        Self::new(expected_p4_state_sha256, train_sha256, weights)
    }

    pub(super) fn apply(
        &self,
        mode: S4SourceMode,
        pick_class: u8,
        role_scores: [i64; 3],
        pick_source_pos: Option<usize>,
        history: &[u16],
        scores: &mut [u64],
    ) -> S4SourceTrace {
        let resolved_role = unique_role(role_scores);
        let mut trace = S4SourceTrace {
            resolved_role,
            ..S4SourceTrace::default()
        };
        if mode == S4SourceMode::Disabled || usize::from(pick_class) >= PICK_CLASSES {
            return trace;
        }
        let Some(source_pos) = pick_source_pos
            .filter(|&pos| pos >= 2 && history.get(pos).is_some_and(|&token| legal_source(token)))
        else {
            return trace;
        };
        let Some(role) = resolved_role else {
            return trace;
        };
        let (effective_role, gate_index) = if mode == S4SourceMode::RoleBlind {
            (None, ROLE_ROWS + usize::from(pick_class))
        } else {
            let role = if mode == S4SourceMode::ShuffledRole {
                role % 3 + 1
            } else {
                role
            };
            (
                Some(role),
                usize::from(pick_class) * S4_ROLES + usize::from(role - 1),
            )
        };
        let Some(&weight) = self.weights.get(gate_index) else {
            return trace;
        };
        trace.effective_role = effective_role;
        trace.gate_index = Some(gate_index as u8);
        trace.weight_i8 = weight;
        let source_pos = if mode == S4SourceMode::WrongAddress {
            let Some(shifted) = shifted_source(history, source_pos) else {
                return trace;
            };
            shifted
        } else {
            source_pos
        };
        let source = history[source_pos];
        let mut rival = 1;
        for candidate in 2..scores.len() {
            if candidate != usize::from(source) && scores[candidate] > scores[rival] {
                rival = candidate;
            }
        }
        let source = usize::from(source);
        trace.source_pos = Some(source_pos as u32);
        trace.source_token = Some(source as u16);
        trace.rival_token = Some(rival as u16);
        trace.source_mass_before_q48 = Some(scores[source]);
        trace.rival_mass_before_q48 = Some(scores[rival]);

        let (departing, receiving) = if weight < 0 {
            (source, rival)
        } else {
            (rival, source)
        };
        let transfer = (((scores[departing] - 1) * u64::from(weight.unsigned_abs())) >> 8)
            .min(P4_Q48_ONE - scores[receiving]);
        scores[departing] -= transfer;
        scores[receiving] += transfer;
        trace.transfer_q48 = transfer;
        trace
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct S4SourceTrace {
    pub resolved_role: Option<u8>,
    pub effective_role: Option<u8>,
    pub gate_index: Option<u8>,
    /// Zero-based position in the exact causal window whose digest the step reports.
    pub source_pos: Option<u32>,
    pub source_token: Option<u16>,
    pub rival_token: Option<u16>,
    pub weight_i8: i8,
    pub transfer_q48: u64,
    pub source_mass_before_q48: Option<u64>,
    pub rival_mass_before_q48: Option<u64>,
}

fn legal_source(token: u16) -> bool {
    (2..P4_VOCAB).contains(&usize::from(token))
}

fn unique_role(scores: [i64; S4_ROLES]) -> Option<u8> {
    scores.iter().enumerate().find_map(|(i, &score)| {
        (score > 0
            && scores
                .iter()
                .enumerate()
                .all(|(j, &other)| i == j || score > other))
        .then_some(i as u8 + 1)
    })
}

fn shifted_source(history: &[u16], source_pos: usize) -> Option<usize> {
    let original_token = history[source_pos];
    ((source_pos + 1)..history.len())
        .chain(2..source_pos)
        .find(|&pos| {
            let token = history[pos];
            token != original_token && legal_source(token)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_wire_checks_seal_dimensions_and_weight_cap() {
        let mut weights = [0; S4_SOURCE_WEIGHT_COUNT];
        weights[19] = -64;
        let gate = S4SourceGate::new([11; 32], [22; 32], weights).unwrap();
        let mut bytes = gate.encode_bytes();
        assert_eq!(S4SourceGate::from_bytes(&bytes, [11; 32]).unwrap(), gate);
        assert!(S4SourceGate::from_bytes(&bytes, [12; 32]).is_err());
        assert!(S4SourceGate::from_bytes(&bytes[..bytes.len() - 1], [11; 32]).is_err());
        bytes[72] = 65;
        assert!(S4SourceGate::from_bytes(&bytes, [11; 32]).is_err());
        let digest = Sha256::digest(&bytes[..SOURCE_BODY_BYTES]);
        bytes[SOURCE_BODY_BYTES..].copy_from_slice(&digest);
        assert!(S4SourceGate::from_bytes(&bytes, [11; 32]).is_err());
    }

    #[test]
    fn source_modes_preserve_floor_mass_and_route_eligibility() {
        let mut weights = [0; S4_SOURCE_WEIGHT_COUNT];
        weights[4] = 64;
        weights[5] = -64;
        weights[16] = 32;
        let gate = S4SourceGate::new([11; 32], [22; 32], weights).unwrap();
        let history = [0, 12, 2, 3, 3];
        let base = [0, 2561, 2001, 2561, 1];
        let role_scores = [10, 100, 0];
        let run = |mode, role_scores, source_pos| {
            let mut scores = base;
            let trace = gate.apply(mode, 1, role_scores, source_pos, &history, &mut scores);
            assert_eq!(scores.iter().sum::<u64>(), base.iter().sum::<u64>());
            assert!(scores[1..].iter().all(|&mass| mass >= 1));
            (trace, scores)
        };
        let (active, scores) = run(S4SourceMode::Active, role_scores, Some(2));
        assert_eq!(active.source_pos, Some(2));
        assert_eq!(active.source_token, Some(2));
        assert_eq!(active.rival_token, Some(1));
        assert_eq!(active.effective_role, Some(2));
        assert_eq!(scores, [0, 1921, 2641, 2561, 1]);
        let (wrong, scores) = run(S4SourceMode::WrongAddress, role_scores, Some(4));
        assert_eq!(wrong.source_pos, Some(2));
        assert_eq!(wrong.gate_index, active.gate_index);
        assert_eq!(scores, [0, 1921, 2641, 2561, 1]);
        let (shuffle, scores) = run(S4SourceMode::ShuffledRole, role_scores, Some(2));
        assert_eq!(shuffle.effective_role, Some(3));
        assert_eq!(shuffle.weight_i8, -64);
        assert_eq!(scores, [0, 3061, 1501, 2561, 1]);
        let (blind, scores) = run(S4SourceMode::RoleBlind, role_scores, Some(2));
        assert_eq!(blind.effective_role, None);
        assert_eq!(blind.gate_index, Some(16));
        assert_eq!(scores, [0, 2241, 2321, 2561, 1]);
        for mode in [S4SourceMode::Active, S4SourceMode::RoleBlind] {
            for (votes, address) in [([100, 100, 0], Some(2)), (role_scores, None)] {
                let (trace, scores) = run(mode, votes, address);
                assert_eq!(trace.source_pos, None);
                assert_eq!(scores, base);
            }
        }
        let (_, disabled_scores) = run(S4SourceMode::Disabled, role_scores, Some(2));
        assert_eq!(disabled_scores, base);
    }
}
