// Identical state and release_score are assembled from #4325 before this optimizer.
constexpr uint32_t GROUP_WORDS = 5;
constexpr uint32_t PAIR_EPOCHS = 4;
constexpr uint32_t PAIR_AUDIT_HEADER = 11;
constexpr uint32_t PAIR_AUDIT_WORDS = PAIR_AUDIT_HEADER + COORDINATE_CAP;
constexpr uint32_t NO_PEER = UINT32_MAX;

struct Support {
    InstructionBuckets bits = {};
    uint16_t addresses[WORDS];
    uint32_t count = 0;

    __device__ void insert(uint32_t bucket) {
        bits.insert(bucket);
        addresses[count++] = static_cast<uint16_t>(bucket);
    }

    __device__ uint32_t shared(const Support* peer) const {
        if (peer == nullptr) return 0;
        uint32_t count = 0;
        for (uint32_t word = 0; word < WORDS / 32; ++word) {
            count += __popc(bits.seen[word] & peer->bits.seen[word]);
        }
        return count;
    }
};

// Filtering keeps first-visit order. The stride and phase match the old row-wise update.
__device__ static uint32_t fit_member(
    int16_t* residual, const Support& support, const Support* peer, bool distinguish,
    int32_t score, int32_t sign, uint32_t epoch, uint32_t visits, int32_t* audit) {
    const uint32_t shared = support.shared(peer);
    const uint32_t options = distinguish ? support.count - shared : support.count;
    audit[1] = score;
    audit[2] = score;
    audit[3] = score;
    audit[4] = visits;
    audit[5] = support.count;
    audit[6] = shared;
    audit[7] = options;
    if (support.count == 0) {
        audit[0] = 1;
        return 0;
    }
    if (options == 0 || sign * score >= MARGIN) return 0;

    const uint32_t cap = options < COORDINATE_CAP ? options : COORDINATE_CAP;
    uint32_t phase = 17 * epoch;
    while (phase >= options) phase -= options;
    // Select the same rotated ranks as #4325. Visit them in sorted order because
    // their writes are distinct and position within a single member does not affect state.
    uint32_t ranks[COORDINATE_CAP];
    for (uint32_t visit = 0; visit < cap; ++visit) {
        const uint32_t stride = options < COORDINATE_CAP ? visit : (visit * options) >> 4;
        const uint32_t rank = phase + stride;
        ranks[visit] = rank < options ? rank : rank - options;
    }
    uint32_t rank = 0;
    for (uint32_t index = 0; index < support.count; ++index) {
        const uint32_t bucket = support.addresses[index];
        const bool common = peer != nullptr && includes(peer->bits, bucket);
        if (distinguish && common) continue;
        for (uint32_t visit = 0; visit < cap; ++visit) {
            if (ranks[visit] != rank) continue;
            uint32_t error = 0;
            add_word(residual[bucket], sign, error);
            if (error != 0) {
                audit[0] = 2;
                return 0;
            }
            audit[PAIR_AUDIT_HEADER + visit] = bucket + (common ? WORDS : 0);
            ++audit[8];
            audit[9] += common;
            audit[10] += !common && peer != nullptr;
            audit[2] += sign;
            audit[3] += sign;
            break;
        }
        ++rank;
    }
    return audit[9];
}

// Exactly one worker walks the sealed groups. Neither residual writes nor membership
// may depend on launched block order, floating point, or a second set of weights.
extern "C" __global__ void paired_fit(
    const uint32_t* roots, const uint32_t* instructions, uint32_t rows,
    const uint32_t* groups, uint32_t count, uint32_t paired, int16_t* state, int32_t* audit) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (uint32_t group = 0; group < count; ++group) {
        const uint32_t* record = groups + (uint64_t)group * GROUP_WORDS;
        int32_t* first = audit + (uint64_t)group * 2 * PAIR_AUDIT_WORDS;
        int32_t* second = first + PAIR_AUDIT_WORDS;
        const uint32_t stage = record[2];
        const bool has_peer = record[1] != NO_PEER;
        if (record[0] >= rows || (has_peer && (record[1] >= rows || record[1] == record[0]))
            || stage >= 2 || record[3] > 1 || record[4] >= PAIR_EPOCHS || paired > 1) {
            first[0] = 3;
            return;
        }
        const uint32_t* a = instructions + (uint64_t)record[0] * STRIDE;
        const uint32_t* b = has_peer ? instructions + (uint64_t)record[1] * STRIDE : nullptr;
        const int32_t sign = record[3] == 1 ? 1 : -1;
        Support a_support, b_support;
        const int32_t a_score = release_score(
            roots, a, state, stage, [&](uint32_t bucket) { a_support.insert(bucket); });
        int32_t b_score = 0;
        if (has_peer) {
            b_score = release_score(
                roots, b, state, stage, [&](uint32_t bucket) { b_support.insert(bucket); });
        }
        const bool distinguish = paired == 1 && has_peer;
        const uint32_t first_shared = fit_member(
            state + RELEASE_OFFSET, a_support, has_peer ? &b_support : nullptr, distinguish,
            a_score, sign, record[4], link_visits(a[0]), first);
        if (first[0] != 0) return;
        if (has_peer) {
            const uint32_t second_shared = fit_member(
                state + RELEASE_OFFSET, b_support, &a_support, distinguish,
                b_score + sign * static_cast<int32_t>(first_shared), -sign, record[4],
                link_visits(b[0]), second);
            if (second[0] != 0) return;
            first[3] -= sign * static_cast<int32_t>(second_shared);
        }
    }
}
