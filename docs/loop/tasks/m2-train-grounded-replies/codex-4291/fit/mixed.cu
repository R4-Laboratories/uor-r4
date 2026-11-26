#include <filesystem>
#include <fstream>

// Reuse the released enrollment, measurement, update and eight-sweep driver.
#include "margin-capture.cuh"

int main(int argc, char **argv) {
    try {
        if (argc != 6) {
            throw std::runtime_error(
                "usage: transport-mixed-sm89 TEXT LABELS PARENT_STATE DISTANCE_TXT NEW_OUTPUT_DIR");
        }
        using namespace margin_fit;
        const auto packet = load_packet(argv[1]);
        const auto labels = load_labels(argv[2], packet, Permutation::unchanged);
        const auto parent = load_state(argv[3]);
        const auto distance = load_distance(argv[4], packet.rows.size());
        const MixedCapture capture{argv[5], distance};
        const auto trace = capture.path("trace.jsonl");
        run(Objective::contrastive, packet, labels, parent, trace.c_str(), nullptr, nullptr, &capture);
    } catch (const std::exception &error) {
        fprintf(stderr, "transport-mixed-sm89: %s\n", error.what());
        return 1;
    }
    return 0;
}
