// Included in margin_fit in place of the Wick admission function only.
void require_lease() {
    constexpr int inherited = 20;
    constexpr std::string_view expected = "/root/gpu0.lock";
    const auto *visible = getenv("CUDA_VISIBLE_DEVICES");
    char target[256];
    const auto size = readlink("/proc/self/fd/20", target, sizeof(target));
    if (visible == nullptr || std::string_view(visible) != "0" ||
        size != ssize_t(expected.size()) || std::string_view(target, size) != expected ||
        fcntl(inherited, F_GETFD) < 0) {
        throw std::runtime_error("run under the inherited uor-pod GPU 0 lease on descriptor 20");
    }
    // A second open must conflict before re-checking the inherited descriptor.
    // This rejects a descriptor merely opened on the expected lock file.
    const int independent = open(expected.data(), O_RDONLY | O_CLOEXEC);
    if (independent < 0) throw std::runtime_error("cannot inspect the inherited uor-pod GPU lock");
    const int probe = flock(independent, LOCK_EX | LOCK_NB);
    const int failure = errno;
    close(independent);
    if (probe == 0 || failure != EWOULDBLOCK || flock(inherited, LOCK_EX | LOCK_NB) != 0) {
        throw std::runtime_error("uor-pod GPU 0 descriptor does not hold the exclusive lease");
    }
    require_device();
}
