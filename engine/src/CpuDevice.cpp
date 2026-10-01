// The CPU device: the reference implementation of the per-device seam.
//
// Host memory is ordinary aligned memory, transfers are memcpy, and the kernel
// table is owned by the device and delegates to the CPU operator table. Only
// the C++17 standard library is used.

#include "engine/Device.hpp"

#include <cstddef>
#include <cstring>
#include <new>

#include "kernels/Kernels.hpp"

namespace engine {
namespace {

// The planner lays tensors out on 64-byte boundaries, so the arena allocator
// hands back storage aligned to at least that.
constexpr std::size_t kArenaAlignment = 64;

class CpuAllocator final : public Allocator {
public:
    // Returns 64-byte-aligned storage of at least `bytes` bytes, or null when
    // `bytes` is not positive or the allocation fails. The memory is not zeroed.
    void* allocate(int64_t bytes) override {
        if (bytes <= 0) {
            return nullptr;
        }
        try {
            return ::operator new(static_cast<std::size_t>(bytes), std::align_val_t{kArenaAlignment});
        } catch (const std::bad_alloc&) {
            return nullptr;
        }
    }

    // Frees a buffer from allocate(); a null pointer is a safe no-op.
    void release(void* buffer) override {
        if (buffer != nullptr) {
            ::operator delete(buffer, std::align_val_t{kArenaAlignment});
        }
    }
};

class CpuTransfer final : public Transfer {
public:
    // Copies `bytes` bytes from the host into device memory.
    void host_to_device(void* dst, const void* src, int64_t bytes) override {
        if (bytes == 0) {
            return;
        }
        std::memcpy(dst, src, static_cast<std::size_t>(bytes));
    }

    // Copies `bytes` bytes from device memory back to the host.
    void device_to_host(void* dst, const void* src, int64_t bytes) override {
        if (bytes == 0) {
            return;
        }
        std::memcpy(dst, src, static_cast<std::size_t>(bytes));
    }
};

// The operator lookup for the CPU device. It forwards to the kernel table
// declared in kernels/Kernels.hpp, so an unknown op type resolves to null.
class CpuKernelTable final : public KernelTable {
public:
    KernelFn find(const std::string& op_type) const override {
        return kernels::FindCpuKernel(op_type);
    }
};

class CpuDeviceImpl final : public Device {
public:
    const char* name() const override { return "cpu"; }
    Allocator& allocator() override { return allocator_; }
    Transfer& transfer() override { return transfer_; }
    const KernelTable& kernels() const override { return kernels_; }

private:
    CpuAllocator allocator_;
    CpuTransfer transfer_;
    CpuKernelTable kernels_;
};

} // namespace

const Device& CpuDevice() {
    // One process-lifetime instance; the reference is stable across calls. It is
    // deliberately not const itself so the mutable allocator/transfer accessors
    // remain callable through the returned const reference.
    static CpuDeviceImpl device;
    return device;
}

} // namespace engine
