#pragma once
// The per-device seam: memory, transfers, and the kernel table.
//
// The executor talks to a device only through this interface, so a backend can
// supply its own allocator, host<->device transfers, and operator kernels
// without the rest of the engine knowing which device is in use. A TensorView
// is a borrowed, non-owning handle to one tensor's storage and shape; the
// kernel calling convention is fixed here and implemented by the kernel tables.
//
// Must not include or reference the backend abstraction layer.

#include <cstdint>
#include <string>
#include <vector>

#include "engine/Graph.hpp"

namespace engine {

// A non-owning view of one tensor: where its data lives, its element type, and
// its concrete dimensions. `data` may be null for a zero-element tensor.
struct TensorView {
    void* data = nullptr;
    DataType dtype = DataType::Unknown;
    std::vector<int64_t> dims;
};

// A kernel implementation for one operator type. It reads the node's attributes
// and the input views and writes the output views; it owns no storage.
using KernelFn = void (*)(const Node& node, const std::vector<TensorView>& inputs,
                          const std::vector<TensorView>& outputs);

// Device memory: reserves and returns aligned buffers. The planner's offsets
// assume the allocator's alignment, so implementations must honour it.
class Allocator {
public:
    virtual ~Allocator() = default;
    virtual void* allocate(int64_t bytes) = 0;
    virtual void release(void* buffer) = 0;
};

// Copies tensor bytes between host and device memory. On a host device both
// directions are plain memory copies.
class Transfer {
public:
    virtual ~Transfer() = default;
    virtual void host_to_device(void* dst, const void* src, int64_t bytes) = 0;
    virtual void device_to_host(void* dst, const void* src, int64_t bytes) = 0;
};

// The device's operator lookup: maps an op type to its kernel, or null when the
// device has no implementation for that op.
class KernelTable {
public:
    virtual ~KernelTable() = default;
    virtual KernelFn find(const std::string& op_type) const = 0;
};

// One execution device. The table is owned by the device; there is no global
// mutable registry.
class Device {
public:
    virtual ~Device() = default;
    virtual const char* name() const = 0;
    virtual Allocator& allocator() = 0;
    virtual Transfer& transfer() = 0;
    virtual const KernelTable& kernels() const = 0;
};

// The process-lifetime CPU device, the reference implementation of the seam.
const Device& CpuDevice();

} // namespace engine
