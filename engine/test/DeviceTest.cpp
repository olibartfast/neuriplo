// CPU device-seam tests.
//
// One suite (EngineDevice.*) covering the fixed device interface contract: a
// stable process-lifetime CPU device, 64-byte-aligned arena storage, safe
// release (including null), memcpy transfers in both directions with a
// zero-byte no-op, and an empty kernel table that resolves unknown ops to null.

#include "engine/Device.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

// Two calls return the same object, and the device names itself "cpu".
TEST(EngineDevice, ReferenceIsStableAndNamedCpu) {
    const engine::Device& first = engine::CpuDevice();
    const engine::Device& second = engine::CpuDevice();
    EXPECT_EQ(&first, &second);
    EXPECT_STREQ(first.name(), "cpu");
}

// allocate(128) returns non-null storage aligned to a 64-byte boundary, and the
// block is writable then readable.
TEST(EngineDevice, AllocateIsAlignedAndUsable) {
    engine::Allocator& allocator = const_cast<engine::Device&>(engine::CpuDevice()).allocator();
    void* block = allocator.allocate(128);
    ASSERT_NE(block, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(block) % 64, 0U);

    std::memset(block, 0xAB, 128);
    const unsigned char* bytes = static_cast<const unsigned char*>(block);
    for (int i = 0; i < 128; ++i) {
        ASSERT_EQ(bytes[i], 0xAB) << "byte " << i;
    }

    allocator.release(block);
    allocator.release(nullptr); // safe no-op
}

// A non-positive request is refused with null rather than a partial buffer.
TEST(EngineDevice, NonPositiveAllocationReturnsNull) {
    engine::Allocator& allocator = const_cast<engine::Device&>(engine::CpuDevice()).allocator();
    EXPECT_EQ(allocator.allocate(0), nullptr);
    EXPECT_EQ(allocator.allocate(-1), nullptr);
}

// host_to_device then device_to_host round-trips an arbitrary byte pattern.
TEST(EngineDevice, TransferRoundTripsBytePattern) {
    engine::Device& device = const_cast<engine::Device&>(engine::CpuDevice());
    engine::Allocator& allocator = device.allocator();
    engine::Transfer& transfer = device.transfer();

    const std::vector<std::uint8_t> source = {0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF, 0x5A, 0xA5};
    void* device_block = allocator.allocate(static_cast<int64_t>(source.size()));
    ASSERT_NE(device_block, nullptr);

    transfer.host_to_device(device_block, source.data(), static_cast<int64_t>(source.size()));

    std::vector<std::uint8_t> round_trip(source.size(), 0);
    transfer.device_to_host(round_trip.data(), device_block, static_cast<int64_t>(source.size()));
    EXPECT_EQ(round_trip, source);

    allocator.release(device_block);
}

// A zero-byte copy is a no-op even when both pointers are null.
TEST(EngineDevice, ZeroByteTransferIsNoOp) {
    engine::Transfer& transfer = const_cast<engine::Device&>(engine::CpuDevice()).transfer();
    transfer.host_to_device(nullptr, nullptr, 0);
    transfer.device_to_host(nullptr, nullptr, 0);
}

// The kernel table owns no known op in this layer: an unknown op resolves null.
TEST(EngineDevice, UnknownOpResolvesNull) {
    const engine::KernelTable& kernels = engine::CpuDevice().kernels();
    EXPECT_EQ(kernels.find("NoSuchOp"), nullptr);
}

} // namespace
