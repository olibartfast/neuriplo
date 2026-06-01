#pragma once

#include <cstddef>
#include <new>

// Minimal pure-virtual allocation interface. Backends that manage their own
// device/host memory can implement this; non-migrated backends can ignore it.
class IAllocator {
  public:
    virtual ~IAllocator() = default;
    virtual void* allocate(std::size_t bytes) = 0;
    virtual void deallocate(void* ptr) noexcept = 0;
    virtual const char* name() const noexcept = 0;
};

// Header-only default allocator backed by global operator new/delete.
class HostAllocator : public IAllocator {
  public:
    void* allocate(std::size_t bytes) override { return ::operator new(bytes); }
    void deallocate(void* ptr) noexcept override { ::operator delete(ptr); }
    const char* name() const noexcept override { return "HostAllocator"; }
};
