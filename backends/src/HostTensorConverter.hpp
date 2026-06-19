#pragma once

#include "ITensorConverter.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

// Default host-side tensor converter.
//
// Reinterprets a contiguous host buffer as a sequence of typed elements and
// lifts them into neuriplo's TensorElement variant (and vice versa for raw
// input bytes). It is the converter counterpart to HostAllocator: a concrete,
// dependency-free default that Abstract Factory implementations can hand out so
// create_converter() never returns null.
//
// NOTE: this converter is NOT wired into any backend's inference path. Backends
// continue to do their own (unchanged) tensor conversion inline. This type only
// exists so the factory family is complete and independently testable.
//
// The TensorElement variant only carries {float, int32_t, int64_t, uint8_t}, so
// the two narrower request types are widened to the nearest representable
// alternative: Int8 -> int32_t and Bool -> uint8_t.
class HostTensorConverter : public ITensorConverter {
  public:
    std::vector<TensorElement> to_typed(const std::vector<uint8_t>& raw_bytes, TensorDataType type) const override {
        const std::size_t elem = element_size(type);
        const std::size_t count = elem == 0 ? 0 : raw_bytes.size() / elem;
        return decode(raw_bytes.data(), count, type);
    }

    std::vector<TensorElement> from_backend(const void* data, std::size_t num_elements,
                                            TensorDataType type) const override {
        return decode(data, num_elements, type);
    }

    const char* name() const noexcept override { return "HostTensorConverter"; }

  private:
    static std::size_t element_size(TensorDataType type) noexcept {
        switch (type) {
        case TensorDataType::Float32:
            return sizeof(float);
        case TensorDataType::Int32:
            return sizeof(int32_t);
        case TensorDataType::Int64:
            return sizeof(int64_t);
        case TensorDataType::UInt8:
            return sizeof(uint8_t);
        case TensorDataType::Int8:
            return sizeof(int8_t);
        case TensorDataType::Bool:
            return sizeof(uint8_t);
        }
        return 0;
    }

    // Copies `count` elements of `type` out of `data` into TensorElement values.
    // Uses memcpy to avoid unaligned-access UB on the raw byte buffer.
    static std::vector<TensorElement> decode(const void* data, std::size_t count, TensorDataType type) {
        std::vector<TensorElement> out;
        if (data == nullptr || count == 0) {
            return out;
        }
        out.reserve(count);
        const auto* bytes = static_cast<const uint8_t*>(data);

        for (std::size_t i = 0; i < count; ++i) {
            switch (type) {
            case TensorDataType::Float32: {
                float v;
                std::memcpy(&v, bytes + i * sizeof(float), sizeof(float));
                out.emplace_back(v);
                break;
            }
            case TensorDataType::Int32: {
                int32_t v;
                std::memcpy(&v, bytes + i * sizeof(int32_t), sizeof(int32_t));
                out.emplace_back(v);
                break;
            }
            case TensorDataType::Int64: {
                int64_t v;
                std::memcpy(&v, bytes + i * sizeof(int64_t), sizeof(int64_t));
                out.emplace_back(v);
                break;
            }
            case TensorDataType::UInt8: {
                out.emplace_back(static_cast<uint8_t>(bytes[i]));
                break;
            }
            case TensorDataType::Int8: {
                int8_t v;
                std::memcpy(&v, bytes + i * sizeof(int8_t), sizeof(int8_t));
                out.emplace_back(static_cast<int32_t>(v));
                break;
            }
            case TensorDataType::Bool: {
                out.emplace_back(static_cast<uint8_t>(bytes[i] != 0 ? 1 : 0));
                break;
            }
            }
        }
        return out;
    }
};
