#include "WireReader.hpp"

#include <string>

#include "engine/Graph.hpp" // ModelLoadException

namespace engine {

namespace {

// Throwable helper keeps the arithmetic parts succinct.
[[noreturn]] void malformed(const std::string& why) {
    throw ModelLoadException("protobuf wire: " + why);
}

} // namespace

WireReader::WireReader(const uint8_t* data, size_t size)
    : cur_(data), end_(data + size) {}

WireReader::Field WireReader::read_tag() {
    const uint64_t key = read_varint();
    const unsigned wt = static_cast<unsigned>(key & 0x7u);
    if (wt == 3 || wt == 4) { // start-group / end-group: not produced here
        malformed("group wire types are unsupported");
    }
    if (wt == 6 || wt == 7) {
        malformed("reserved wire type");
    }
    const int field_number = static_cast<int>(key >> 3);
    if (field_number <= 0) {
        malformed("zero field number");
    }
    return Field{field_number, static_cast<WireType>(wt)};
}

uint64_t WireReader::read_varint() {
    if (cur_ >= end_) {
        malformed("truncated varint (no bytes)");
    }
    uint64_t value = 0;
    int shift = 0;
    for (int i = 0; i < 10; ++i) { // at most 10 bytes for 64 bits
        if (cur_ == end_) {
            malformed("truncated varint");
        }
        const uint8_t byte = *cur_++;
        value |= static_cast<uint64_t>(byte & 0x7Fu) << shift;
        if ((byte & 0x80u) == 0) {
            return value;
        }
        shift += 7;
    }
    malformed("varint exceeds 10 bytes");
}

uint32_t WireReader::read_fixed32() {
    if (static_cast<size_t>(end_ - cur_) < 4) {
        malformed("truncated fixed32");
    }
    uint32_t value = 0;
    value |= static_cast<uint32_t>(cur_[0]);
    value |= static_cast<uint32_t>(cur_[1]) << 8;
    value |= static_cast<uint32_t>(cur_[2]) << 16;
    value |= static_cast<uint32_t>(cur_[3]) << 24;
    cur_ += 4;
    return value;
}

uint64_t WireReader::read_fixed64() {
    if (static_cast<size_t>(end_ - cur_) < 8) {
        malformed("truncated fixed64");
    }
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(cur_[i]) << (8 * i);
    }
    cur_ += 8;
    return value;
}

std::pair<const uint8_t*, size_t> WireReader::read_length_delimited() {
    const uint64_t length = read_varint();
    if (length > static_cast<uint64_t>(end_ - cur_)) {
        malformed("truncated length-delimited field");
    }
    const uint8_t* data = cur_;
    cur_ += static_cast<size_t>(length);
    return {data, static_cast<size_t>(length)};
}

std::vector<uint8_t> WireReader::read_bytes() {
    const auto span = read_length_delimited();
    return std::vector<uint8_t>(span.first, span.first + span.second);
}

void WireReader::skip_field(WireType wire_type) {
    switch (wire_type) {
    case WireType::Varint:
        read_varint();
        break;
    case WireType::Fixed64:
        read_fixed64();
        break;
    case WireType::Fixed32:
        read_fixed32();
        break;
    case WireType::LengthDelimited:
        read_length_delimited();
        break;
    }
}

void WireReader::skip_length_delimited_message(const uint8_t* data,
                                               size_t size) {
    WireReader sub(data, size);
    while (!sub.at_end()) {
        const Field field = sub.read_tag();
        sub.skip_field(field.wire_type);
    }
}

} // namespace engine
