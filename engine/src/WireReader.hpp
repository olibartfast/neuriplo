#pragma once
// Minimal hand-written protobuf wire-format reader ([D-7], private to
// engine/src/). Generic and ONNX-agnostic: no field numbers, no protobuf
// message names — the caller supplies the tag constants it wants.
//
// Operates on a plain byte buffer and throws engine::ModelLoadException
// (declared in engine/Graph.hpp) on truncation or malformed encoding.

#include <cstdint>
#include <vector>

namespace engine {

// Protobuf wire types (format constants, not message-specific tags).
enum class WireType : uint8_t {
    Varint = 0,
    Fixed64 = 1,
    LengthDelimited = 2,
    Fixed32 = 5,
};

class WireReader {
public:
    // Non-owning view over the raw bytes to decode.
    WireReader(const uint8_t* data, size_t size);

    bool at_end() const { return cur_ == end_; }

    // A single (field_number, wire_type) key from the stream.
    struct Field {
        int field_number;
        WireType wire_type;
    };

    // Reads the next tag. Throws ModelLoadException on truncation and on
    // field numbers/wire types that cannot occur on the wire.
    Field read_tag();

    // -- Payload readers, matching the wire type of the current field. ----

    uint64_t read_varint();      // WireType::Varint
    uint32_t read_fixed32();     // WireType::Fixed32
    uint64_t read_fixed64();     // WireType::Fixed64

    // Length-delimited payload: bytes, strings, embedded messages.
    // Returns the sub-buffer (into the same backing storage) and advances.
    std::pair<const uint8_t*, size_t> read_length_delimited();

    // Convenience: length-delimited payload as bytes.
    // (Reserved here so callers never roll their own copies of it.)
    std::vector<uint8_t> read_bytes();

    // Skips forward past one field of the given wire type, whose tag has
    // already been consumed. Throws ModelLoadException if the field is
    // malformed or truncated. Supports varint, fixed32, fixed64,
    // length-delimited encodings.
    void skip_field(WireType wire_type);

    // Skipping one length-delimited field that is itself a sub-message:
    // parse the sub-message on a sub-reader and skip its fields, so that
    // unknown containers with unknown inner fields are fully skipped.
    void skip_length_delimited_message(const uint8_t* data, size_t size);

private:
    const uint8_t* cur_; // read cursor
    const uint8_t* end_; // one past the last valid byte
};

} // namespace engine
