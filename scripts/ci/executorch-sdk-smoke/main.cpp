#include <cstdint>
#include <executorch/runtime/core/evalue.h>

int main() {
    const executorch::runtime::EValue value(int64_t{42});
    return value.toInt() == 42 ? 0 : 1;
}
