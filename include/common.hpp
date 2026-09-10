#pragma once
// Umbrella header for the shared std/glog surface. It deliberately declares
// nothing itself. OpenCV used to be included here, which forced every backend
// and every consumer to parse it; OpenCV now lives only in the opencv-dnn
// backend. Removing it also removed a transitive supplier of std headers
// (<array> <chrono> <cstddef> <cstdint> <cstring> <functional> <limits> <map>
// <memory> <stdexcept> <string> <tuple> <utility> <vector>), so those are now
// listed explicitly. Without them the tree still builds on libstdc++, where
// <iostream> happens to drag most of them in, but not on MSVC.
//
// OpenCV also supplied the glibc C headers -- <cassert>, <cstdlib> and the
// rest. Those are deliberately not listed here: the four files that had been
// relying on them now include what they use, which is where it belongs.
#include <algorithm>
#include <any>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits> // for std::remove_pointer
#include <utility>
#include <vector>
