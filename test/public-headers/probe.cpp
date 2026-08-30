// Includes only. This translation unit exists so a compiler will object if any
// public header stops being self-sufficient: either because it grew an
// <opencv2/...> include again, or because it leans on OpenCV to supply a std
// header transitively (see the note in include/common.hpp).
#include "InferenceBackendSetup.hpp"
#include "InferenceInterface.hpp"
#include "common.hpp"

int neuriplo_public_headers_probe() { return 0; }
