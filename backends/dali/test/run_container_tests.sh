#!/usr/bin/env bash
# Compile and exercise the real DALI backend in the same NVIDIA image used to
# serialize its fixtures. Host GoogleTest sources and logging headers are read-only.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
image="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.12-py3}"
output_dir="${DALI_TEST_OUTPUT_DIR:-${repo_root}/build-release-dali-container}"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"

docker run --rm -i --gpus all --pull=never --network=none \
    --user "$(id -u):$(id -g)" --entrypoint bash \
    -v "${repo_root}:/src:ro" -v "${output_dir}:/results" \
    -v "${GTEST_SOURCE_DIR:-/usr/src/googletest}:/opt/googletest:ro" \
    -v "${GLOG_INCLUDE_DIR:-/usr/include/glog}:/opt/host-include/glog:ro" \
    -v "${GFLAGS_INCLUDE_DIR:-/usr/include/gflags}:/opt/host-include/gflags:ro" \
    -w /results "$image" -s <<'CONTAINER'
set -euo pipefail
export HOME=/tmp
DALI_ROOT="$(python3 -c 'import pathlib,nvidia.dali; print(pathlib.Path(nvidia.dali.__file__).parent)')"
python3 -c 'import nvidia.dali; print("DALI version:", nvidia.dali.__version__)'
export LD_LIBRARY_PATH="${DALI_ROOT}:${DALI_ROOT}/.libs:${LD_LIBRARY_PATH:-}"
python3 /src/backends/dali/test/generate_test_pipelines.py --output-dir /results/fixtures
python3 /src/export/dali/generate_yolo_pipeline.py --size 640 --output /results/yolo_preprocess.dali
printf '%s\n' /results/yolo_preprocess.dali > dali_pipeline_path.txt
printf '%s\n' /src/data/neuriplo.png > dali_image_path.txt
g++ -std=c++17 -pthread -O1 \
    -I/src/backends/dali/src -I/src/backends/src -I/src/include \
    -I/opt/host-include -I/opt/googletest/googletest/include \
    -I/opt/googletest/googletest -I"${DALI_ROOT}/include" \
    -I/usr/local/cuda/include \
    /src/backends/dali/src/DALIInfer.cpp \
    /src/backends/src/InferenceInterface.cpp \
    /src/backends/src/InferenceMetadata.cpp \
    /src/backends/dali/test/DALIInferTest.cpp \
    /opt/googletest/googletest/src/gtest-all.cc \
    /opt/googletest/googletest/src/gtest_main.cc \
    -L"${DALI_ROOT}" -ldali -ldali_operators -o DALIInferTest
NEURIPLO_DALI_TEST_FIXTURES=/results/fixtures \
    ./DALIInferTest --gtest_output=xml:/results/results.xml
python3 - <<'PY'
import xml.etree.ElementTree as ET
root = ET.parse('/results/results.xml').getroot()
tests = list(root.iter('testcase'))
assert len(tests) >= 18, f'Incomplete suite: {len(tests)} tests'
for test in tests:
    assert test.get('status') == 'run', test.attrib
    assert test.find('skipped') is None, test.attrib
    assert test.find('failure') is None, test.attrib
    assert test.find('error') is None, test.attrib
print(f'Acceptance: {len(tests)} tests executed; zero skips, failures, or errors')
PY
CONTAINER
