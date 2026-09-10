#!/usr/bin/env bash
# Serialize DALI pipelines inside the NVIDIA container.
#
# The container is the toolchain: it carries a Python DALI whose version matches
# the C++ libdali the backend links, which is what a serialized pipeline (and
# any operator plugin) is tied to. Generating from an ad-hoc virtualenv invites a
# silent version skew -- a plugin built against DALI 1.51.2 refuses to load into
# 1.50.0, and the failure surfaces only when the pipeline is deserialized.
#
# No Python is needed on the host, and none runs at inference time: the C++
# backend deserializes the artifact through the DALI C API.
set -euo pipefail

TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.12-py3}"
SIZE="${SIZE:-640}"
OUTPUT_DIR="${OUTPUT_DIR:-data/dali}"
PLUGIN=""

usage() {
    cat <<'USAGE'
Usage: export/dali/generate_pipelines.sh [options]

  --size N            model input size (default 640)
  --output-dir DIR    where to write .dali artifacts (default data/dali)
  --plugin PATH       DALI operator plugin (.so) for postprocessing pipelines,
                      relative to the repository root
  --triton-image REF  container image (default nvcr.io/nvidia/tritonserver:25.12-py3)

Writes <output-dir>/yolo_preprocess_<size>.dali, plus a postprocess pipeline
when --plugin is given.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size) SIZE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --plugin) PLUGIN="$2"; shift 2 ;;
        --triton-image) TRITON_IMAGE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "${repo_root}/${OUTPUT_DIR}"

echo "Generating DALI pipelines with ${TRITON_IMAGE}"

docker run --gpus all --rm \
    -v "${repo_root}:/workspace" -w /workspace \
    "${TRITON_IMAGE}" \
    python3 export/dali/generate_yolo_pipeline.py \
        --size "${SIZE}" \
        --output "${OUTPUT_DIR}/yolo_preprocess_${SIZE}.dali"

if [[ -n "${PLUGIN}" ]]; then
    # A postprocessing pipeline is built on custom CUDA operators, so its plugin
    # must be loaded before serialization and again before deserialization at
    # serve time (model_path suffix "|plugin=<path>").
    docker run --gpus all --rm \
        -v "${repo_root}:/workspace" -w /workspace \
        "${TRITON_IMAGE}" \
        python3 export/dali/generate_yolo_postprocess_pipeline.py \
            --plugin "${PLUGIN}" \
            --output "${OUTPUT_DIR}/yolo_postprocess.dali"
fi

echo "Wrote pipelines to ${repo_root}/${OUTPUT_DIR}"
