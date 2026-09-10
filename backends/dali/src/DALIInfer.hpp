#pragma once

#include "InferenceInterface.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// GPU preprocessing backend: hosts a serialized NVIDIA DALI pipeline in-process
// through the DALI C API.
//
// This is not an inference engine, and that is the point. It occupies the same
// InferenceInterface slot as one so that a serving pipeline can chain
// "decode + resize + normalize on the GPU" ahead of a TensorRT model without
// the chaining layer needing a second concept. Feed it encoded image bytes; it
// returns the preprocessed tensor the model expects.
//
// Pipelines are authored offline and shipped as serialized .dali artifacts (see
// export/dali/). Nothing here runs Python: the pipeline is deserialized and
// executed entirely in C++.
//
// Two DALI libraries must be linked, libdali.so and libdali_operators.so.
// libdali.so does not pull in the operator library -- DALI's Python bindings
// dlopen it -- so a C++ host must link it and call daliInitOperators(),
// otherwise every run fails with `No schema found for operator
// "decoders__Image"`.
class DALIInfer : public InferenceInterface {
  public:
    // `model_path` is a serialized DALI pipeline, optionally followed by
    // `|plugin=<path.so>` for pipelines built on custom DALI operators (GPU
    // postprocessing plugins are the case that needs it).
    //
    // `input_sizes` declares the pipeline's input shapes. DALI cannot report
    // output shapes before a run; an optional `out=` model-path suffix supplies
    // the first output shape for metadata at load.
    DALIInfer(const std::string& model_path, bool use_gpu = true, size_t batch_size = 1,
              const std::vector<std::vector<int64_t>>& input_sizes = {});
    ~DALIInfer() override;

    DALIInfer(const DALIInfer&) = delete;
    DALIInfer& operator=(const DALIInfer&) = delete;

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override;

    std::vector<RawOutputTensor> get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) override;

    InferenceMetadata get_inference_metadata() override;

    // Conventional name for an encoded-image external source, matching the
    // generator in export/dali/. Pipelines may declare any inputs they like:
    // the backend discovers them and feeds them in declaration order.
    static constexpr const char* kEncodedInputName = "IMAGE";
    // Outputs are metadata-addressed positionally ("output0", ...). The
    // serialized pipeline format carries no output names; deployments that
    // need semantic names must pass `|outnames=A,B,...` on the model path.

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    // Declared shapes for the pipeline inputs, supplied by the caller because
    // DALI reports neither before a run.
    std::vector<std::vector<int64_t>> input_sizes_;
    std::vector<std::vector<int64_t>> input_shapes_;
};
