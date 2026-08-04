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
    // `model_path` is a serialized DALI pipeline. `input_sizes[0]` declares the
    // pipeline's output shape (for example {3, 640, 640}); DALI cannot report it
    // before a run, and callers need it to advertise model metadata at load.
    DALIInfer(const std::string& model_path, bool use_gpu = true, size_t batch_size = 1,
              const std::vector<std::vector<int64_t>>& input_sizes = {});
    ~DALIInfer() override;

    DALIInfer(const DALIInfer&) = delete;
    DALIInfer& operator=(const DALIInfer&) = delete;

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override;

    std::vector<RawOutputTensor> get_infer_results_raw(const std::vector<std::vector<uint8_t>>& input_tensors) override;

    InferenceMetadata get_inference_metadata() override;

    // Name of the external source the pipeline must declare, matching the
    // generator in export/dali/.
    static constexpr const char* kEncodedInputName = "IMAGE";
    // Output 0: the model input tensor. Output 1: the source image shape (HWC),
    // which postprocessing needs to map results back onto the original frame.
    static constexpr const char* kPreprocessedOutputName = "preprocessed";
    static constexpr const char* kImageShapeOutputName = "IMAGE_SHAPE";

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
