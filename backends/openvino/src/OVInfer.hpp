#pragma once
#include "InferenceInterface.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include <sstream>




class OVInfer : public InferenceInterface
{
public:
    OVInfer(const std::string& model_path, 
        bool use_gpu = false, 
        size_t batch_size = 1, 
        const std::vector<std::vector<int64_t>>& input_sizes = std::vector<std::vector<int64_t>>());

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override;

private:  
     // Helper function to print ov::Shape and ov::PartialShape
    template <typename ShapeType>
    std::string print_shape(const ShapeType& shape);
    
    ov::Core core_;
    ov::Tensor input_tensor_;
    ov::InferRequest infer_request_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;
};