#pragma once
#include "BackendDecorator.hpp"
#include "BackendState.hpp"
#include "InferenceInterface.hpp"

#include <cstddef>
#include <glog/logging.h>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Decorator that logs lifecycle and inference events for the wrapped backend.
//
// Wraps any InferenceInterface and emits glog messages around load() and
// get_infer_results(). The numeric inference output is forwarded unchanged;
// this decorator only augments observability and never alters results.
class LoggingBackend : public BackendDecorator {

  public:
    explicit LoggingBackend(std::unique_ptr<InferenceInterface> inner) : BackendDecorator(std::move(inner)) {}

    void load() override {
        LOG(INFO) << "LoggingBackend: loading model '" << get_model_path() << "'";
        BackendDecorator::load();
        LOG(INFO) << "LoggingBackend: load complete, state=" << to_string(state());
    }

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override {
        LOG(INFO) << "LoggingBackend: running inference on " << input_tensors.size() << " input tensor(s)";
        try {
            auto result = BackendDecorator::get_infer_results(input_tensors);
            const auto& shapes = std::get<1>(result);
            LOG(INFO) << "LoggingBackend: inference produced " << std::get<0>(result).size()
                      << " output tensor(s); shapes=" << format_shapes(shapes);
            return result;
        } catch (const std::exception& e) {
            LOG(ERROR) << "LoggingBackend: inference failed: " << e.what();
            throw;
        }
    }

  private:
    static std::string format_shapes(const std::vector<std::vector<int64_t>>& shapes) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shapes.size(); ++i) {
            if (i != 0) {
                oss << ", ";
            }
            oss << "(";
            for (size_t j = 0; j < shapes[i].size(); ++j) {
                if (j != 0) {
                    oss << "x";
                }
                oss << shapes[i][j];
            }
            oss << ")";
        }
        oss << "]";
        return oss.str();
    }
};
