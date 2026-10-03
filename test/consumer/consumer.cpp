// Consumer proof, C++17 ([T-14]): the only neuriplo header is neuriplo.hpp.
// Usage: consumer_cpp <plugin_dir>. Same check and output as consumer.c.
#include <cstdio>
#include <exception>
#include <neuriplo/neuriplo.hpp>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <plugin_dir>\n", argv[0]);
        return 2;
    }
    try {
        neuriplo::EngineConfig config;
        config.backend_id = "FIXTURE_GOOD";
        config.model_path = "ok";
        config.plugin_dir = argv[1];
        neuriplo::Engine engine(config);

        const std::vector<float> in{1.0f, 2.0f, 3.0f, 4.0f};
        const neuriplo::Result result = engine.infer({in});
        if (result.size() != 1) {
            std::fprintf(stderr, "consumer_cpp: expected 1 output, got %zu\n", result.size());
            return 1;
        }
        const neuriplo::TensorView view = result.output(0);
        if (view.element_count() != in.size()) {
            std::fprintf(stderr, "consumer_cpp: unexpected element_count %zu\n", view.element_count());
            return 1;
        }
        const float* out = view.data_as<float>();
        for (size_t i = 0; i < in.size(); ++i) {
            if (out[i] != 2.0f * in[i]) {
                std::fprintf(stderr, "consumer_cpp: element %zu is %g, expected %g\n", i, static_cast<double>(out[i]),
                             static_cast<double>(2.0f * in[i]));
                return 1;
            }
        }
        std::printf("OK %s %g %g %g %g\n", engine.backend_id().c_str(), static_cast<double>(out[0]),
                    static_cast<double>(out[1]), static_cast<double>(out[2]), static_cast<double>(out[3]));
        return 0;
    } catch (const neuriplo::Error& error) {
        std::fprintf(stderr, "consumer_cpp: %s (%d): %s\n", neuriplo_status_string(error.status()),
                     static_cast<int>(error.status()), error.what());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "consumer_cpp: %s\n", error.what());
    }
    return 1;
}
