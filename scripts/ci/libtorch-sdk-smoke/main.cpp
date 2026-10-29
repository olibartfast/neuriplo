#include <ATen/Version.h>
#include <iostream>
#include <string>
#include <torch/torch.h>
#include <torch/version.h>

int main(int argc, char** argv) {
    if (argc != 3 || std::string(TORCH_VERSION) != argv[1]) {
        return 2;
    }
    const auto config = at::show_config();
    if (config.find("TORCH_VERSION=" + std::string(argv[1]) + ",") == std::string::npos) {
        return 3;
    }
    if (std::string(argv[2]) == "cpu" &&
        (config.find("USE_CUDA=0,") == std::string::npos || config.find("USE_ROCM=OFF,") == std::string::npos)) {
        return 4;
    }
    const auto value = torch::ones({2, 3}).sum().item<float>();
    std::cout << "LibTorch " << TORCH_VERSION << ": CPU tensor sum = " << value << '\n';
    return value == 6.0f ? 0 : 1;
}
