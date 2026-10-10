#include "NertInfer.hpp"
#include "ORTInfer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace {

constexpr const char* kManifestEnv = "NEURIPLO_MODEL_ZOO_MANIFEST";
constexpr const char* kOnlyEnv = "NEURIPLO_MODEL_ZOO_ONLY";

// One manifest row: `name  path  dims[;dims...]  mode  atol  [input.bin|-]
// [int64=<v>]`, whitespace separated. `dims` is one `AxBxC` shape per graph input. `mode` is
// `parity` (run both backends, compare every output), `e2e` (end-to-end
// detection output [1,N,C>=6] with the score in column 4: compare only rows ONNX Runtime scores above 0.25,
// since near-tied background rows select backend-noise-dependent cells), or
// `detr` (DETR-family decoder: output 0 boxes [1,Q,4] as cxcywh, output 1 class
// logits [1,Q,C], optional output 2 mask logits [1,Q,H,W]; only queries with
// sigmoid(max logit) > 0.5 are compared, greedily matched one-to-one by class
// and box IoU, since near-tied low-score queries order differently between
// backends; fixed thresholds: box IoU >= 0.99, |score diff| <= 1e-3, mask IoU
// >= 0.98, and `atol` is ignored), or `detr_post` (DETR-family export with
// built-in postprocessing: output 0 labels int64 [1,Q], output 1 boxes [1,Q,4]
// as x1y1x2y2 pixels, output 2 scores [1,Q]; only detections ONNX Runtime
// scores above 0.5 are compared, greedily matched one-to-one by label and box
// IoU, since the internal TopK breaks near-ties differently between backends;
// same fixed thresholds as `detr`, `atol` is ignored), or `load` (NERT must
// load the model and
// plan it; nothing runs). Inputs are generated per graph input dtype: float32
// inputs get the seeded uniform [0, 1) values, int64 inputs (RT-DETR's
// orig_target_sizes, token ids) are filled with the `int64=<v>` value (default
// 1), and bool inputs are filled with false. The optional input file (`-` for
// none) replaces the seeded values of the first float32 input. In `parity` mode
// each float element must satisfy |native - ort| <= atol * (1 + |ort|), and
// int64 / bool outputs must match exactly.
struct ZooEntry {
    std::string name;
    std::string path;
    std::vector<std::vector<int64_t>> input_sizes;
    std::string mode;
    double atol = 1e-4;
    std::string input_file; // optional raw little-endian float32 input
    int64_t int64_fill = 1; // value for every int64 graph input
};

std::vector<int64_t> parse_dims(const std::string& text) {
    std::vector<int64_t> dims;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, 'x')) {
        dims.push_back(std::stoll(part));
    }
    return dims;
}

std::vector<ZooEntry> read_manifest(const std::string& path) {
    std::vector<ZooEntry> entries;
    std::ifstream stream(path);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream fields(line);
        ZooEntry entry;
        std::string shapes;
        fields >> entry.name >> entry.path >> shapes >> entry.mode >> entry.atol >> entry.input_file;
        if (entry.name.empty()) {
            continue;
        }
        if (entry.input_file == "-") {
            entry.input_file.clear();
        }
        std::string token;
        while (fields >> token) {
            const std::string int64_key = "int64=";
            if (token.compare(0, int64_key.size(), int64_key) == 0) {
                entry.int64_fill = std::stoll(token.substr(int64_key.size()));
            } else {
                ADD_FAILURE() << "manifest row " << entry.name << ": unknown token " << token;
            }
        }
        std::stringstream shape_stream(shapes);
        std::string one;
        while (std::getline(shape_stream, one, ';')) {
            entry.input_sizes.push_back(parse_dims(one));
        }
        entries.push_back(entry);
    }
    return entries;
}

// Deterministic inputs, identical for both backends, one per graph input in
// its own dtype: float32 is seeded uniform [0, 1) (the seed advances over
// float inputs only, so float-only models see the values they always did),
// int64 is `int64_fill` everywhere, and bool is all false.
std::vector<std::vector<uint8_t>> make_inputs(const std::vector<LayerInfo>& layers, int64_t int64_fill) {
    std::vector<std::vector<uint8_t>> inputs;
    uint32_t state = 0x9E3779B9U;
    for (const LayerInfo& layer : layers) {
        size_t count = 1;
        for (int64_t dim : layer.shape) {
            count *= static_cast<size_t>(dim);
        }
        std::vector<uint8_t> bytes;
        switch (layer.datatype) {
        case TensorDataType::Int64:
            bytes.resize(count * sizeof(int64_t));
            for (size_t i = 0; i < count; ++i) {
                std::memcpy(bytes.data() + i * sizeof(int64_t), &int64_fill, sizeof(int64_t));
            }
            break;
        case TensorDataType::Bool:
            bytes.assign(count, 0U);
            break;
        default:
            bytes.resize(count * sizeof(float));
            for (size_t i = 0; i < count; ++i) {
                state = state * 1664525U + 1013904223U;
                const float value = static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
                std::memcpy(bytes.data() + i * sizeof(float), &value, sizeof(float));
            }
            break;
        }
        inputs.push_back(std::move(bytes));
    }
    return inputs;
}

// An integer or bool element (anything but float) as int64 for exact compare.
bool is_integral(const TensorElement& element) { return !std::holds_alternative<float>(element); }

int64_t as_int64(const TensorElement& element) {
    return std::visit([](auto value) { return static_cast<int64_t>(value); }, element);
}

double as_double(const TensorElement& element) {
    return std::visit([](auto value) { return static_cast<double>(value); }, element);
}

struct DetrQuery {
    size_t index = 0;
    double score = 0.0;
    size_t cls = 0;
};

std::vector<DetrQuery> confident_queries(const std::vector<TensorElement>& logits, size_t queries, size_t classes) {
    std::vector<DetrQuery> result;
    for (size_t q = 0; q < queries; ++q) {
        double best = as_double(logits[q * classes]);
        size_t best_class = 0;
        for (size_t c = 1; c < classes; ++c) {
            const double value = as_double(logits[q * classes + c]);
            if (value > best) {
                best = value;
                best_class = c;
            }
        }
        const double score = 1.0 / (1.0 + std::exp(-best));
        if (score > 0.5) {
            result.push_back({q, score, best_class});
        }
    }
    return result;
}

double box_iou(const std::vector<TensorElement>& a, size_t ia, const std::vector<TensorElement>& b, size_t ib) {
    auto corners = [](const std::vector<TensorElement>& v, size_t i, double out[4]) {
        const double cx = as_double(v[i * 4]);
        const double cy = as_double(v[i * 4 + 1]);
        const double w = as_double(v[i * 4 + 2]);
        const double h = as_double(v[i * 4 + 3]);
        out[0] = cx - w / 2;
        out[1] = cy - h / 2;
        out[2] = cx + w / 2;
        out[3] = cy + h / 2;
    };
    double ba[4];
    double bb[4];
    corners(a, ia, ba);
    corners(b, ib, bb);
    const double iw = std::max(0.0, std::min(ba[2], bb[2]) - std::max(ba[0], bb[0]));
    const double ih = std::max(0.0, std::min(ba[3], bb[3]) - std::max(ba[1], bb[1]));
    const double inter = iw * ih;
    const double area_a = std::max(0.0, ba[2] - ba[0]) * std::max(0.0, ba[3] - ba[1]);
    const double area_b = std::max(0.0, bb[2] - bb[0]) * std::max(0.0, bb[3] - bb[1]);
    const double uni = area_a + area_b - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

double mask_iou(const std::vector<TensorElement>& a, const std::vector<TensorElement>& b, size_t qa, size_t qb,
                size_t pixels) {
    size_t inter = 0;
    size_t uni = 0;
    for (size_t p = 0; p < pixels; ++p) {
        const bool ma = as_double(a[qa * pixels + p]) > 0.0;
        const bool mb = as_double(b[qb * pixels + p]) > 0.0;
        inter += (ma && mb) ? 1U : 0U;
        uni += (ma || mb) ? 1U : 0U;
    }
    return uni == 0 ? 1.0 : static_cast<double>(inter) / static_cast<double>(uni);
}

void check_detr(const ZooEntry& entry, const std::vector<std::vector<TensorElement>>& native_out,
                const std::vector<std::vector<int64_t>>& native_shapes,
                const std::vector<std::vector<TensorElement>>& ort_out,
                const std::vector<std::vector<int64_t>>& ort_shapes) {
    ASSERT_GE(ort_out.size(), 2U) << "detr needs boxes and class logits";
    ASSERT_EQ(ort_shapes[0].size(), 3U);
    ASSERT_EQ(ort_shapes[0][2], 4);
    ASSERT_EQ(ort_shapes[1].size(), 3U);
    for (size_t o = 0; o < ort_out.size(); ++o) {
        ASSERT_EQ(native_shapes[o], ort_shapes[o]) << "output " << o << " shape differs";
        ASSERT_EQ(native_out[o].size(), ort_out[o].size()) << "output " << o << " size differs";
    }
    const size_t queries = static_cast<size_t>(ort_shapes[1][1]);
    const size_t classes = static_cast<size_t>(ort_shapes[1][2]);
    const bool has_masks = ort_out.size() >= 3;
    size_t pixels = 0;
    if (has_masks) {
        ASSERT_EQ(ort_shapes[2].size(), 4U);
        pixels = static_cast<size_t>(ort_shapes[2][2] * ort_shapes[2][3]);
    }

    std::vector<DetrQuery> ort_conf = confident_queries(ort_out[1], queries, classes);
    const std::vector<DetrQuery> native_conf = confident_queries(native_out[1], queries, classes);
    std::stable_sort(ort_conf.begin(), ort_conf.end(),
                     [](const DetrQuery& x, const DetrQuery& y) { return x.score > y.score; });

    double min_box_iou = 1.0;
    double max_score_diff = 0.0;
    double min_mask_iou = 1.0;
    std::string failure;
    std::vector<bool> used(native_conf.size(), false);
    for (const DetrQuery& oq : ort_conf) {
        double best_iou = -1.0;
        size_t best = native_conf.size();
        for (size_t n = 0; n < native_conf.size(); ++n) {
            if (used[n] || native_conf[n].cls != oq.cls) {
                continue;
            }
            const double iou = box_iou(ort_out[0], oq.index, native_out[0], native_conf[n].index);
            if (iou > best_iou) {
                best_iou = iou;
                best = n;
            }
        }
        if (best == native_conf.size()) {
            if (failure.empty()) {
                failure = "ORT query " + std::to_string(oq.index) + " has no unmatched NERT query of class " +
                          std::to_string(oq.cls);
            }
            continue;
        }
        used[best] = true;
        const DetrQuery& nq = native_conf[best];
        const double score_diff = std::fabs(nq.score - oq.score);
        min_box_iou = std::min(min_box_iou, best_iou);
        max_score_diff = std::max(max_score_diff, score_diff);
        if (best_iou < 0.99 && failure.empty()) {
            failure = "ORT query " + std::to_string(oq.index) + " box IoU " + std::to_string(best_iou) + " < 0.99";
        }
        if (score_diff > 1e-3 && failure.empty()) {
            failure = "ORT query " + std::to_string(oq.index) + " score diff " + std::to_string(score_diff) + " > 1e-3";
        }
        if (has_masks) {
            const double miou = mask_iou(ort_out[2], native_out[2], oq.index, nq.index, pixels);
            min_mask_iou = std::min(min_mask_iou, miou);
            if (miou < 0.98 && failure.empty()) {
                failure = "ORT query " + std::to_string(oq.index) + " mask IoU " + std::to_string(miou) + " < 0.98";
            }
        }
    }
    std::cout << "ZOO " << entry.name << " confident_queries=" << ort_conf.size() << " min_box_iou=" << min_box_iou
              << " max_score_diff=" << max_score_diff;
    if (has_masks) {
        std::cout << " min_mask_iou=" << min_mask_iou;
    }
    std::cout << std::endl;

    EXPECT_GT(ort_conf.size(), 0U) << "no confident queries; the input is not a real image";
    EXPECT_EQ(ort_conf.size(), native_conf.size()) << "confident query counts differ";
    EXPECT_TRUE(failure.empty()) << failure;
}

double corner_iou(const std::vector<TensorElement>& a, size_t ia, const std::vector<TensorElement>& b, size_t ib) {
    double ba[4];
    double bb[4];
    for (size_t k = 0; k < 4; ++k) {
        ba[k] = as_double(a[ia * 4 + k]);
        bb[k] = as_double(b[ib * 4 + k]);
    }
    const double iw = std::max(0.0, std::min(ba[2], bb[2]) - std::max(ba[0], bb[0]));
    const double ih = std::max(0.0, std::min(ba[3], bb[3]) - std::max(ba[1], bb[1]));
    const double inter = iw * ih;
    const double area_a = std::max(0.0, ba[2] - ba[0]) * std::max(0.0, ba[3] - ba[1]);
    const double area_b = std::max(0.0, bb[2] - bb[0]) * std::max(0.0, bb[3] - bb[1]);
    const double uni = area_a + area_b - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

struct DetrDetection {
    size_t index = 0;
    double score = 0.0;
    int64_t label = 0;
};

std::vector<DetrDetection> confident_detections(const std::vector<TensorElement>& labels,
                                                const std::vector<TensorElement>& scores) {
    std::vector<DetrDetection> result;
    for (size_t q = 0; q < scores.size(); ++q) {
        const double score = as_double(scores[q]);
        if (score > 0.5) {
            result.push_back({q, score, as_int64(labels[q])});
        }
    }
    return result;
}

void check_detr_post(const ZooEntry& entry, const std::vector<std::vector<TensorElement>>& native_out,
                     const std::vector<std::vector<int64_t>>& native_shapes,
                     const std::vector<std::vector<TensorElement>>& ort_out,
                     const std::vector<std::vector<int64_t>>& ort_shapes) {
    ASSERT_EQ(ort_out.size(), 3U) << "detr_post needs labels, boxes, and scores";
    ASSERT_EQ(ort_shapes[0].size(), 2U);
    ASSERT_EQ(ort_shapes[1].size(), 3U);
    ASSERT_EQ(ort_shapes[1][2], 4);
    ASSERT_EQ(ort_shapes[2].size(), 2U);
    ASSERT_EQ(ort_shapes[0][1], ort_shapes[1][1]);
    ASSERT_EQ(ort_shapes[2][1], ort_shapes[1][1]);
    for (size_t o = 0; o < ort_out.size(); ++o) {
        ASSERT_EQ(native_shapes[o], ort_shapes[o]) << "output " << o << " shape differs";
        ASSERT_EQ(native_out[o].size(), ort_out[o].size()) << "output " << o << " size differs";
    }
    ASSERT_TRUE(ort_out[0].empty() || is_integral(ort_out[0][0])) << "ORT labels are not integral";
    ASSERT_TRUE(native_out[0].empty() || is_integral(native_out[0][0])) << "NERT labels are not integral";

    std::vector<DetrDetection> ort_conf = confident_detections(ort_out[0], ort_out[2]);
    const std::vector<DetrDetection> native_conf = confident_detections(native_out[0], native_out[2]);
    std::stable_sort(ort_conf.begin(), ort_conf.end(),
                     [](const DetrDetection& x, const DetrDetection& y) { return x.score > y.score; });

    double min_box_iou = 1.0;
    double max_score_diff = 0.0;
    std::string failure;
    std::vector<bool> used(native_conf.size(), false);
    for (const DetrDetection& od : ort_conf) {
        double best_iou = -1.0;
        size_t best = native_conf.size();
        for (size_t n = 0; n < native_conf.size(); ++n) {
            if (used[n] || native_conf[n].label != od.label) {
                continue;
            }
            const double iou = corner_iou(ort_out[1], od.index, native_out[1], native_conf[n].index);
            if (iou > best_iou) {
                best_iou = iou;
                best = n;
            }
        }
        if (best == native_conf.size()) {
            if (failure.empty()) {
                failure = "ORT detection " + std::to_string(od.index) + " has no unmatched NERT detection of label " +
                          std::to_string(od.label);
            }
            continue;
        }
        used[best] = true;
        const double score_diff = std::fabs(native_conf[best].score - od.score);
        min_box_iou = std::min(min_box_iou, best_iou);
        max_score_diff = std::max(max_score_diff, score_diff);
        if (best_iou < 0.99 && failure.empty()) {
            failure = "ORT detection " + std::to_string(od.index) + " box IoU " + std::to_string(best_iou) + " < 0.99";
        }
        if (score_diff > 1e-3 && failure.empty()) {
            failure =
                "ORT detection " + std::to_string(od.index) + " score diff " + std::to_string(score_diff) + " > 1e-3";
        }
    }
    std::cout << "ZOO " << entry.name << " confident_queries=" << ort_conf.size() << " min_box_iou=" << min_box_iou
              << " max_score_diff=" << max_score_diff << std::endl;

    EXPECT_GT(ort_conf.size(), 0U) << "no confident detections; the input is not a real image";
    EXPECT_EQ(ort_conf.size(), native_conf.size()) << "confident detection counts differ";
    EXPECT_TRUE(failure.empty()) << failure;
}

void check_entry(const ZooEntry& entry) {
    SCOPED_TRACE(entry.name);
    ASSERT_TRUE(std::filesystem::exists(entry.path)) << "model does not exist: " << entry.path;

    if (entry.mode == "load") {
        try {
            NertInfer native(entry.path, /*use_gpu=*/false, 1, entry.input_sizes);
            std::cout << "ZOO " << entry.name << " LOADED" << std::endl;
        } catch (const std::exception& ex) {
            FAIL() << entry.name << ": NERT load failed: " << ex.what();
        }
        return;
    }
    ASSERT_TRUE(entry.mode == "parity" || entry.mode == "e2e" || entry.mode == "detr" || entry.mode == "detr_post")
        << "unknown mode " << entry.mode;

    std::unique_ptr<NertInfer> native;
    try {
        native = std::make_unique<NertInfer>(entry.path, /*use_gpu=*/false, 1, entry.input_sizes);
    } catch (const std::exception& ex) {
        FAIL() << entry.name << ": NERT load failed: " << ex.what();
    }
    const std::vector<LayerInfo> layers = native->get_inference_metadata().getInputs();
    auto inputs = make_inputs(layers, entry.int64_fill);
    if (!entry.input_file.empty()) {
        size_t target = layers.size();
        for (size_t i = 0; i < layers.size(); ++i) {
            if (layers[i].datatype == TensorDataType::Float32) {
                target = i;
                break;
            }
        }
        ASSERT_LT(target, layers.size()) << "an input file needs a float32 graph input";
        std::ifstream stream(entry.input_file, std::ios::binary);
        ASSERT_TRUE(stream.good()) << "cannot open input file " << entry.input_file;
        stream.read(reinterpret_cast<char*>(inputs[target].data()),
                    static_cast<std::streamsize>(inputs[target].size()));
        ASSERT_EQ(stream.gcount(), static_cast<std::streamsize>(inputs[target].size())) << "input file is truncated";
    }
    std::vector<std::vector<TensorElement>> native_out;
    std::vector<std::vector<int64_t>> native_shapes;
    try {
        std::tie(native_out, native_shapes) = native->get_infer_results(inputs);
    } catch (const std::exception& ex) {
        FAIL() << entry.name << ": NERT failed: " << ex.what();
    }
    std::vector<std::vector<TensorElement>> ort_out;
    std::vector<std::vector<int64_t>> ort_shapes;
    try {
        // ORTInfer takes per-input dims without the batch axis, plus the batch
        // size; NertInfer takes full shapes.
        std::vector<std::vector<int64_t>> ort_sizes;
        size_t batch = 1;
        for (const auto& dims : entry.input_sizes) {
            ort_sizes.emplace_back(dims.begin() + 1, dims.end());
            batch = static_cast<size_t>(dims.front());
        }
        ORTInfer ort(entry.path, /*use_gpu=*/false, batch, ort_sizes);
        std::tie(ort_out, ort_shapes) = ort.get_infer_results(inputs);
    } catch (const std::exception& ex) {
        FAIL() << entry.name << ": ONNX_RUNTIME failed: " << ex.what();
    }

    ASSERT_EQ(native_out.size(), ort_out.size()) << "output count differs";
    if (entry.mode == "detr") {
        check_detr(entry, native_out, native_shapes, ort_out, ort_shapes);
        return;
    }
    if (entry.mode == "detr_post") {
        check_detr_post(entry, native_out, native_shapes, ort_out, ort_shapes);
        return;
    }
    if (entry.mode == "e2e") {
        ASSERT_EQ(ort_shapes[0].size(), 3U);
        ASSERT_GE(ort_shapes[0][2], 6);
        const size_t rows = static_cast<size_t>(ort_shapes[0][1]);
        const size_t cols = static_cast<size_t>(ort_shapes[0][2]);
        size_t confident = 0;
        double max_scaled = 0.0;
        for (size_t r = 0; r < rows; ++r) {
            if (as_double(ort_out[0][r * cols + 4]) <= 0.25) {
                continue;
            }
            ++confident;
            for (size_t c = 0; c < cols; ++c) {
                const double ref = as_double(ort_out[0][r * cols + c]);
                const double diff = std::fabs(as_double(native_out[0][r * cols + c]) - ref);
                max_scaled = std::max(max_scaled, diff / (1.0 + std::fabs(ref)));
            }
        }
        std::cout << "ZOO " << entry.name << " confident_rows=" << confident << " max_scaled_diff=" << max_scaled
                  << " atol=" << entry.atol << std::endl;
        EXPECT_GT(confident, 0U) << "no confident detections; the input is not a real image";
        EXPECT_LE(max_scaled, entry.atol) << "confident rows exceed tolerance";
        return;
    }
    for (size_t o = 0; o < ort_out.size(); ++o) {
        ASSERT_EQ(native_shapes[o], ort_shapes[o]) << "output " << o << " shape differs";
        ASSERT_EQ(native_out[o].size(), ort_out[o].size()) << "output " << o << " size differs";
        if (!ort_out[o].empty() && (is_integral(ort_out[o][0]) || is_integral(native_out[o][0]))) {
            // int64 / bool outputs (labels, masks) compare exactly.
            size_t mismatches = 0;
            size_t first = 0;
            for (size_t i = 0; i < ort_out[o].size(); ++i) {
                if (!is_integral(native_out[o][i]) || !is_integral(ort_out[o][i]) ||
                    as_int64(native_out[o][i]) != as_int64(ort_out[o][i])) {
                    if (mismatches == 0) {
                        first = i;
                    }
                    ++mismatches;
                }
            }
            std::cout << "ZOO " << entry.name << " output" << o << " exact mismatches=" << mismatches << std::endl;
            EXPECT_EQ(mismatches, 0U) << "output " << o << " differs first at element " << first;
            continue;
        }
        double max_ref = 0.0;
        double max_diff = 0.0;
        double max_scaled = 0.0;
        for (size_t i = 0; i < ort_out[o].size(); ++i) {
            const double ref = as_double(ort_out[o][i]);
            const double diff = std::fabs(as_double(native_out[o][i]) - ref);
            max_ref = std::max(max_ref, std::fabs(ref));
            max_diff = std::max(max_diff, diff);
            max_scaled = std::max(max_scaled, diff / (1.0 + std::fabs(ref)));
        }
        std::cout << "ZOO " << entry.name << " output" << o << " max_abs_diff=" << max_diff << " max_ref=" << max_ref
                  << " max_scaled_diff=" << max_scaled << " atol=" << entry.atol << std::endl;
        EXPECT_LE(max_scaled, entry.atol) << "output " << o << " exceeds tolerance";
    }
}

} // namespace

// Gate check for the neuriplo-tasks operator coverage packet: every model in
// the manifest loads on NERT, and `parity` rows match ONNX_RUNTIME on the
// same seeded input. A missing manifest fails, never skips.
TEST(ModelZooParityTest, ManifestModelsMatchOnnxRuntime) {
    const char* manifest = std::getenv(kManifestEnv);
    ASSERT_TRUE(manifest != nullptr && manifest[0] != '\0') << kManifestEnv << " is not set";
    const std::vector<ZooEntry> entries = read_manifest(manifest);
    ASSERT_FALSE(entries.empty()) << "manifest has no entries: " << manifest;

    const char* only = std::getenv(kOnlyEnv);
    for (const ZooEntry& entry : entries) {
        if (only != nullptr && only[0] != '\0' && entry.name != only) {
            continue;
        }
        check_entry(entry);
    }
}
