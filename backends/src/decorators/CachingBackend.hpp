#pragma once
#include "BackendDecorator.hpp"
#include "InferenceInterface.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

// Decorator that memoizes inference results in a bounded LRU cache.
//
// Wraps any InferenceInterface and keys cached outputs on a hash of the raw
// input bytes plus the per-tensor sizes. On a cache hit the wrapped backend is
// not invoked and a copy of the previously computed tuple is returned; on a
// miss the call is forwarded, the result stored, and the least-recently-used
// entry evicted once the capacity is exceeded.
//
// DETERMINISM: this decorator assumes that identical inputs always yield
// identical outputs. It is strictly opt-in (only present when explicitly
// constructed) and MUST NOT be placed around nondeterministic backends, since
// returning a stale cached result would silently change observable behavior.
class CachingBackend : public BackendDecorator {

  public:
    explicit CachingBackend(std::unique_ptr<InferenceInterface> inner, size_t capacity = 16)
        : BackendDecorator(std::move(inner)), capacity_(capacity == 0 ? 1 : capacity) {}

    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>& input_tensors) override {
        const size_t key = compute_key(input_tensors);

        auto it = entries_.find(key);
        if (it != entries_.end()) {
            // Cache hit: promote to most-recently-used and return a copy.
            lru_order_.splice(lru_order_.begin(), lru_order_, it->second.order_it);
            return it->second.value;
        }

        // Cache miss: forward to the wrapped backend before mutating state so an
        // exception leaves the cache untouched.
        auto result = BackendDecorator::get_infer_results(input_tensors);

        lru_order_.push_front(key);
        entries_.emplace(key, Entry{lru_order_.begin(), result});

        if (entries_.size() > capacity_) {
            const size_t evict_key = lru_order_.back();
            lru_order_.pop_back();
            entries_.erase(evict_key);
        }

        return result;
    }

    void clear_cache() noexcept override {
        // noexcept: container operations should not throw here, but guard anyway
        // so a faulty allocator/inner backend can never escape this contract.
        try {
            entries_.clear();
            lru_order_.clear();
            BackendDecorator::clear_cache();
        } catch (...) {
        }
    }

  private:
    using ResultTuple = std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>;

    struct Entry {
        std::list<size_t>::iterator order_it;
        ResultTuple value;
    };

    static void hash_combine(size_t& seed, size_t value) noexcept {
        // Boost-style mixing constant to spread bits across combined hashes.
        seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    }

    static size_t compute_key(const std::vector<std::vector<uint8_t>>& input_tensors) noexcept {
        size_t seed = 0;
        // Fold in the tensor count and each tensor's size before its bytes so
        // that differing partitions of the same byte stream do not collide.
        hash_combine(seed, std::hash<size_t>{}(input_tensors.size()));
        for (const auto& tensor : input_tensors) {
            hash_combine(seed, std::hash<size_t>{}(tensor.size()));
            for (const uint8_t byte : tensor) {
                hash_combine(seed, std::hash<uint8_t>{}(byte));
            }
        }
        return seed;
    }

    size_t capacity_;
    std::list<size_t> lru_order_;
    std::unordered_map<size_t, Entry> entries_;
};
