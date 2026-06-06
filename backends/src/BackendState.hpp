#pragma once

// Lifecycle state of an inference backend.
enum class BackendState { Uninitialized, Loading, Ready, Failed };

// Returns true only for the explicitly allowed lifecycle transitions.
inline bool is_valid_transition(BackendState from, BackendState to) noexcept {
    switch (from) {
    case BackendState::Uninitialized:
        return to == BackendState::Loading;
    case BackendState::Loading:
        return to == BackendState::Ready || to == BackendState::Failed;
    case BackendState::Ready:
        return to == BackendState::Failed;
    case BackendState::Failed:
        return to == BackendState::Loading;
    }
    return false;
}

inline const char* to_string(BackendState s) noexcept {
    switch (s) {
    case BackendState::Uninitialized:
        return "Uninitialized";
    case BackendState::Loading:
        return "Loading";
    case BackendState::Ready:
        return "Ready";
    case BackendState::Failed:
        return "Failed";
    }
    return "Unknown";
}
