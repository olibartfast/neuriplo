// Consumer C ABI -- log callback (neuriplo_set_log_callback, [D-16]).
//
// One glog LogSink is registered the first time a callback is installed and
// stays registered for the life of the process. The sink forwards to the
// current {callback, min_level, user_data}, guarded by one mutex that send()
// holds while invoking the callback and neuriplo_set_log_callback takes to
// swap the state: when set returns, no invocation of the old callback is in
// progress. glog's own stderr output is untouched.

#include "neuriplo/neuriplo_c.h"
#include "neuriplo_c_internal.hpp"

#include <glog/logging.h>
#include <mutex>
#include <string>

namespace {

struct CallbackState {
    neuriplo_log_callback_t callback = nullptr;
    neuriplo_log_level_t min_level = NEURIPLO_LOG_LEVEL_INFO;
    void* user_data = nullptr;
};

std::mutex g_state_mutex;
CallbackState g_state;
std::once_flag g_sink_registered;

neuriplo_log_level_t map_severity(google::LogSeverity severity) noexcept {
    switch (severity) {
    case google::GLOG_INFO:
        return NEURIPLO_LOG_LEVEL_INFO;
    case google::GLOG_WARNING:
        return NEURIPLO_LOG_LEVEL_WARNING;
    case google::GLOG_ERROR:
    case google::GLOG_FATAL:
        return NEURIPLO_LOG_LEVEL_ERROR;
    default:
        return NEURIPLO_LOG_LEVEL_ERROR;
    }
}

class CallbackSink : public google::LogSink {
  public:
    void send(google::LogSeverity severity, const char* /*full_filename*/, const char* /*base_filename*/, int /*line*/,
              const google::LogMessageTime& /*logmsgtime*/, const char* message, size_t message_len) override {
        try {
            const neuriplo_log_level_t level = map_severity(severity);
            const std::lock_guard<std::mutex> lock(g_state_mutex);
            if (g_state.callback == nullptr || level < g_state.min_level) {
                return;
            }
            // message is not NUL-terminated at message_len.
            const std::string text(message, message_len);
            g_state.callback(level, text.c_str(), g_state.user_data);
        } catch (...) {
        }
    }
};

bool is_defined_level(neuriplo_log_level_t level) noexcept {
    return level == NEURIPLO_LOG_LEVEL_INFO || level == NEURIPLO_LOG_LEVEL_WARNING || level == NEURIPLO_LOG_LEVEL_ERROR;
}

} // namespace

neuriplo_status_t NEURIPLO_CALL neuriplo_set_log_callback(neuriplo_log_callback_t callback,
                                                          neuriplo_log_level_t min_level,
                                                          void* user_data) NEURIPLO_NOEXCEPT {
    if (callback != nullptr && !is_defined_level(min_level)) {
        return neuriplo_capi::invalid_argument("neuriplo_set_log_callback", "min_level is not a defined level");
    }
    return neuriplo_capi::guarded(
        neuriplo_capi::GuardContext::Other, "neuriplo_set_log_callback", [&]() -> neuriplo_status_t {
            if (callback != nullptr) {
                // Registered once, never removed or deleted: glog may
                // call it from any thread until process exit.
                std::call_once(g_sink_registered, [] { google::AddLogSink(new CallbackSink()); });
            }
            {
                const std::lock_guard<std::mutex> lock(g_state_mutex);
                if (callback != nullptr) {
                    g_state.callback = callback;
                    g_state.min_level = min_level;
                    g_state.user_data = user_data;
                } else {
                    g_state = CallbackState{};
                }
            }
            neuriplo_capi::clear_last_error();
            return NEURIPLO_STATUS_OK;
        });
}
