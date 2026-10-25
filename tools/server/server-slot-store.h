#pragma once

#include "server-http.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

// slot files on a shared directory, see --slot-store
// layout: <dir>/<fingerprint>/<key>.slot, the directory can be a soft-mounted NFS volume shared by many servers

// a fingerprint or a key: 1 to 128 characters of [A-Za-z0-9_-]
bool slot_store_valid_name(const std::string & name);

// "<key>.slot" with a valid key
bool slot_store_valid_filename(const std::string & filename);

// a unique temporary file next to path: .<key>.<random>.tmp
std::string slot_store_tmp_path(const std::string & path);

// fsync tmp, check that it holds n_bytes, and rename it to path; on failure tmp is removed
bool slot_store_commit(const std::string & tmp, const std::string & path, uint64_t n_bytes, std::string & err);

struct server_slot_store {
    // creates <dir>/<fingerprint>/ and starts the janitor, throws on error
    server_slot_store(const std::string & dir, const std::string & fingerprint, const std::string & instance);
    ~server_slot_store();

    // <dir>/<fingerprint>/
    std::string save_path() const;

    // drain handshake: the router sets the flag when it has saved what it needs from this server
    std::atomic<bool> drained = false;

    // true if no router request came in the last lost_ms milliseconds, or never
    bool router_lost(int64_t lost_ms) const;

    // answered on the HTTP threads with plain file system calls
    server_http_context::handler_t get_info;
    server_http_context::handler_t get_list;
    server_http_context::handler_t post_evict;
    server_http_context::handler_t get_drained;
    server_http_context::handler_t post_drained;

private:
    std::string dir;
    std::string fingerprint;
    std::string instance;

    std::atomic<int64_t> t_router_seen_ms = -1; // ggml_time_ms(), -1 = never

    std::mutex              janitor_mutex;
    std::condition_variable janitor_cv;
    bool                    janitor_stop = false;
    std::thread             janitor;

    void router_seen();
    void remove_old_tmp() const;
};
