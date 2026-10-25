#include "server-slot-store.h"
#include "server-common.h"

#include "ggml.h"

#include "blake3/blake3.h"

#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

static constexpr int  SLOT_STORE_JANITOR_SECONDS = 60;
static constexpr auto SLOT_STORE_TMP_MAX_AGE     = std::chrono::hours(1);

bool slot_store_valid_name(const std::string & name) {
    if (name.empty() || name.size() > 128) {
        return false;
    }
    for (char c : name) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_')) {
            return false;
        }
    }
    return true;
}

static const std::string SLOT_EXT = ".slot";

bool slot_store_valid_filename(const std::string & filename) {
    return filename.size() > SLOT_EXT.size() &&
           filename.compare(filename.size() - SLOT_EXT.size(), SLOT_EXT.size(), SLOT_EXT) == 0 &&
           slot_store_valid_name(filename.substr(0, filename.size() - SLOT_EXT.size()));
}

std::string slot_store_tmp_path(const std::string & path) {
    thread_local std::mt19937_64 rng{std::random_device{}()};
    const fs::path p = fs::u8path(path);
    char rnd[17];
    snprintf(rnd, sizeof(rnd), "%016" PRIx64, (uint64_t) rng());
    return (p.parent_path() / ("." + p.stem().string() + "." + rnd + ".tmp")).string();
}

static const char SLOT_TRAILER_MAGIC[8] = {'B', '3', 'S', 'L', 'O', 'T', '0', '1'};

static void blake3_of(const uint8_t * data, size_t size, uint8_t * out) {
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, data, size);
    blake3_hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
}

// appends the trailer to the file at path
static bool append_trailer(const std::string & path, std::string & err) {
    FILE * f = ggml_fopen(path.c_str(), "rb");
    if (!f) {
        err = std::string("open: ") + strerror(errno);
        return false;
    }
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    std::vector<uint8_t> buf(1 << 20);
    size_t n;
    errno = 0;
    while ((n = fread(buf.data(), 1, buf.size(), f)) > 0) {
        blake3_hasher_update(&hasher, buf.data(), n);
    }
    const bool failed = ferror(f) != 0;
    const int errno_read = errno;
    fclose(f);
    if (failed) {
        err = std::string("read: ") + strerror(errno_read ? errno_read : EIO);
        return false;
    }
    uint8_t digest[BLAKE3_OUT_LEN];
    blake3_hasher_finalize(&hasher, digest, sizeof(digest));

    f = ggml_fopen(path.c_str(), "ab");
    if (!f) {
        err = std::string("open: ") + strerror(errno);
        return false;
    }
    bool ok = fwrite(SLOT_TRAILER_MAGIC, 1, sizeof(SLOT_TRAILER_MAGIC), f) == sizeof(SLOT_TRAILER_MAGIC) &&
              fwrite(digest, 1, sizeof(digest), f) == sizeof(digest);
    int errno_write = errno;
    if (fclose(f) != 0 && ok) {
        ok = false;
        errno_write = errno;
    }
    if (!ok) {
        err = std::string("write: ") + strerror(errno_write);
    }
    return ok;
}

bool slot_store_read(const std::string & path, bool checked, std::vector<uint8_t> & data, std::string & err) {
    // test hook: overwrites the file once it was read, like another server replacing it during a restore
    static const bool test_replace = getenv("LLAMA_SERVER_TEST_RESTORE_REPLACE") != nullptr;

    data.clear();
    FILE * f = ggml_fopen(path.c_str(), "rb");
    if (!f) {
        err = std::string("open: ") + strerror(errno);
        return false;
    }
    std::error_code ec;
    const uintmax_t size_hint = fs::file_size(fs::u8path(path), ec);
    if (!ec) {
        data.reserve(size_hint);
    }
    std::vector<uint8_t> buf(1 << 20);
    size_t n;
    errno = 0;
    while ((n = fread(buf.data(), 1, buf.size(), f)) > 0) {
        data.insert(data.end(), buf.begin(), buf.begin() + n);
    }
    const bool failed = ferror(f) != 0;
    const int errno_read = errno;
    fclose(f);
    if (failed) {
        err = std::string("read: ") + strerror(errno_read ? errno_read : EIO);
        return false;
    }
    if (checked && test_replace) {
        FILE * fr = ggml_fopen(path.c_str(), "wb");
        if (fr) {
            fputs("replaced", fr);
            fclose(fr);
        }
    }
    if (!checked) {
        return true;
    }

    if (data.size() < SLOT_STORE_TRAILER_SIZE ||
        memcmp(data.data() + data.size() - SLOT_STORE_TRAILER_SIZE, SLOT_TRAILER_MAGIC, sizeof(SLOT_TRAILER_MAGIC)) != 0) {
        err = "no checksum trailer";
        return false;
    }
    const size_t n_payload = data.size() - SLOT_STORE_TRAILER_SIZE;
    uint8_t digest[BLAKE3_OUT_LEN];
    blake3_of(data.data(), n_payload, digest);
    if (memcmp(data.data() + n_payload + sizeof(SLOT_TRAILER_MAGIC), digest, sizeof(digest)) != 0) {
        err = "checksum mismatch";
        return false;
    }
    data.resize(n_payload);
    return true;
}

// random id generated once per server process, lets a router notice a restart
static const std::string & slot_store_boot_id() {
    static const std::string id = [] {
        std::random_device rd;
        std::mt19937_64 rng{((uint64_t) rd() << 32) ^ rd()};
        char buf[17];
        snprintf(buf, sizeof(buf), "%016" PRIx64, (uint64_t) rng());
        return std::string(buf);
    }();
    return id;
}

bool slot_store_commit(const std::string & tmp, const std::string & path, uint64_t n_bytes, std::string & err) {
    // test hook: "before" fails the rename, "after" renames and then reports an error
    static const std::string test_rename_error = [] {
        const char * val = getenv("LLAMA_SERVER_TEST_RENAME_ERROR");
        return std::string(val ? val : "");
    }();
    // test hook: drops the last byte of tmp, like a buffered tail write that failed at close
    static const bool test_truncate = getenv("LLAMA_SERVER_TEST_SAVE_TRUNCATE") != nullptr;

    auto fail = [&](const std::string & msg) {
        std::error_code ec;
        fs::remove(fs::u8path(tmp), ec);
        err = msg;
        return false;
    };

    if (test_truncate && n_bytes > 0) {
        std::error_code ec;
        fs::resize_file(fs::u8path(tmp), n_bytes - 1, ec);
    }

    // the hash is in the one file that the rename publishes, so the two cannot get out of step
    std::string err_trailer;
    if (!append_trailer(tmp, err_trailer)) {
        return fail("trailer: " + err_trailer);
    }
    n_bytes += SLOT_STORE_TRAILER_SIZE;

#ifndef _WIN32
    const int fd = open(tmp.c_str(), O_WRONLY | O_CLOEXEC);
    if (fd < 0) {
        return fail(std::string("open: ") + strerror(errno));
    }
    // over NFS, write errors can show up only on fsync or close
    const int rc_sync  = fsync(fd);
    const int errno_sync = errno;
    const int rc_close = close(fd);
    if (rc_sync != 0) {
        return fail(std::string("fsync: ") + strerror(errno_sync));
    }
    if (rc_close != 0) {
        return fail(std::string("close: ") + strerror(errno));
    }
#endif

    // the writers do not check every close, a tail lost there shows up as a short file
    std::error_code ec;
    const uintmax_t size = fs::file_size(fs::u8path(tmp), ec);
    if (ec) {
        return fail("stat: " + ec.message());
    }
    if (size != n_bytes) {
        return fail(string_format("the file has %ju bytes, %" PRIu64 " were written", size, n_bytes));
    }

    if (test_rename_error != "before") {
        fs::rename(fs::u8path(tmp), fs::u8path(path), ec);
    }
    if (!test_rename_error.empty()) {
        ec = std::make_error_code(std::errc::io_error);
    }
    if (!ec) {
        return true;
    }

    // over NFS, a retried rename can report an error after the first attempt was done
    std::error_code ec_path;
    std::error_code ec_tmp;
    const bool path_exists = fs::exists(fs::u8path(path), ec_path);
    const bool tmp_exists  = fs::exists(fs::u8path(tmp),  ec_tmp);
    if (path_exists && !ec_path && !tmp_exists && !ec_tmp) {
        SRV_WRN("rename '%s' reported '%s', but it was done\n", path.c_str(), ec.message().c_str());
        return true;
    }
    return fail("rename: " + ec.message());
}

static int64_t to_unix_seconds(fs::file_time_type t) {
    const auto t_sys = std::chrono::system_clock::now() +
        std::chrono::duration_cast<std::chrono::system_clock::duration>(t - fs::file_time_type::clock::now());
    return std::chrono::duration_cast<std::chrono::seconds>(t_sys.time_since_epoch()).count();
}

static bool is_vanished(const std::error_code & ec) {
    return ec == std::errc::no_such_file_or_directory;
}

// creates the directory with mode 0770, the umask must not take the group bits away
// an existing directory is left as is, on a shared volume it can belong to another user
static void make_dir(const fs::path & path) {
    std::error_code ec;
    if (fs::create_directory(path, ec)) {
        fs::permissions(path, fs::perms::owner_all | fs::perms::group_all, fs::perm_options::replace, ec);
    } else if (!ec) {
        const fs::perms perms = fs::status(path, ec).permissions();
        if (!ec && (perms & fs::perms::group_all) != fs::perms::group_all) {
            SRV_WRN("slot store: '%s' has mode %03o, other servers of the group may not be able to use it\n",
                    path.string().c_str(), (unsigned) (perms & fs::perms::mask));
        }
    }
    if (ec) {
        throw std::runtime_error("cannot create '" + path.string() + "': " + ec.message());
    }
}

static server_http_res_ptr make_res(const json & data) {
    auto res = std::make_unique<server_http_res>();
    res->data = safe_json_to_str(data);
    return res;
}

static server_http_res_ptr make_err(const std::string & message, error_type type) {
    const json err = format_error_response(message, type);
    auto res = std::make_unique<server_http_res>();
    res->status = json_value(err, "code", 500);
    res->data = safe_json_to_str({{"error", err}});
    return res;
}

server_slot_store::server_slot_store(const std::string & dir, const std::string & fingerprint, const std::string & instance)
        : dir(dir), fingerprint(fingerprint), instance(instance) {
    if (!slot_store_valid_name(fingerprint)) {
        throw std::invalid_argument("invalid --slot-fingerprint '" + fingerprint + "', expected 1 to 128 characters of [A-Za-z0-9_-]");
    }
    make_dir(fs::u8path(dir));
    make_dir(fs::u8path(dir) / fingerprint);

    get_info = [this](const server_http_req &) {
        router_seen();
        return make_res({{"fingerprint", this->fingerprint}, {"instance", this->instance}, {"boot_id", slot_store_boot_id()}});
    };

    get_list = [this](const server_http_req &) {
        router_seen();
        json files = json::array();
        std::error_code ec;
        for (fs::directory_iterator it(fs::u8path(this->dir), ec), end; !ec && it != end; it.increment(ec)) {
            const std::string fp = it->path().filename().string();
            if (!slot_store_valid_name(fp)) {
                continue;
            }
            std::error_code ec_fp;
            for (fs::directory_iterator jt(it->path(), ec_fp), end_fp; !ec_fp && jt != end_fp; jt.increment(ec_fp)) {
                const std::string name = jt->path().filename().string();
                if (!slot_store_valid_filename(name)) {
                    continue;
                }
                std::error_code ec_file;
                const uintmax_t size = fs::file_size(jt->path(), ec_file);
                fs::file_time_type mtime;
                if (!ec_file) {
                    mtime = fs::last_write_time(jt->path(), ec_file);
                }
                if (is_vanished(ec_file)) {
                    continue;
                }
                if (ec_file) {
                    return make_err("cannot read '" + jt->path().string() + "': " + ec_file.message(), ERROR_TYPE_SERVER);
                }
                files.push_back({
                    {"fingerprint", fp},
                    {"key",         name.substr(0, name.size() - SLOT_EXT.size())},
                    {"size",        (uint64_t) size},
                    {"mtime",       to_unix_seconds(mtime)},
                });
            }
            if (ec_fp && !is_vanished(ec_fp) && ec_fp != std::errc::not_a_directory) {
                return make_err("cannot list '" + it->path().string() + "': " + ec_fp.message(), ERROR_TYPE_SERVER);
            }
        }
        if (ec) {
            return make_err("cannot list '" + this->dir + "': " + ec.message(), ERROR_TYPE_SERVER);
        }
        return make_res({{"fingerprint", this->fingerprint}, {"files", files}});
    };

    post_evict = [this](const server_http_req & req) {
        router_seen();
        const json body = json::parse(req.body);
        if (!body.is_object() || !body.contains("files") || !body.at("files").is_array()) {
            return make_err("\"files\" must be an array", ERROR_TYPE_INVALID_REQUEST);
        }
        auto get_name = [](const json & file, const char * key) {
            if (!file.is_object() || !file.contains(key) || !file.at(key).is_string()) {
                return std::string();
            }
            return file.at(key).get<std::string>();
        };
        // validate all entries first, so that a bad request removes nothing
        std::vector<fs::path> paths;
        for (const auto & file : body.at("files")) {
            const std::string fp  = get_name(file, "fingerprint");
            const std::string key = get_name(file, "key");
            if (!slot_store_valid_name(fp) || !slot_store_valid_name(key)) {
                return make_err("invalid file entry: " + file.dump(), ERROR_TYPE_INVALID_REQUEST);
            }
            paths.push_back(fs::u8path(this->dir) / fp / (key + SLOT_EXT));
        }
        int n_removed = 0;
        int n_failed  = 0;
        std::string first_err;
        for (const auto & path : paths) {
            std::error_code ec;
            if (fs::remove(path, ec)) {
                n_removed++;
            } else if (ec && !is_vanished(ec)) {
                if (n_failed++ == 0) {
                    first_err = "cannot remove '" + path.string() + "': " + ec.message();
                }
            }
        }
        if (n_failed > 0) {
            return make_err(string_format("removed %d, failed %d, first error: %s", n_removed, n_failed, first_err.c_str()), ERROR_TYPE_SERVER);
        }
        return make_res({{"removed", n_removed}});
    };

    // note: not router contact, and not counted in flight (see server-http.cpp), so that local polling neither hides a lost router nor resets the drain idle time
    get_drained = [this](const server_http_req &) {
        const int64_t t = t_router_seen_ms.load();
        return make_res({
            {"drained",             drained.load()},
            {"router_seen_seconds", t < 0 ? (int64_t) -1 : (ggml_time_ms() - t) / 1000},
        });
    };

    post_drained = [this](const server_http_req &) {
        router_seen();
        if (!drained.exchange(true)) {
            SRV_INF("%s", "slot store: the router set the drained flag\n");
        }
        return make_res({{"drained", true}});
    };

    janitor = std::thread([this]() {
        std::unique_lock<std::mutex> lock(janitor_mutex);
        while (!janitor_stop) {
            lock.unlock();
            remove_old_tmp();
            lock.lock();
            janitor_cv.wait_for(lock, std::chrono::seconds(SLOT_STORE_JANITOR_SECONDS), [this] { return janitor_stop; });
        }
    });
}

server_slot_store::~server_slot_store() {
    {
        std::lock_guard<std::mutex> lock(janitor_mutex);
        janitor_stop = true;
    }
    janitor_cv.notify_all();
    if (janitor.joinable()) {
        janitor.join();
    }
}

std::string server_slot_store::save_path() const {
    return (fs::u8path(dir) / fingerprint).string() + DIRECTORY_SEPARATOR;
}

bool server_slot_store::router_lost(int64_t lost_ms) const {
    const int64_t t = t_router_seen_ms.load();
    return t < 0 || ggml_time_ms() - t > lost_ms;
}

void server_slot_store::router_seen() {
    t_router_seen_ms.store(ggml_time_ms());
}

// removes <dir>/*/*.tmp older than one hour, left by servers that died during a save
void server_slot_store::remove_old_tmp() const {
    const auto now = fs::file_time_type::clock::now();
    std::error_code ec;
    for (fs::directory_iterator it(fs::u8path(dir), ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code ec_fp;
        for (fs::directory_iterator jt(it->path(), ec_fp), end_fp; !ec_fp && jt != end_fp; jt.increment(ec_fp)) {
            if (jt->path().extension() != ".tmp") {
                continue;
            }
            std::error_code ec_file;
            const auto mtime = fs::last_write_time(jt->path(), ec_file);
            if (!ec_file && now - mtime > SLOT_STORE_TMP_MAX_AGE && fs::remove(jt->path(), ec_file)) {
                SRV_INF("slot store: removed stale '%s'\n", jt->path().string().c_str());
            }
        }
    }
    if (ec) {
        SRV_WRN("slot store: cannot list '%s': %s\n", dir.c_str(), ec.message().c_str());
    }
}
