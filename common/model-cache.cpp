#include "model-cache.h"

#include "build-info.h"
#include "download.h"
#include "log.h"

#include <stdexcept>

std::string common_model_cache_path(const std::string & dir, const std::string & sha256) {
    return dir + "/" + sha256 + ".gguf";
}

#ifdef _WIN32

std::vector<std::string> common_model_cache_get(const std::string &, const std::vector<common_model_cache_file> &, int64_t, const common_download_opts &) {
    throw std::runtime_error("--model-cache is not supported on Windows");
}

#else

#include "http.h"

extern "C" {
#include "hash/sha256/sha256.h"
}
#include "blake3/blake3.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <mutex>
#include <random>
#include <set>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

// a waiter takes over the lock when the owner and its heartbeat have not changed for STALE_SECONDS
static constexpr int     STALE_SECONDS     = 120;
static constexpr int     BEAT_SECONDS      = 10;
static constexpr int     POLL_SECONDS      = 5;
static constexpr int     LOW_SPEED_SECONDS = 60;
static constexpr int64_t LOW_SPEED_BYTES   = 100 * 1024; // per second
static constexpr int     MAX_ATTEMPTS      = 3;
static constexpr int     HASH_ATTEMPTS     = 3;
static constexpr int64_t LEFTOVER_MAX_AGE  = 3600;
static constexpr int64_t LOAD_MAX_AGE      = 6 * 3600;
static constexpr int     TOUCH_SECONDS     = 24 * 3600;

static bool is_hex64(const std::string & s, size_t pos) {
    if (s.size() < pos + 64) {
        return false;
    }
    for (size_t i = pos; i < pos + 64; i++) {
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f'))) {
            return false;
        }
    }
    return true;
}

static std::string to_hex(const unsigned char * digest, size_t len) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    for (size_t i = 0; i < len; i++) {
        out += hex[digest[i] >> 4];
        out += hex[digest[i] & 0xf];
    }
    return out;
}

static std::string to_hex(sha256_t & ctx) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&ctx, digest);
    return to_hex(digest, sizeof(digest));
}

static std::string to_hex(const blake3_hasher & ctx) {
    unsigned char digest[BLAKE3_OUT_LEN];
    blake3_hasher_finalize(&ctx, digest, sizeof(digest));
    return to_hex(digest, sizeof(digest));
}

// <dir>/<sha256>.gguf.blake3: the blake3 of a file that hashed right with sha256, checked instead of the slower sha256
static std::string sidecar_path(const std::string & final_path) {
    return final_path + ".blake3";
}

// the creation time in the name of a private link .<sha>.<token>.<time>.load, or -1
static int64_t load_link_time(const std::string & name) {
    if (name.size() < 7 || name.compare(name.size() - 5, 5, ".load") != 0) {
        return -1;
    }
    const size_t end = name.size() - 5;
    const size_t begin = name.rfind('.', end - 1) + 1;
    if (begin == 0 || begin == end || name.find_first_not_of("0123456789", begin) != end) {
        return -1;
    }
    return std::stoll(name.substr(begin, end - begin));
}

// errors of link() that mean the file system has no hard links
static bool link_unsupported(int err) {
    return err == EPERM || err == EXDEV || err == EMLINK || err == ENOTSUP
#if EOPNOTSUPP != ENOTSUP
        || err == EOPNOTSUPP
#endif
        ;
}

static double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

static bool path_exists(const std::string & path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

// returns false when the file cannot be read, e.g. it does not exist
static bool read_text(const std::string & path, std::string & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    char buf[1024];
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    out.assign(buf, n);
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) {
        out.pop_back();
    }
    return true;
}

static bool write_text(const std::string & path, const std::string & text) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    bool ok = fwrite(text.data(), 1, text.size(), f) == text.size();
    return fclose(f) == 0 && ok;
}

static bool write_all(int fd, const char * data, size_t len) {
    while (len > 0) {
        ssize_t n = write(fd, data, len);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        data += n;
        len  -= n;
    }
    return true;
}

static void touch(const std::string & path) {
    utimensat(AT_FDCWD, path.c_str(), nullptr, 0);
}

// the URL without credentials and query string, which can hold a token
static std::string url_for_log(const std::string & url) {
    try {
        const auto parts = common_http_parse_url(url);
        return parts.scheme + "://" + common_http_format_host(parts.host) + ":" + std::to_string(parts.port) +
               parts.path.substr(0, parts.path.find_first_of("?#"));
    } catch (const std::exception &) {
        return "(invalid URL)";
    }
}

static time_t newest_mtime(const std::string & path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return 0;
    }
    time_t t = st.st_mtime;
    if (S_ISDIR(st.st_mode)) {
        std::error_code ec;
        for (const auto & e : fs::directory_iterator(path, ec)) {
            if (stat(e.path().c_str(), &st) == 0) {
                t = std::max(t, st.st_mtime);
            }
        }
    }
    return t;
}

// removes <sha>.gguf files not touched for ttl_s seconds, except keep, with their <sha>.gguf.blake3, sidecars
// without their model, and leftovers .<sha>.* older than an hour
static void evict(const std::string & dir, const std::set<std::string> & keep, int64_t ttl_s) {
    const time_t now = time(nullptr);
    std::error_code ec;
    try {
        for (const auto & e : fs::directory_iterator(dir, ec)) {
            const std::string name = e.path().filename().string();
            const std::string path = e.path().string();
            if (name.size() > 66 && name[0] == '.' && is_hex64(name, 1) && name[65] == '.') {
                // a temporary or moved-aside file, a lock whose heartbeat stopped, or a private link of a server that
                // died while loading; a link shares the times of the model, so its age is in its name
                const int64_t t_link = load_link_time(name);
                if (t_link >= 0 ? now - t_link > LOAD_MAX_AGE : now - newest_mtime(path) > LEFTOVER_MAX_AGE) {
                    LOG_INF("model cache: removing leftover %s\n", path.c_str());
                    fs::remove_all(path, ec);
                    if (ec) {
                        LOG_WRN("model cache: cannot remove %s: %s\n", path.c_str(), ec.message().c_str());
                    }
                }
            } else if (name.size() == 69 && is_hex64(name, 0) && name.compare(64, 5, ".gguf") == 0 && keep.count(name) == 0) {
                // stat right before unlink: another process may have touched the file since the listing
                struct stat st;
                if (stat(path.c_str(), &st) == 0 && now - st.st_mtime > ttl_s) {
                    LOG_INF("model cache: removing %s, not used for %lld s\n", path.c_str(), (long long) (now - st.st_mtime));
                    if (unlink(path.c_str()) != 0 && errno != ENOENT) {
                        LOG_WRN("model cache: cannot remove %s: %s\n", path.c_str(), strerror(errno));
                    } else {
                        unlink(sidecar_path(path).c_str());
                    }
                }
            } else if (name.size() == 76 && is_hex64(name, 0) && name.compare(64, 12, ".gguf.blake3") == 0 &&
                       keep.count(name.substr(0, 69)) == 0) {
                // only a model that is surely gone takes its sidecar along, an I/O error leaves both alone
                const std::string model = path.substr(0, path.size() - 7);
                struct stat st;
                if (stat(model.c_str(), &st) == 0) {
                    continue;
                }
                if (errno != ENOENT) {
                    LOG_WRN("model cache: cannot check %s: %s, keeping %s\n", model.c_str(), strerror(errno), path.c_str());
                    continue;
                }
                LOG_INF("model cache: removing %s without its model\n", path.c_str());
                if (unlink(path.c_str()) != 0 && errno != ENOENT) {
                    LOG_WRN("model cache: cannot remove %s: %s\n", path.c_str(), strerror(errno));
                }
            }
        }
    } catch (const std::exception & e) {
        LOG_WRN("model cache: cannot clean up %s: %s\n", dir.c_str(), e.what());
    }
}

namespace {

// puts one file into the cache, a port of the fetch-model.sh init script
//
// the directory may be on NFS mounted soft: an error from a write may mean it happened anyway, and file
// times seen from another host may be stale. correctness rests on the hash and on links that never
// replace: a file gets the final name only after its download hashed right, and a final file is deleted
// only after the copy moved aside hashed wrong. the lock only saves duplicate downloads
struct fetcher {
    std::string dir;
    std::string url;
    std::string sha;
    std::string final_path;
    std::string sidecar;
    std::string lock;
    std::string me;
    std::string token;
    std::string tmp;
    std::string tmp_blake3; // of the verified download in tmp
    httplib::Headers headers;
    int asides = 0;

    // the file last hashed, the download, and the final file found good
    struct stat st_hashed = {};
    struct stat st_tmp    = {};
    struct stat st_good   = {};

    bool sidecar_unreadable = false;

    // bytes hashed and downloaded, written to the heartbeat
    std::atomic<uint64_t> progress{0};

    fetcher(const std::string & dir, const common_model_cache_file & file, const common_download_opts & opts)
        : dir(dir), url(file.url), sha(file.sha256) {
        final_path = common_model_cache_path(dir, sha);
        sidecar = sidecar_path(final_path);
        lock = dir + "/." + sha + ".lock";

        char host[256] = "host";
        gethostname(host, sizeof(host) - 1);
        me = std::string(host) + "." + std::to_string(getpid());
        char rnd[9];
        snprintf(rnd, sizeof(rnd), "%08x", (unsigned) std::random_device{}());
        token = me + "." + rnd;
        tmp = dir + "/." + sha + "." + token + ".tmp";

        for (const auto & h : opts.headers) {
            headers.emplace(h.first, h.second);
        }
        if (headers.find("User-Agent") == headers.end()) {
            headers.emplace("User-Agent", "llama-cpp/" + std::string(llama_build_info()));
        }
        if (!opts.bearer_token.empty()) {
            headers.emplace("Authorization", "Bearer " + opts.bearer_token);
        }
    }

    // computes the blake3 of the file, and its sha256 unless sha_out is null, in one pass
    // returns 0, or the errno of the failed open or read
    int hash_once(const std::string & path, std::string * sha_out, std::string & blake3_out) {
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) {
            return errno;
        }
        if (fstat(fileno(f), &st_hashed) != 0) {
            const int err = errno;
            fclose(f);
            return err;
        }
        sha256_t ctx;
        sha256_init(&ctx);
        blake3_hasher ctx_b3;
        blake3_hasher_init(&ctx_b3);
        std::vector<unsigned char> buf(1 << 20);
        size_t n;
        errno = 0;
        while ((n = fread(buf.data(), 1, buf.size(), f)) > 0) {
            if (sha_out) {
                sha256_update(&ctx, buf.data(), n);
            }
            blake3_hasher_update(&ctx_b3, buf.data(), n);
            progress += n;
        }
        const int err = ferror(f) ? (errno ? errno : EIO) : 0;
        fclose(f);
        if (err == 0) {
            if (sha_out) {
                *sha_out = to_hex(ctx);
            }
            blake3_out = to_hex(ctx_b3);
        }
        return err;
    }

    // like hash_once, but tries again after a transient read error
    int hash_file(const std::string & path, std::string * sha_out, std::string & blake3_out) {
        int err = 0;
        for (int attempt = 1; attempt <= HASH_ATTEMPTS; attempt++) {
            err = hash_once(path, sha_out, blake3_out);
            if ((err != EIO && err != ESTALE) || attempt == HASH_ATTEMPTS) {
                break;
            }
            LOG_WRN("model cache: cannot read %s: %s, trying again\n", path.c_str(), strerror(err));
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        return err;
    }

    // publishes the sidecar of a file that hashed right with sha256; a failure only costs a sha256 check later
    void write_sidecar(const std::string & blake3) {
        if (sidecar_unreadable) {
            return;
        }
        const std::string path_tmp = dir + "/." + sha + "." + token + ".blake3.tmp";
        const std::string line = blake3 + "\n";
        int fd = open(path_tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        bool ok = fd >= 0 && write_all(fd, line.data(), line.size()) && fsync(fd) == 0;
        ok = fd >= 0 && close(fd) == 0 && ok;
        ok = ok && rename(path_tmp.c_str(), sidecar.c_str()) == 0;
        if (!ok) {
            LOG_WRN("model cache: cannot write %s: %s\n", sidecar.c_str(), strerror(errno));
            unlink(path_tmp.c_str());
        }
    }

    // true: the final file matches its sidecar; false: it does not, or there is no valid sidecar
    // returns the errno of a failed read of the final file in err
    bool check_sidecar(int & err) {
        err = 0;
        std::string want;
        if (!read_text(sidecar, want)) {
            // without the sidecar the sha256 decides; a sidecar that cannot be read is left alone
            if (errno != ENOENT) {
                sidecar_unreadable = true;
                LOG_WRN("model cache: cannot read %s: %s, checking sha256\n", sidecar.c_str(), strerror(errno));
            }
            return false;
        }
        if (want.size() != 64 || !is_hex64(want, 0)) {
            LOG_WRN("model cache: %s is not a blake3 hash, checking sha256\n", sidecar.c_str());
            return false;
        }
        LOG_INF("model cache: verifying %s with blake3\n", final_path.c_str());
        const auto t_start = std::chrono::steady_clock::now();
        std::string got;
        err = hash_file(final_path, nullptr, got);
        if (err != 0) {
            return false;
        }
        if (got != want) {
            LOG_WRN("model cache: %s has blake3 %s, %s has %s, checking sha256\n", final_path.c_str(), got.c_str(), sidecar.c_str(), want.c_str());
            return false;
        }
        LOG_INF("model cache: verified %s with blake3 in %.2f s\n", final_path.c_str(), seconds_since(t_start));
        return true;
    }

    // true: the final file is good, false: there is no good final file; throws when it cannot be read
    // the blake3 sidecar is checked first; without it, or on a mismatch, the sha256 decides and the sidecar is
    // written again. a mismatching final file is renamed aside (only one process's rename succeeds) and hashed
    // again there: another process may have replaced it with a good file in between, which is then linked back.
    // only an aside file that hashed wrong is deleted
    bool check_final() {
        for (int round = 0; path_exists(final_path); round++) {
            if (round == 3) {
                throw std::runtime_error("cannot settle " + final_path);
            }
            touch(final_path);
            int err = 0;
            if (check_sidecar(err)) {
                st_good = st_hashed;
                return true;
            }
            std::string got;
            std::string got_blake3;
            const auto t_start = std::chrono::steady_clock::now();
            if (err == 0) {
                LOG_INF("model cache: verifying %s with sha256\n", final_path.c_str());
                err = hash_file(final_path, &got, got_blake3);
            }
            if (err == ENOENT || err == ESTALE) {
                // another process moved it aside meanwhile
                continue;
            }
            if (err != 0) {
                throw std::runtime_error("cannot read " + final_path + ": " + strerror(err));
            }
            if (got == sha) {
                LOG_INF("model cache: verified %s with sha256 in %.2f s\n", final_path.c_str(), seconds_since(t_start));
                write_sidecar(got_blake3);
                st_good = st_hashed;
                return true;
            }
            const std::string aside = dir + "/." + sha + ".bad." + token + "." + std::to_string(++asides);
            LOG_WRN("model cache: mismatch: %s has sha256 %s, moving it aside to %s\n", final_path.c_str(), got.c_str(), aside.c_str());
            rename(final_path.c_str(), aside.c_str());
            if (!path_exists(aside)) {
                continue;
            }
            if (const int err = hash_file(aside, &got, got_blake3); err != 0) {
                throw std::runtime_error("cannot read " + aside + ": " + strerror(err) + ", leaving it there");
            }
            if (got != sha) {
                LOG_INF("model cache: removing bad %s\n", aside.c_str());
                unlink(aside.c_str());
                continue;
            }
            LOG_INF("model cache: %s was replaced with a good file meanwhile, putting it back\n", final_path.c_str());
            if (link(aside.c_str(), final_path.c_str()) == 0 || (hash_file(final_path, &got, got_blake3) == 0 && got == sha)) {
                unlink(aside.c_str());
                LOG_INF("model cache: verified %s\n", final_path.c_str());
                write_sidecar(got_blake3);
                st_good = st_hashed;
                return true;
            }
            LOG_WRN("model cache: cannot put %s back as %s, leaving it there\n", aside.c_str(), final_path.c_str());
            return false;
        }
        return false;
    }

    // the lock is held by whoever creates lock/owner, O_EXCL makes that exclusive also on NFS
    // returns 0 when taken, EEXIST when another process holds it, otherwise the errno of the failure
    int take_lock() {
        const int err_dir = mkdir(lock.c_str(), 0777) == 0 ? 0 : errno;
        int fd = open((lock + "/owner").c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
        if (fd < 0) {
            // without the lock directory, the reason it could not be made says more
            return errno == ENOENT && err_dir != 0 && err_dir != EEXIST ? err_dir : errno;
        }
        const std::string line = token + "\n";
        errno = 0;
        const bool ok = write_all(fd, line.data(), line.size());
        const int err_write = ok ? 0 : (errno ? errno : EIO);
        const int err_close = close(fd) == 0 ? 0 : errno;
        if (err_write != 0 || err_close != 0) {
            unlink((lock + "/owner").c_str());
            return err_write != 0 ? err_write : err_close;
        }
        return 0;
    }

    bool holds_lock() {
        std::string owner;
        return read_text(lock + "/owner", owner) && owner == token;
    }

    // runs while holding the lock; returns when the final file is good, throws otherwise
    void own() {
        std::mutex mtx;
        std::condition_variable cv;
        bool stop = false;

        // heartbeat: the bytes hashed and downloaded so far, so a stuck transfer looks stale
        std::thread beat([&] {
            std::unique_lock<std::mutex> lk(mtx);
            while (!stop && holds_lock()) {
                if (write_text(lock + "/beat.tmp", std::to_string(progress.load()) + "\n")) {
                    rename((lock + "/beat.tmp").c_str(), (lock + "/beat").c_str());
                }
                cv.wait_for(lk, std::chrono::seconds(BEAT_SECONDS), [&] { return stop; });
            }
        });

        struct cleanup {
            std::function<void()> fn;
            ~cleanup() { fn(); }
        } on_exit{[&] {
            {
                std::lock_guard<std::mutex> lk(mtx);
                stop = true;
            }
            cv.notify_all();
            beat.join();
            unlink(tmp.c_str());
            // release: the owner file goes last, a new owner may create it right after
            if (holds_lock()) {
                unlink((lock + "/beat").c_str());
                unlink((lock + "/beat.tmp").c_str());
                unlink((lock + "/owner").c_str());
                rmdir(lock.c_str());
            }
        }};

        // published while the lock was being taken?
        if (check_final()) {
            return;
        }
        const uint64_t size = download();
        if (!publish(size)) {
            throw std::runtime_error("cannot publish " + final_path);
        }
    }

    // downloads url to tmp and checks its hash; returns its size, throws on error
    uint64_t download() {
        LOG_INF("model cache: downloading %s to %s\n", url_for_log(url).c_str(), tmp.c_str());
        int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            throw std::runtime_error("cannot create " + tmp + ": " + strerror(errno));
        }
        struct fd_closer {
            int & fd;
            ~fd_closer() { if (fd >= 0) { close(fd); } }
        } closer{fd};
        if (fstat(fd, &st_tmp) != 0) {
            throw std::runtime_error("cannot stat " + tmp + ": " + strerror(errno));
        }

        auto [cli, parts] = common_http_client(url);
        cli.set_default_headers(headers);
        cli.set_connection_timeout(30);
        cli.set_read_timeout(LOW_SPEED_SECONDS);

        sha256_t ctx;
        blake3_hasher ctx_b3;
        uint64_t n = 0;
        for (int attempt = 1; ; attempt++) {
            // no resume: every attempt starts from the beginning
            if (ftruncate(fd, 0) != 0 || lseek(fd, 0, SEEK_SET) != 0) {
                throw std::runtime_error("cannot truncate " + tmp + ": " + strerror(errno));
            }
            sha256_init(&ctx);
            blake3_hasher_init(&ctx_b3);
            n = 0;
            int status = -1;
            bool write_failed = false;
            bool too_slow = false;
            auto t_window = std::chrono::steady_clock::now();
            uint64_t n_window = 0;

            auto res = cli.Get(parts.path, httplib::Headers(),
                [&](const httplib::Response & r) {
                    status = r.status;
                    return r.status == 200;
                },
                [&](const char * data, size_t len) {
                    if (!write_all(fd, data, len)) {
                        write_failed = true;
                        return false;
                    }
                    sha256_update(&ctx, (const unsigned char *) data, len);
                    blake3_hasher_update(&ctx_b3, data, len);
                    n += len;
                    n_window += len;
                    progress += len;
                    const auto now = std::chrono::steady_clock::now();
                    if (now - t_window >= std::chrono::seconds(LOW_SPEED_SECONDS)) {
                        if (n_window < (uint64_t) (LOW_SPEED_BYTES * LOW_SPEED_SECONDS)) {
                            too_slow = true;
                            return false;
                        }
                        t_window = now;
                        n_window = 0;
                    }
                    return true;
                },
                nullptr);

            if (res && status == 200) {
                break;
            }
            if (write_failed) {
                throw std::runtime_error("cannot write " + tmp + ": " + strerror(errno));
            }
            std::string err = too_slow ? "slower than 100 KiB/s for 60 s" :
                              status > 0 && status != 200 ? "HTTP status " + std::to_string(status) :
                              httplib::to_string(res.error());
            // a client error such as 404 is not retried
            if (attempt == MAX_ATTEMPTS || (status >= 400 && status < 500)) {
                throw std::runtime_error("download of " + url_for_log(url) + " failed: " + err);
            }
            LOG_WRN("model cache: download failed: %s, retrying in %d s\n", err.c_str(), 2 * attempt);
            std::this_thread::sleep_for(std::chrono::seconds(2 * attempt));
        }

        if (fsync(fd) != 0) {
            throw std::runtime_error("cannot sync " + tmp + ": " + strerror(errno));
        }
        const int rc = close(fd);
        fd = -1;
        if (rc != 0) {
            throw std::runtime_error("cannot close " + tmp + ": " + strerror(errno));
        }
        const std::string got = to_hex(ctx);
        if (got != sha) {
            throw std::runtime_error("mismatch: " + url_for_log(url) + " has sha256 " + got + ", want " + sha);
        }
        tmp_blake3 = to_hex(ctx_b3);
        return n;
    }

    // links the verified tmp to the final name; true when the final file ends up good
    bool publish(uint64_t size) {
        for (int i = 0; i < 3; i++) {
            if (link(tmp.c_str(), final_path.c_str()) == 0) {
                struct stat st;
                if (stat(final_path.c_str(), &st) == 0 && (uint64_t) st.st_size == size) {
                    LOG_INF("model cache: downloaded %s\n", final_path.c_str());
                    write_sidecar(tmp_blake3);
                    st_good = st_tmp;
                    return true;
                }
            }
            // another file holds the name, or the link could not be confirmed
            if (check_final()) {
                return true;
            }
        }
        return false;
    }

    // returns when the final file may exist, the lock is gone, or it was taken over: 0, or the errno of
    // the failed takeover of a stale lock
    int wait_owner() {
        std::string owner;
        read_text(lock + "/owner", owner);
        LOG_INF("model cache: waiting for %s\n", owner.empty() ? "the lock owner" : owner.c_str());
        std::string last;
        auto since = std::chrono::steady_clock::now();
        for (bool first = true; ; first = false) {
            std::this_thread::sleep_for(std::chrono::seconds(POLL_SECONDS));
            if (path_exists(final_path)) {
                return 0;
            }
            // an empty owner file is a lock too: its owner died before it wrote the token
            if (!read_text(lock + "/owner", owner)) {
                LOG_INF("model cache: lock released without %s\n", final_path.c_str());
                return 0;
            }
            std::string beat;
            read_text(lock + "/beat", beat);
            const auto now = std::chrono::steady_clock::now();
            if (first || owner + " " + beat != last) {
                last  = owner + " " + beat;
                since = now;
                continue;
            }
            if (now - since <= std::chrono::seconds(STALE_SECONDS)) {
                continue;
            }
            const std::string stale = dir + "/." + sha + ".lock.stale." + me;
            std::error_code ec;
            fs::remove_all(stale, ec);
            // ENOENT: the lock was released or taken over by another process meanwhile
            const int err = rename(lock.c_str(), stale.c_str()) == 0 ? 0 : errno;
            if (err == 0) {
                std::string moved;
                read_text(stale + "/owner", moved);
                if (moved == owner) {
                    LOG_WRN("model cache: took over the stale lock of %s\n", owner.empty() ? "an unknown owner" : owner.c_str());
                } else {
                    // the stale lock was released and a new one taken meanwhile: put it back unless yet
                    // another lock holds the name (then the worst case is a duplicate download)
                    LOG_WRN("model cache: moved the fresh lock of %s, putting it back\n", moved.c_str());
                    if (!path_exists(lock)) {
                        rename(stale.c_str(), lock.c_str());
                    }
                }
            }
            fs::remove_all(stale, ec);
            return err == ENOENT ? 0 : err;
        }
    }

    // a transient error (EIO, ESTALE, as from NFS mounted soft) is tried again a few times, other errors fail
    void transient(int err, int & n_transient, const std::string & what) {
        if ((err != EIO && err != ESTALE) || ++n_transient == MAX_ATTEMPTS) {
            throw std::runtime_error(what + " in " + dir + ": " + strerror(err));
        }
        LOG_WRN("model cache: %s in %s: %s, trying again\n", what.c_str(), dir.c_str(), strerror(err));
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // runs, then links the final file to a name private to this process, so that the file loaded is the one verified
    // even if the final file is replaced meanwhile; returns the link, or an empty string when no link can be made
    std::string get() {
        // test hook: the first link fails with this error
        static const std::string test_link_error = [] {
            const char * val = getenv("LLAMA_MODEL_CACHE_TEST_LINK_ERROR");
            return std::string(val ? val : "");
        }();

        for (int round = 0; round < 3; round++) {
            run();
            // the time in the name is when the link is made, run() may download or wait for long
            const std::string path_link = dir + "/." + sha + "." + token + "." + std::to_string((long long) time(nullptr)) + ".load";
            unlink(path_link.c_str());
            int err = link(final_path.c_str(), path_link.c_str()) == 0 ? 0 : errno;
            if (round == 0 && !test_link_error.empty()) {
                unlink(path_link.c_str());
                err = test_link_error == "ENOENT" ? ENOENT : test_link_error == "EXDEV" ? EXDEV : EIO;
            }
            if (err == ENOENT) {
                // moved aside by another server since it was verified
                LOG_WRN("model cache: %s is gone, verifying again\n", final_path.c_str());
                continue;
            }
            if (link_unsupported(err)) {
                LOG_WRN("model cache: cannot link %s to %s: %s, loading it by its name\n", final_path.c_str(), path_link.c_str(), strerror(err));
                return "";
            }
            if (err != 0) {
                throw std::runtime_error("cannot link " + final_path + " to " + path_link + ": " + strerror(err));
            }
            struct stat st;
            if (stat(path_link.c_str(), &st) == 0 && st.st_dev == st_good.st_dev && st.st_ino == st_good.st_ino) {
                return path_link;
            }
            unlink(path_link.c_str());
            LOG_WRN("model cache: %s changed after it was verified, verifying again\n", final_path.c_str());
        }
        throw std::runtime_error("cannot settle " + final_path);
    }

    void run() {
        int tries = 0;
        int n_transient = 0;
        while (true) {
            if (check_final()) {
                return;
            }
            const int err = take_lock();
            if (err == 0) {
                own();
                return;
            }
            if (err != EEXIST) {
                transient(err, n_transient, "cannot create the lock " + lock + "/owner");
                continue;
            }
            if (!path_exists(lock + "/owner")) {
                // released just now
                if (++tries == 3) {
                    throw std::runtime_error("cannot create " + lock + "/owner");
                }
                continue;
            }
            tries = 0;
            if (const int err_wait = wait_owner(); err_wait != 0) {
                transient(err_wait, n_transient, "cannot take over the stale lock " + lock);
            }
        }
    }
};

} // namespace

std::vector<std::string> common_model_cache_get(const std::string & dir, const std::vector<common_model_cache_file> & files, int64_t ttl_s, const common_download_opts & opts) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (!fs::is_directory(dir)) {
        throw std::runtime_error("cannot create model cache directory " + dir);
    }

    std::set<std::string> keep;
    std::vector<std::string> paths;
    for (const auto & f : files) {
        keep.insert(f.sha256 + ".gguf");
        paths.push_back(common_model_cache_path(dir, f.sha256));
        // so that another process starting now does not evict it
        touch(paths.back());
    }
    evict(dir, keep, ttl_s);

    std::vector<std::string> links;
    for (const auto & f : files) {
        links.push_back(fetcher(dir, f, opts).get());
    }

    for (const auto & p : paths) {
        touch(p);
    }
    // touched well within the ttl, so that a server with a shorter ttl does not evict a file in use
    const int64_t touch_s = std::min<int64_t>(TOUCH_SECONDS, std::max<int64_t>(ttl_s / 4, 1));
    std::thread([paths, touch_s] {
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(touch_s));
            for (const auto & p : paths) {
                touch(p);
            }
        }
    }).detach();
    return links;
}

#endif
