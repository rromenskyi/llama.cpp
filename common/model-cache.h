#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct common_download_opts;

// a model cache is a directory shared by processes on several hosts (e.g. over NFS)
// it holds each file as <sha256>.gguf, see --model-cache

struct common_model_cache_file {
    std::string url;
    std::string sha256;
};

// returns <dir>/<sha256>.gguf
std::string common_model_cache_path(const std::string & dir, const std::string & sha256);

// removes cache files not used for ttl_s seconds, then downloads the files that are not in the cache yet;
// keeps the files in use: touches them now and once a day
// returns for each file a hard link to the verified file, private to this process, to load from and remove after
// loading; an empty string when no link could be made
// throws on error
std::vector<std::string> common_model_cache_get(const std::string & dir, const std::vector<common_model_cache_file> & files, int64_t ttl_s, const common_download_opts & opts);
