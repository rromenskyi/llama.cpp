// checks the vendored BLAKE3 against known digests of the input pattern i % 251 used by the official test vectors

#include "blake3/blake3.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::string hex(const uint8_t * d, size_t n) {
    std::string s;
    char buf[3];
    for (size_t i = 0; i < n; i++) {
        snprintf(buf, sizeof(buf), "%02x", d[i]);
        s += buf;
    }
    return s;
}

// hashes the input in pieces of `step` bytes, to cross chunk and SIMD batch boundaries
static std::string blake3_hex(const std::vector<uint8_t> & data, size_t step) {
    blake3_hasher h;
    blake3_hasher_init(&h);
    for (size_t i = 0; i < data.size(); i += step) {
        blake3_hasher_update(&h, data.data() + i, std::min(step, data.size() - i));
    }
    uint8_t out[BLAKE3_OUT_LEN];
    blake3_hasher_finalize(&h, out, sizeof(out));
    return hex(out, sizeof(out));
}

int main() {
    struct test_case {
        size_t      len;
        const char * want;
    };
    const test_case cases[] = {
        // from test_vectors/test_vectors.json of BLAKE3 1.8.7
        {0,       "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"},
        {1,       "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"},
        {1024,    "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"},
        {1025,    "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444"},
        {102400,  "bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085"},
        // from the blake3 Python package 1.0.11
        {1 << 20, "74cb441fd087764ca9c3694da742ebe30cbeb3060a17009ca81825c7a8d10343"},
    };

    int n_fail = 0;
    for (const auto & tc : cases) {
        std::vector<uint8_t> data(tc.len);
        for (size_t i = 0; i < tc.len; i++) {
            data[i] = (uint8_t) (i % 251);
        }
        for (size_t step : {(size_t) 1 << 30, (size_t) 1000, (size_t) 65536}) {
            const std::string got = blake3_hex(data, step);
            if (got != tc.want) {
                fprintf(stderr, "FAIL: len %zu step %zu: got %s, want %s\n", tc.len, step, got.c_str(), tc.want);
                n_fail++;
            }
        }
    }

    if (n_fail > 0) {
        return 1;
    }
    printf("OK: BLAKE3 %s\n", blake3_version());
    return 0;
}
