set(TARGET vendor-blake3)
license_add_file("BLAKE3" "LICENSE_CC0")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# single-threaded C API only (no TBB); SIMD code is picked at run time on x86
set(BLAKE3_SRCS
    blake3.c
    blake3_dispatch.c
    blake3_portable.c
)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i686|i386|x86|X86)$")
    if (MSVC AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
        set(BLAKE3_FLAGS_SSE2   "/arch:SSE2")
        set(BLAKE3_FLAGS_SSE41  "/arch:AVX")
        set(BLAKE3_FLAGS_AVX2   "/arch:AVX2")
        set(BLAKE3_FLAGS_AVX512 "/arch:AVX512")
    else()
        set(BLAKE3_FLAGS_SSE2   "-msse2")
        set(BLAKE3_FLAGS_SSE41  "-msse4.1")
        set(BLAKE3_FLAGS_AVX2   "-mavx2")
        set(BLAKE3_FLAGS_AVX512 "-mavx512f -mavx512vl")
    endif()
    list(APPEND BLAKE3_SRCS blake3_sse2.c blake3_sse41.c blake3_avx2.c blake3_avx512.c)
    set_source_files_properties(blake3_sse2.c   PROPERTIES COMPILE_FLAGS "${BLAKE3_FLAGS_SSE2}")
    set_source_files_properties(blake3_sse41.c  PROPERTIES COMPILE_FLAGS "${BLAKE3_FLAGS_SSE41}")
    set_source_files_properties(blake3_avx2.c   PROPERTIES COMPILE_FLAGS "${BLAKE3_FLAGS_AVX2}")
    set_source_files_properties(blake3_avx512.c PROPERTIES COMPILE_FLAGS "${BLAKE3_FLAGS_AVX512}")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    list(APPEND BLAKE3_SRCS blake3_neon.c)
else()
    set(BLAKE3_DEFS BLAKE3_USE_NEON=0 BLAKE3_NO_SSE2 BLAKE3_NO_SSE41 BLAKE3_NO_AVX2 BLAKE3_NO_AVX512)
endif()

add_library(${TARGET} STATIC ${BLAKE3_SRCS} blake3.h)
add_library(vendor::blake3 ALIAS ${TARGET})

if (BLAKE3_DEFS)
    target_compile_definitions(${TARGET} PRIVATE ${BLAKE3_DEFS})
endif()

# disable warnings in 3rd party code
if (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${TARGET} PRIVATE /w)
else()
    target_compile_options(${TARGET} PRIVATE -w)
endif()

# consumers include via "blake3/blake3.h"
target_include_directories(${TARGET} PUBLIC ..)
