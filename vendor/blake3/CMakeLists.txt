set(TARGET vendor-blake3)
license_add_file("BLAKE3" "LICENSE_CC0")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# single-threaded C API only (no TBB); SIMD code is picked at run time on x86
set(BLAKE3_SRCS
    blake3.c
    blake3_dispatch.c
    blake3_portable.c
)

set(BLAKE3_ARCH ${CMAKE_SYSTEM_PROCESSOR})
if (CMAKE_OSX_ARCHITECTURES)
    set(BLAKE3_ARCH "${CMAKE_OSX_ARCHITECTURES}")
endif()

# the SIMD files need per-file flags that match their run time dispatch: MSVC and multi-arch macOS builds
# use the portable code
if (NOT MSVC AND BLAKE3_ARCH MATCHES "^(x86_64|AMD64|amd64|i686|i386|x86|X86)$")
    list(APPEND BLAKE3_SRCS blake3_sse2.c blake3_sse41.c blake3_avx2.c blake3_avx512.c)
    set_source_files_properties(blake3_sse2.c   PROPERTIES COMPILE_FLAGS "-msse2")
    set_source_files_properties(blake3_sse41.c  PROPERTIES COMPILE_FLAGS "-msse4.1")
    set_source_files_properties(blake3_avx2.c   PROPERTIES COMPILE_FLAGS "-mavx2")
    set_source_files_properties(blake3_avx512.c PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512vl")
elseif (NOT MSVC AND BLAKE3_ARCH MATCHES "^(aarch64|arm64|ARM64)$")
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
