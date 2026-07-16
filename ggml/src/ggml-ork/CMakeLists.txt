# ork-driver NPU matmul backend. ork-driver is vendored as a git submodule (ork-driver/) and its
# C sources are compiled directly into the backend library (no external .so dependency — it talks
# to the in-tree rknpu DRM driver via ioctls). ARM64 Linux only.

set(ORK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ork-driver)

if (NOT EXISTS ${ORK_DIR}/src/npu.c)
    message(FATAL_ERROR "ork-driver submodule missing — run: git submodule update --init --recursive")
endif()

find_package(Threads REQUIRED)

ggml_add_backend_library(ggml-ork
                         ggml-ork.cpp
                         ${ORK_DIR}/src/npu.c
                         ${ORK_DIR}/src/soc.c
                         ${ORK_DIR}/src/soc/rk3588.c
                         ${ORK_DIR}/src/soc/rk3576.c
                        )

target_include_directories(ggml-ork PRIVATE ${ORK_DIR}/include ${ORK_DIR}/src)
target_link_libraries     (ggml-ork PRIVATE Threads::Threads)

# ork-driver's npu.c is C and needs _GNU_SOURCE (set in the file); keep its C sources as C.
set_source_files_properties(
    ${ORK_DIR}/src/npu.c ${ORK_DIR}/src/soc.c ${ORK_DIR}/src/soc/rk3588.c ${ORK_DIR}/src/soc/rk3576.c
    PROPERTIES LANGUAGE C)

# Activation int8-quant in ggml-ork.cpp is parallelized with #pragma omp parallel for (per-row,
# independent). The pragmas are no-ops unless OpenMP is linked into THIS target (ggml-base linking it
# does not propagate here). Link it so prefill activation-quant runs multi-threaded.
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(ggml-ork PRIVATE OpenMP::OpenMP_CXX)
endif()

# ork_bench / ork_ppl — open-stack perf + quality harnesses that drive the llama.cpp C API against the
# ggml-ork backend (tools/re in the vendored ork-driver). Built here so they ALWAYS compile alongside the
# backend (ccache keeps rebuilds cheap) instead of an out-of-tree g++. Board-only; guarded on the source.
# `llama`/`ggml` are forward target references (defined later in the project) — CMake resolves at generate.
foreach(orktool ork_bench ork_ppl)
    if (EXISTS ${ORK_DIR}/tools/re/${orktool}.cpp)
        add_executable(${orktool} ${ORK_DIR}/tools/re/${orktool}.cpp)
        target_link_libraries(${orktool} PRIVATE llama ggml)
        target_include_directories(${orktool} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${ORK_DIR}/include)
        set_target_properties(${orktool} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endif()
endforeach()
