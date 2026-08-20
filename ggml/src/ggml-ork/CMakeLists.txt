# ork-driver NPU matmul backend. ork-driver is vendored as a git submodule (ork-driver/) and its
# C sources are compiled directly into the backend library (no external .so dependency — it talks
# to the in-tree rknpu DRM driver via ioctls). ARM64 Linux only.

set(ORK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ork-driver)
# ork-driver's npu.c is being split into src/npu/*.c by precision (see its MODULARIZE_PLAN.md). Glob the
# modules so each new one does not need a CMake edit in lockstep; CONFIGURE_DEPENDS re-globs every build.
file(GLOB ORK_NPU_MODULES CONFIGURE_DEPENDS ${ORK_DIR}/src/npu/*.c ${ORK_DIR}/src/npu/*/*.c)

if (NOT EXISTS ${ORK_DIR}/src/npu.c)
    message(FATAL_ERROR "ork-driver submodule missing — run: git submodule update --init --recursive")
endif()

find_package(Threads REQUIRED)

ggml_add_backend_library(ggml-ork
                         ggml-ork.cpp
                         ${ORK_DIR}/src/npu.c
                         ${ORK_NPU_MODULES}
                         ${ORK_DIR}/src/ork_ops.c
                         ${ORK_DIR}/src/orkd_client.c
                         ${ORK_DIR}/src/soc.c
                         ${ORK_DIR}/src/soc/rk3588.c
                         ${ORK_DIR}/src/soc/rk3576.c
                        )

target_include_directories(ggml-ork PRIVATE ${ORK_DIR}/include ${ORK_DIR}/src)
target_link_libraries     (ggml-ork PRIVATE Threads::Threads)

# ork-driver's npu.c is C and needs _GNU_SOURCE (set in the file); keep its C sources as C.
# orkd_client.c (v1.0.0) is the transparent orkd-daemon routing shim npu.c calls into under ORK_USE_ORKD.
set_source_files_properties(
    ${ORK_DIR}/src/npu.c ${ORK_DIR}/src/ork_ops.c ${ORK_DIR}/src/orkd_client.c ${ORK_DIR}/src/soc.c ${ORK_DIR}/src/soc/rk3588.c ${ORK_DIR}/src/soc/rk3576.c
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
# fa_diff_probe / bmm_diff_probe: differential CORRECTNESS oracles — build ONE attention node, compute it
# on the ggml CPU backend (oracle) and on the ORK backend, and diff. ~20s answers vs 6-min model loads;
# they call the ork backend directly so supports_op cannot silently decline the path under test.
foreach(orktool ork_bench ork_ppl fa_diff_probe bmm_diff_probe)
    if (EXISTS ${ORK_DIR}/tools/re/${orktool}.cpp)
        add_executable(${orktool} ${ORK_DIR}/tools/re/${orktool}.cpp)
        target_link_libraries(${orktool} PRIVATE llama ggml)
        target_include_directories(${orktool} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${ORK_DIR}/include)
        set_target_properties(${orktool} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endif()
endforeach()

# orkd — the NPU daemon (ork-driver >=1.0.0). It OWNS the single-stream NPU and serializes submits from
# every client process; the backend's orkd client (ORK_USE_ORKD=1) auto-spawns the binary named by ORKD_BIN.
# Building it here means a board build produces the daemon alongside the backend — otherwise ORK_USE_ORKD has
# nothing to connect to. It links the same core NPU C sources as the backend (own binary, its own ork_npu ctx).
# Board-only; guarded on the source (absent in <1.0.0 submodule checkouts). ORKD_BIN should point at bin/orkd.
if (EXISTS ${ORK_DIR}/src/orkd.c)
    add_executable(orkd
                   ${ORK_DIR}/src/orkd.c
                   ${ORK_DIR}/src/npu.c
                   ${ORK_DIR}/src/ork_ops.c
                   ${ORK_DIR}/src/orkd_client.c
                   ${ORK_DIR}/src/neon_activations.c
                   ${ORK_DIR}/src/soc.c
                   ${ORK_DIR}/src/soc/rk3588.c
                   ${ORK_DIR}/src/soc/rk3576.c)
    set_source_files_properties(
        ${ORK_DIR}/src/orkd.c ${ORK_DIR}/src/npu.c ${ORK_DIR}/src/orkd_client.c
        ${ORK_DIR}/src/neon_activations.c ${ORK_DIR}/src/soc.c
        ${ORK_DIR}/src/soc/rk3588.c ${ORK_DIR}/src/soc/rk3576.c
        PROPERTIES LANGUAGE C)
    target_include_directories(orkd PRIVATE ${ORK_DIR}/include ${ORK_DIR}/src)
    target_link_libraries(orkd PRIVATE Threads::Threads m)
    set_target_properties(orkd PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
