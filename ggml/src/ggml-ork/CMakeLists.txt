# ork-driver NPU matmul backend. ork-driver is vendored as a git submodule (ork-driver/) and its
# C sources are compiled directly into the backend library (no external .so dependency — it talks
# to the in-tree rknpu DRM driver via ioctls). ARM64 Linux only.

set(ORK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ork-driver)
# ork-driver's npu.c was split by precision into src/npu/ (its MODULARIZE_PLAN.md): a thin scaffold plus
# src/npu/{core,i8,f16,i4,i16}/ and src/npu/{sdp,ssm,norm}.c. Glob the modules so a new one does not need a
# CMake edit in lockstep; CONFIGURE_DEPENDS re-globs every build rather than only at configure time.
file(GLOB ORK_NPU_MODULES CONFIGURE_DEPENDS ${ORK_DIR}/src/npu/*.c ${ORK_DIR}/src/npu/*/*.c)
# The daemon and its client shim were split the same way, so glob both — a further split needs no edit here.
#   ORK_ORKD_CLIENT  the CLIENT-side RPC shim (round 9: orkd_client.c -> + orkd_client_ops.c behind
#                    orkd_client_internal.h). Linked into BOTH the backend library and the daemon.
#                    Missing a TU here does NOT fail the build — a shared library happily keeps undefined
#                    symbols and defers them to load time — it strands them (31 undefined orkd_* in
#                    libggml-ork.so when orkd_client_ops.c was left out) and breaks ORK_USE_ORKD at RUNTIME.
#   ORK_ORKD_MODULES the daemon proper (round 8: orkd.c -> + orkd_handlers.c behind orkd_internal.h),
#                    i.e. src/orkd*.c minus the client shim.
file(GLOB ORK_ORKD_CLIENT  CONFIGURE_DEPENDS ${ORK_DIR}/src/orkd_client*.c)
file(GLOB ORK_ORKD_MODULES CONFIGURE_DEPENDS ${ORK_DIR}/src/orkd*.c)
list(REMOVE_ITEM ORK_ORKD_MODULES ${ORK_ORKD_CLIENT})

if (NOT EXISTS ${ORK_DIR}/src/npu.c)
    message(FATAL_ERROR "ork-driver submodule missing — run: git submodule update --init --recursive")
endif()
# A submodule checkout from before the split (or a partially-synced one) still has src/npu.c, so the check
# above passes while the glob comes back short — which links a scaffold with ~93 undefined references. Fail
# at configure time instead, where the message can say what is actually wrong.
if (NOT EXISTS ${ORK_DIR}/src/npu/internal.h OR NOT ORK_NPU_MODULES)
    message(FATAL_ERROR "ork-driver submodule is stale or partially synced: src/npu/ split modules missing "
                        "(found ${ORK_DIR}/src/npu.c but no src/npu/internal.h + modules). "
                        "Run: git submodule update --init --recursive")
endif()

find_package(Threads REQUIRED)

ggml_add_backend_library(ggml-ork
                         ggml-ork.cpp
                         ${ORK_DIR}/src/npu.c
                         ${ORK_NPU_MODULES}
                         ${ORK_DIR}/src/ork_ops.c
                         ${ORK_DIR}/src/ork_gptq.c
                         ${ORK_ORKD_CLIENT}
                         ${ORK_DIR}/src/soc.c
                         ${ORK_DIR}/src/soc/rk3588.c
                         ${ORK_DIR}/src/soc/rk3576.c
                        )

target_include_directories(ggml-ork PRIVATE ${ORK_DIR}/include ${ORK_DIR}/src)
target_link_libraries     (ggml-ork PRIVATE Threads::Threads)

# ork-driver's npu.c is C and needs _GNU_SOURCE (set in the file); keep its C sources as C.
# orkd_client.c (v1.0.0) is the transparent orkd-daemon routing shim npu.c calls into under ORK_USE_ORKD.
set_source_files_properties(
    ${ORK_DIR}/src/npu.c ${ORK_NPU_MODULES} ${ORK_DIR}/src/ork_ops.c ${ORK_DIR}/src/ork_gptq.c ${ORK_ORKD_CLIENT} ${ORK_DIR}/src/soc.c ${ORK_DIR}/src/soc/rk3588.c ${ORK_DIR}/src/soc/rk3576.c
    PROPERTIES LANGUAGE C)

# Activation int8-quant in ggml-ork.cpp is parallelized with #pragma omp parallel for (per-row,
# independent). The pragmas are no-ops unless OpenMP is linked into THIS target (ggml-base linking it
# does not propagate here). Link it so prefill activation-quant runs multi-threaded.
# NOTE both languages. OpenMP::OpenMP_CXX carries -fopenmp guarded by $<COMPILE_LANGUAGE:CXX>, so linking
# only it leaves ork-driver's C sources (npu/, ork_gptq.c, the offline CPU GEMM) compiling WITHOUT the flag
# — every #pragma omp in them is then a silent no-op. That was measured: an offline scoring run pinned at
# 145% CPU on a 16-core host because orki_cpu_gemm_i32's per-row parallel for never took effect.
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(ggml-ork PRIVATE OpenMP::OpenMP_CXX)
endif()
if (OpenMP_C_FOUND)
    target_link_libraries(ggml-ork PRIVATE OpenMP::OpenMP_C)
endif()

# ork_bench / ork_ppl — open-stack perf + quality harnesses that drive the llama.cpp C API against the
# ggml-ork backend (tools/re in the vendored ork-driver). Built here so they ALWAYS compile alongside the
# backend (ccache keeps rebuilds cheap) instead of an out-of-tree g++. Board-only; guarded on the source.
# `llama`/`ggml` are forward target references (defined later in the project) — CMake resolves at generate.
# fa_diff_probe / bmm_diff_probe: differential CORRECTNESS oracles — build ONE attention node, compute it
# on the ggml CPU backend (oracle) and on the ORK backend, and diff. ~20s answers vs 6-min model loads;
# they call the ork backend directly so supports_op cannot silently decline the path under test.
foreach(orktool ork_bench ork_ppl ork_qerr ork_calibrate fa_diff_probe bmm_diff_probe)
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
    # TWO globs are load-bearing here, for the same reason: orkd is a separate binary with its own ork_npu
    # ctx, not a consumer of ggml-ork, so it cannot inherit anything from that target.
    #   ${ORK_NPU_MODULES} — it links the same NPU core, and after the precision split src/npu.c alone left
    #                        it with 93 undefined references (ork_mm_run_i8, ork_npu_enter, ork_dyn_*, ...).
    #   ${ORK_ORKD_MODULES} — round 8 split the daemon itself; src/orkd.c alone left it with 19 (the
    #                        orkd_handle_* dispatch table, now in orkd_handlers.c).
    # Both were by-name lists that broke silently one refactor round apart. Globs, not names.
    add_executable(orkd
                   ${ORK_ORKD_MODULES}
                   ${ORK_DIR}/src/npu.c
                   ${ORK_NPU_MODULES}
                   ${ORK_DIR}/src/ork_ops.c
                   ${ORK_ORKD_CLIENT}
                   ${ORK_DIR}/src/neon_activations.c
                   ${ORK_DIR}/src/soc.c
                   ${ORK_DIR}/src/soc/rk3588.c
                   ${ORK_DIR}/src/soc/rk3576.c)
    set_source_files_properties(
        ${ORK_ORKD_MODULES} ${ORK_DIR}/src/npu.c ${ORK_NPU_MODULES} ${ORK_ORKD_CLIENT}
        ${ORK_DIR}/src/neon_activations.c ${ORK_DIR}/src/soc.c
        ${ORK_DIR}/src/soc/rk3588.c ${ORK_DIR}/src/soc/rk3576.c
        PROPERTIES LANGUAGE C)
    target_include_directories(orkd PRIVATE ${ORK_DIR}/include ${ORK_DIR}/src)
    target_link_libraries(orkd PRIVATE Threads::Threads m)
    set_target_properties(orkd PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
