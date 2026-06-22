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

if (GGML_OPENMP_ENABLED)
    target_compile_definitions(ggml-ork PRIVATE GGML_USE_OPENMP)
    target_link_libraries     (ggml-ork PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
endif()

# ork-driver's npu.c is C and needs _GNU_SOURCE (set in the file); keep its C sources as C.
set_source_files_properties(
    ${ORK_DIR}/src/npu.c ${ORK_DIR}/src/soc.c ${ORK_DIR}/src/soc/rk3588.c ${ORK_DIR}/src/soc/rk3576.c
    PROPERTIES LANGUAGE C)
