#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// ork-driver NPU matmul backend (Rockchip RK35xx). Like the BLAS backend, this is a mul-mat-only
// accelerator: it offloads GGML_OP_MUL_MAT to the NPU via ork-driver and leaves every other op to
// the CPU backend. Uses CPU (host) buffers.

GGML_BACKEND_API ggml_backend_t      ggml_backend_ork_init(void);
GGML_BACKEND_API bool                ggml_backend_is_ork(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t  ggml_backend_ork_reg(void);

#ifdef  __cplusplus
}
#endif
