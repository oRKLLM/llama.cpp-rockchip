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
GGML_BACKEND_API void                ggml_backend_ork_set_hybrid(bool use_hybrid);

// ---- Load-time product config (the ONLY two user-facing options) ----
// Call BEFORE model load (i.e. before ggml_backend_ork_init). Consumed once; reload the model to change.
// Everything else (NO_BF, single-domain, ping-pong, mixed-nothrash, chain-on) is an internal validated
// default implied by the chosen silu path — NOT a user knob.
//   silu_int8_cpu = false (DEFAULT): all-NPU int16 SiLU FFN chain (coherent PPL ~19.0, no CPU crossing).
//   silu_int8_cpu = true:            int8 gate + CPU fp32 SiLU (coherent PPL ~18.4, small prefill crossing).
//   dflash:                          enable the speculative block-diffusion drafter (off by default).
GGML_BACKEND_API void                ggml_backend_ork_set_load_config(bool dflash, bool silu_int8_cpu);

#ifdef  __cplusplus
}
#endif
