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
// default implied by the chosen path — NOT a user knob. Both paths run the round-trip-free on-NPU SwiGLU
// FFN chain; they differ only in the SiLU:
//   silu_int8_fused = false (DEFAULT, RECOMMENDED): int16 COHERENT — the SiLU is the UNFUSED standalone int16
//                                      SDP op on the NPU (int8 gate matmul + separate int16 SiLU). PPL ~19.0,
//                                      ~71 tok/s prefill. Fast AND coherent.
//   silu_int8_fused = true:            int8 FULLY FUSED through-and-through — the SiLU is fused into the gate
//                                      matmul (all-int8, all-NPU). Lower coherence AND currently SLOW (~1.7 tok/s:
//                                      the fused primitive re-warms + reloads its LUT per gate call every layer —
//                                      optimization pending). Prefer the default until that thrash is fixed.
//   dflash:                            enable the speculative block-diffusion drafter (off by default).
GGML_BACKEND_API void                ggml_backend_ork_set_load_config(bool dflash, bool silu_int8_fused);

#ifdef  __cplusplus
}
#endif
