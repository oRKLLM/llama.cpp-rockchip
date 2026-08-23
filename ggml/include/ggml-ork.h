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

// ---- .orkpack auto-persist helper ----
// Returns true iff a usable .orkpack exists at `path` for THIS build: present, footer magic + schema version
// match, and the ork-driver pack-compat token (ork_pack_format_version) matches — i.e. a read-mode load would
// succeed with no re-conversion. Returns false when absent OR stale (a tiling/quant change bumped the token).
// A tool/frontend uses this to decide whether to run a one-time build pass (which packs + writes the .orkpack)
// BEFORE timing, so the measured run always reads a prebuilt pack instead of JIT-packing into the hot path.
GGML_BACKEND_API bool                ggml_backend_ork_orkpack_valid(const char * path);
/* ORK_GPTQ phase 2: quantize every calibrated native-W4A4 weight with the Hessian accumulated over the
 * calibration forwards, then persist. Call ONCE after the calibration batches; a no-op unless ORK_GPTQ
 * is set. Heavy — three O(K^3) factorisations per weight. */
GGML_BACKEND_API void                ggml_backend_ork_gptq_finalize(void);
/* Calibration rows this model NEEDS (= largest registered K) and how many it has. rank(H) <= rows, and
 * below K rows GPTQ silently degrades to round-to-nearest in the null space — so the batch count is derived
 * from these, not configured. Both 0 outside a GPTQ run. */
GGML_BACKEND_API int                 ggml_backend_ork_gptq_min_rows(void);
GGML_BACKEND_API long                ggml_backend_ork_gptq_rows(void);

// ---- Async cross-stream submit path ----
// Run this backend's NPU graph on a worker thread: the launch returns immediately so the caller can do CPU
// work (e.g. a speculative draft's routing/sampling) while the NPU crunches; synchronize joins and returns the
// graph_compute status. The RKNPU is single-stream, so concurrent NPU submits from multiple ork backends
// SERIALIZE (one hardware queue) — the CPU halves overlap for free, the NPU halves cannot. At most ONE in-flight
// async job per backend. Opt-in primitive for a pipelined speculative/dflash decode loop; the plain synchronous
// path is unchanged when these are unused.
GGML_BACKEND_API void                ggml_backend_ork_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph);
GGML_BACKEND_API enum ggml_status    ggml_backend_ork_synchronize(ggml_backend_t backend);

#ifdef  __cplusplus
}
#endif
