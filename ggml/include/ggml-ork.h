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

// ---- .orkpack BUILD CONFIGURATION -------------------------------------------------------------------
// How a pack is BUILT is a decision with consequences (precision, file size, resident RAM), so it is an
// API object, not a scatter of environment variables. What a weight IS gets recorded in the pack itself
// (orkpack v6 stores each entry's tier AND its measured error), so RUNNING a pack needs no configuration
// at all — the file decides. This struct only describes what to WRITE.
//
// MIXED PRECISION IS THE DEFAULT. Uniform int4 is rarely the best use of a byte budget: a few weights
// carry disproportionate quantisation error, and promoting just those to int8 costs little. Pure int4 is
// therefore an explicit opt-in (mixed = false) rather than the implied default.
//
// Choosing WHICH weights to promote needs each weight's measured error, which only exists once it has been
// quantised. So either name them explicitly (promote_list), or point at a pack built earlier
// (qerr_source_pack) and let the policy rank by the stored qerr and promote worst-first until
// promote_budget_mb is spent. With neither, the first build is uniform and records the qerr that makes the
// second build informed.
struct ggml_backend_ork_pack_config {
    int          weight_bits;        // base tier for the bulk of the model: 4 or 8; 0 = UNSET (the
                                     // default), leaving ORK_QUANT / the source-type policy to decide
    bool         mixed;              // default true — promote the worst weights to int8
    const char * promote_list;       // explicit "name,name,..."; NULL = use the policy below
    const char * qerr_source_pack;   // a previously built pack to read qerr from; NULL = no ranking available
    float        promote_budget_mb;  // how much EXTRA file size promotion may spend (default 8 MiB)
    float        promote_qerr_min;   // never promote a weight whose measured error is below this (default 0.05)
};

// Fill with the defaults described above. Always call this first, so adding a field cannot silently
// change behaviour in a caller that zero-initialised the struct.
GGML_BACKEND_API void ggml_backend_ork_pack_config_defaults(struct ggml_backend_ork_pack_config * cfg);

// Apply a build configuration. Call BEFORE model load (like ggml_backend_ork_set_load_config); it affects
// pack WRITING only. Passing NULL restores the defaults. The equivalent environment variables remain as a
// fallback for scripts, but an explicit config always wins.
GGML_BACKEND_API void ggml_backend_ork_set_pack_config(const struct ggml_backend_ork_pack_config * cfg);

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

/* WINDOWED CALIBRATION. The Hessian (K*K doubles per weight) is not disk-backed, so it cannot be paged --
 * only rebuilt by replaying the corpus. Calibrating every weight at once costs sum(K^2*8) resident, which
 * is ~236 GiB for a 64-layer 27B (the 64 ffn_down at K=17408 are 2.42 GiB each). Instead, calibrate a
 * layer range per pass: set_window(lo,hi) -> run the corpus -> finalize() -> repeat. Because finalize
 * writes the quantised weight back into the weight cache and windowed runs keep it pinned, later windows
 * calibrate against already-quantised upstream layers, i.e. SEQUENTIAL GPTQ rather than one-shot.
 * lo < 0 restores unwindowed behaviour. hessian_bytes(K) sizes the window against a RAM budget. */
GGML_BACKEND_API void                ggml_backend_ork_gptq_set_window(int lo, int hi);
GGML_BACKEND_API double              ggml_backend_ork_gptq_hessian_bytes(int K);
/* Largest K over every REGISTERED weight (claimed or not) -- available after the discovery pass, which
 * costs no Hessian memory, and used to size the window against a RAM budget. min_rows() by contrast
 * reports the largest K the CURRENT window claimed, which is what sets its calibration row count. */
GGML_BACKEND_API int                 ggml_backend_ork_gptq_max_k(void);

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

// ---- M-THRESHOLD CALIBRATION (opt-in) ----------------------------------------------------------
// The CPU/NPU routing threshold is where the NPU's fixed submit floor stops dominating. The built-in
// default (8) is MEASURED, but end-to-end and on one board: it moves with the model's shapes, the CPU
// clocks, and the submit floor.
//
// It cannot be calibrated from inside the backend. A per-shape microbenchmark was tried and does NOT
// predict it: timing one matmul in isolation misses the dominant cost at small M, which is graph-level --
// declining every node yields 1 graph split, accepting yields 133, and those 132 backend boundaries and
// their tensor copies are invisible to any single-matmul harness. Measured, that harness regressed M<=8 by
// up to 1.52x. The only instrument that measures the real quantity is an end-to-end sweep, and only the
// CALLER can run one, because only the caller controls the batch size.
//
// So the backend provides the two affordances a caller needs, and stays out of the policy:
//   set_min_m()   force the threshold for one measurement pass (-1 restores the default/pack value)
//   write_calib() store a measured threshold into an existing .orkpack, stamped with the machine state it
//                 was taken under; it is ignored on load by any machine that no longer matches
GGML_BACKEND_API void ggml_backend_ork_set_min_m(int m);
GGML_BACKEND_API bool ggml_backend_ork_write_calib(const char * pack_path, int min_m);
}

#endif
