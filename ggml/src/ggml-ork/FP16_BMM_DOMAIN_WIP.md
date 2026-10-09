# ORK_ATTN × multi-domain int8 orkpack — root cause, fix, and the worth-it verdict

> **READ THE ADDENDUM AT THE BOTTOM FIRST (2026-08-19b).** The domain fix and the perf verdict below still
> stand, but **"the quality defect is in `ggml_backend_ork_flash_attn_ext`'s own CPU-side math" is WRONG** —
> that math was always correct. The PPL blowup was `ggml_backend_ork_bmm_fp16` (transposed weight) plus a
> stale cached weight copy in `ork_mm_repack_f16`; the FA handler had a separate, silent M-envelope bug that
> the PPL harness never even exercised (`ork_ppl` runs with flash attention DISABLED). All fixed;
> PPL is now **10.7607** vs the 10.8332 baseline.

Task: make batched/dynamic matmuls (attention QK^T / A·V, GDN chunk GEMMs) offloadable to the RK3588 NPU on
a model whose dense weights are RESIDENT in a multi-domain int8 `.orkpack` — or prove it can't and scope the
int8 alternative.

## STATUS: two separate answers, both evidence-backed

1. **The blocker is FIXED and validated.** Root cause = IOMMU-domain placement of the transient fp16
   attention scratch. `ORK_ATTN=1 ORK_ATTN_SM_CPU=1` now runs with **0 submit failures** at pp64 **and**
   pp512, ork-driver `make test` ALL PASS, attest refreshed.
2. **But the offload is NOT worth enabling**, on two independent grounds measured after the fix:
   - **Perf: neutral-to-negative.** pp64 31.84 → 30.56 t/s (−4%); pp512 36.35 → 36.58 t/s (+0.6%, noise).
   - **Quality: broken.** PPL 10.83 → 67.86 (6.3×). This is a **pre-existing defect in
     `ggml_backend_ork_flash_attn_ext`'s own CPU-side math**, NOT in the NPU matmuls and NOT caused by the
     domain fix (proof below). The 60 s submit timeouts were previously masking it — you cannot measure
     quality through a run that self-heals 3×60 s.

Recommendation: **keep the domain fix** (it is correct, cheap, and unblocks the path), leave `ORK_ATTN` off
by default, and do **not** invest in the fp16 attention/GDN offload. See "Worth-it verdict".

---

## Repro

Board `board`, `/home/michael/qwen3.6-35b-a3b-q8attn.gguf` (+ `.orkpack`, 18.4 GB), binaries in
`~/llama.cpp/build/bin`. Build on the board: `cd ~/llama.cpp/build && cmake --build . --target llama-bench -j6`.

```
baseline : sudo env ORK_PROFILE=1 taskset -c 4-7 ./llama-bench -m <gguf> -p 64 -n 0 -t 4 -r 1
broken   : + ORK_ATTN=1 ORK_ATTN_SM_CPU=1     -> pp64 1.01 t/s, 9 submit failures
fixed    : same command after this change     -> pp64 30.56 t/s, 0 submit failures
```

---

## FIRST CORRECTION TO THE BRIEF: `ORK_ATTN` has FOUR gates, and the one that fires is #4

The brief lists three (`supports_op` MUL_MAT batched claim; `graph_compute` dispatch to
`ggml_backend_ork_bmm_fp16`; SOFT_MAX claim). There is a fourth, and it is the only one that matters here:

```
ggml-ork.cpp  case GGML_OP_FLASH_ATTN_EXT:   if (getenv("ORK_ATTN") == nullptr) return false;
                                             if (N < 64) return false;   // prefill only
```

llama.cpp fuses QK^T + softmax + A·V into ONE `FLASH_ATTN_EXT` node, so a MUL_MAT with `ne[2]>1` never
appears for attention. **`ggml_backend_ork_bmm_fp16` is never called on this model** — proven: an
`ORK_ATTN_TRACE` print inside its scratch getter produced zero output across a full run, and `ORK_XPROF`
shows no `MC_MM`/`SC_MM` transitions at all. All the fp16 work goes through
`ggml_backend_ork_flash_attn_ext` → `attn_pool_ensure` → the nonblock doorbell (`ork_dyn_begin_mc`,
npu.c:11815, `XP_STREAM_F16`, `OCK_HW`).

`ORK_XPROF` (per forward pass):
```
CHAIN_NT   : COLD=1 F16=20 CHAIN=499     <- 520 int8 matmul calls, 20 of them entered FROM fp16
STREAM_F16 : F16=20 CHAIN=20             <- 40 fp16 calls
```
`[attnprof] calls=60` over 6 prefill passes ⇒ **10 full-attention layers** (rest are GDN); 10 layers × 2
matmuls (QK^T, A·V) = 20 fp16 entries per pass. ✓ consistent.

---

## ROOT CAUSE of the submit timeouts

**The fp16 attention scratch was placed in a different IOMMU domain than the int8 weights it interleaves
with, so every attention layer forced two `switch iommu domain` events. A nonblock doorbell job whose
completion IRQ never fired pins the outgoing domain's job refcount, so the switch stalls the NEXT submit
for the full 60 s submit timeout.** The op that FAILS is therefore never the attention matmul — it is the
int8 matmul immediately AFTER it.

`attn_pool_ensure` called `ork_mm_f16_scratch(c, …)` with **no domain control at all**, so the scratch
landed in `ork_dom(c->pack_domain)` = wherever the weight loader last left the pack cursor, and there was
**ONE pool shared by every attention layer** — while the model's attention layers are split across
domain 0 (1.02 GiB) and domain 1 (0.68 GiB).

### Evidence

Run A log (`scratchpad/runA.log`), pp64 = 1.01 t/s:
```
[ork] WARNING: RKNPU_SUBMIT ioctl failed (rc=-1, errno=110) | submit domain=1 task_number=4 core=0x1
  | last regcmd op=ork_dyn_colsplit_ks weight[K=4096 N=2048 dom=1 imported=1]. Triggering self-healing reset...
  [iova@fail] domain 0 live=1058 MiB (ceil 3900 MiB)
  [iova@fail] domain 1 live=178 MiB (ceil 3900 MiB)
```

1. **9 failures = 3 events × 3 cores.** All three cores of a submit set fail together ⇒ the *domain* is
   stalled, not one core. All 3 events name the same weight, `K=4096 N=2048 dom=1 imported=1`.
2. **`K=4096 N=2048` is `attn_output`** (16 heads × 256 head_dim = 4096 → n_embd 2048) — the int8 matmul
   immediately after the attention block. (The brief's earlier `K=2048 N=512` is attn_k/attn_v: same family.)
3. **3 × 60 s = 180 s ≈ the entire wall.** `t_run` = 186 552 ms for pp64 vs ~2 400 ms fixed. The regression
   is *entirely* 3 timeouts, not a diffuse slowdown.
4. **IOVA exhaustion RULED OUT.** `[iova@fail]` is a dump printed *at the failure point* (npu.c:936-942),
   not a preceding allocation failure. 1058 / 178 MiB against a 3900 MiB ceiling. npu.c's own comment on
   that dump: "shows whether the domain overflowed (size/pressure) vs a **non-size DMA-walk stall**."
5. **Kernel view** (dmesg): `RKNPU: job … commit elapse time: 61.2s, timeout: 60000000us` → `RKNPU: job
   timeout`, with `RKNPU: switch iommu domain from 1 to 0` / `0 to 1` and `soft reset, num: 6`.
   "commit elapse time" = the job was accepted and **blocked before commit** — i.e. blocked on the domain
   switch, not miscomputing.
6. **ork-driver already documents this exact cascade** — `dom_dirty`, npu.c:162: *"a genuine doorbell MISS
   left an unreaped (dropped) job in dom_active — its completion IRQ never fired, so the kernel's per-job
   interrupt_count for this domain stays >0 forever. A stuck job makes the NEXT iommu-domain switch time
   out → cascade."* And `ork_dom_flush_if_dirty` (npu.c:720-747): *"get_and_switch(D+1) waits on D's
   refcount>0 and TIMES OUT … the reap can never happen across the boundary."*
   **But `dom_dirty` is set at only two sites, both int4-gated** (npu.c:11220; npu.c:12504
   `if (h->mc_dt == DT_I4)`). int8/fp16 doorbell drops never arm the reap.
7. **Why only with `ORK_ATTN=1`**: per `dom_dirty`'s own comment, *"single-domain never switches, so it
   never acts on the flag."* The baseline switches domains only at layer boundaries between co-domain int8
   weights; an out-of-domain fp16 scratch inserts two extra switches per attention layer.
8. **Intermittency fits**: 10 attention layers × 2 = 20 F16↔int8 boundaries per pass, of which 2-3 fail.
9. **The fix confirms the mechanism directly**: the trace shows exactly **two** pools created, `dom=0` and
   `dom=1` — i.e. attention layers really do span both domains, so one shared pool could only ever be
   co-domain with one of them. Making it two co-domain pools → 0 failures.

### Ruled out

| Hypothesis | Verdict | Basis |
|---|---|---|
| IOVA / domain exhaustion | **RULED OUT** | `[iova@fail]` is a failure-time dump; 1058/178 MiB vs 3900 MiB ceiling |
| CONTIGUOUS/CMA fragmentation from per-node scratch bcreate/free | **RULED OUT by measurement** | switching `bmm_fp16` to a shared cache left the count unchanged (9 → 9) — and `bmm_fp16` never even runs here |
| fp16↔int8 mode transition | not the mechanism | `mode_probe` all `none (safe)`; and a *commit-stage* stall is not a miscompute |
| stale per-domain warm flags across a global reset | not the mechanism | every run path calls `dom_activate` BEFORE `ork_npu_enter` (colsplit npu.c:11436/11441, `run()` 5306/5337, doorbell 11814/11815), so the transition always clears the *incoming* domain's flags |
| porting fp16 scratch to `bscratch()` / importing scratch | pre-ruled-out, do not retry | `bscratch()`: domain 0 is plain bcreate; import-scratch is a PROVEN DEAD END that hard-wedged this board (2026-08-08/09) |
| MoE router softmax through the int16 exp LUT | **real but minor** | un-claiming SOFT_MAX moved PPL 67.86 → 70.06 — a measurable effect, but not the blowup |

---

## THE FIX (uncommitted)

Place transient fp16 attention scratch in the domain that is **already active**, one pool per domain.
A scratch has no natural home domain, so the cost to minimise is not bytes — it is **domain switches**.
The FA node sits between the same layer's int8 attn_q/k/v and attn_output, which are co-domain by the
layer-aligned residence rule, so the active domain *is* that layer's domain ⇒ attention adds **zero**
switches and the run switches exactly as often as the 0-failure `ORK_ATTN=0` baseline.

| file | change |
|---|---|
| `ork-driver/src/npu.c` | new pure getter `ork_npu_active_domain(c)` → `c->dom_active` (no state change) |
| `ork-driver/include/ork_npu.h` | its declaration + doc comment |
| `ork-driver/tests/sbc_attest.txt` | refreshed by board `make test` (ALL PASS) |
| `ggml-ork.cpp` `struct ork_attn_pool` | + `dom` field |
| `ggml-ork.cpp` ctx | `attnp` → `std::unordered_map<int, ork_attn_pool>` (one pool per domain, ~1-2 MB each) |
| `ggml-ork.cpp` `attn_pool_free` | split into `attn_pool_free_one` + a free-all for teardown |
| `ggml-ork.cpp` `attn_pool_ensure` | selects/creates the pool for `ork_npu_active_domain()`, pins `pack_domain` to it around the scratch allocs and restores it, returns `ork_attn_pool*` (NULL on failure) |
| `ggml-ork.cpp` `ork_get_f16_scratch` | same co-domain treatment + cache key `(dom<<48)|(K<<24)|N` (this is the `bmm_fp16`/FFN-JIT path — correct, but dead code on this model) |
| `ggml-ork.cpp` SOFT_MAX `supports_op` | `ORK_ATTN_SM_CPU` now also un-claims standalone SOFT_MAX (see below) |

**Kept** the pre-existing shared-scratch cache in `bmm_fp16` (was alloc/free per node). It is NOT the fix
(measured 9 → 9) and that path is unreachable on this model, but a cache is the right shape for the
co-domain change and avoiding the documented CMA churn is independently good. Not load-bearing.

### Secondary fix: `ORK_ATTN_SM_CPU` now un-claims standalone SOFT_MAX

The flag previously only switched the FA-internal softmax (`sm_npu`), so with `ORK_ATTN=1` the
`GGML_OP_SOFT_MAX` claim stayed live and swallowed every qualifying softmax in the graph — **including the
MoE expert-router softmax** (`ne[0]`=n_expert, %32==0; `ne[1]`=n_tokens>1 in prefill). Routing weights then
came back through the coarse int16 exp LUT instead of fp32 `expf`. Worth **PPL 70.06 → 67.86**. The LUT is
fine for attention scores (row-renormalized, differences only) but not for a router distribution. This
makes the flag mean what its name says.

---

## THE REMAINING DEFECT (pre-existing, NOT from this fix): FA handler math

PPL, window 1024, ubatch 512, `/tmp/ppl_corpus.txt` (all runs 0 submit failures):

| config | PPL | mean NLL |
|---|---|---|
| baseline (no ORK_ATTN) | **10.8332** | 2.3826 |
| `ORK_ATTN=1 ORK_ATTN_SM_CPU=1` | 67.8630 | 4.2175 |
| + both FA matmuls forced to CPU (`ORK_ATTN_CPU=3`) | **67.8630** | **4.2175** |
| + SOFT_MAX un-claimed | 70.0586 | 4.2493 |

The baseline 10.8332 validates the corpus against the documented 10.77 (+0.6%).

**`ORK_ATTN_CPU=3` gives byte-identical NLL to the NPU run (4.2175 both).** Therefore:
- the NPU fp16 matmuls in the co-domain scratch are numerically fine — this **exonerates the domain fix**;
- the blowup is in `ggml_backend_ork_flash_attn_ext`'s **own CPU-side math** (mask / GQA mapping / output
  layout / padding), independent of the NPU entirely.

By elimination the FA handler is the sole culprit: `bmm_fp16` never runs (proven), the SOFT_MAX claim is
now excluded and accounts for only ~3%, and `ORK_ATTN` gates nothing else.

Not root-caused further (out of scope, and the worth-it verdict makes it moot). Checked and cleared:
attention sinks are correctly declined (`op->src[4]` → CPU); `tnorm_ok` is 0 under `ORK_ATTN_SM_CPU` so the
transposed A·V path is safely skipped; GQA `hkv = h/rk2` is the correct mapping. Prime remaining suspects:
the `nkv` vs `nkvp` padding boundary (llama.cpp pads the KV view to a multiple of 256, so at pp64 `nkv=256`
with only 64 valid positions — masked columns are handled inconsistently between the scale+mask pass, which
writes only `j<nkv`, and the max/exp passes) and the final output scatter layout.

---

## WORTH-IT VERDICT (deliverable 3): do not build this

Measured on the fixed build, `-t 4`, governors verified, 0 submit failures:

| test | baseline | ORK_ATTN=1 + SM_CPU | ratio |
|---|---|---|---|
| pp64 | 31.84 ± 0.02 | 30.56 ± 0.13 | **0.96×** |
| pp512 | 36.35 ± 0.11 | 36.58 ± 0.05 | 1.006× (noise) |

### Composition of the prefill wall (pp512, wall = 3×512/36.58 = **42.0 s**, matching Stage-0's 42 s)

- **ork NPU matmul time = 4.88 s = 11.6 % of the wall** (`run_i8: 1560 calls, 4879.3 ms`). Confirms
  Stage-0's "~10 % NPU dense/attn".
- **Within NPU time** (`ORK_SEG_TIME=1`): QKV proj 54.0 %, FFN gate/up/GLU/down 26.3 %, other mul_mat
  13.7 %, O proj 6.1 %. In the baseline, `QK^T scores` / `softmax` / `A.V` are **0 ms, n=0** — attention is
  entirely CPU-side.
- **The attention block once offloaded** (`ORK_ATTN_PROF=1`, cumulative over 60 FA calls):
  `total = 1422 ms`, of which **NPU matmul = 417 ms (29 %)** and **CPU overhead = 1005 ms (71 %)**:
  densify+repack Q/K^T `dqk = 461 ms`, densify+repack V `dv = 224 ms`, softmax `SM = 291 ms`,
  scatter `sc = 31 ms`.

**The GDN/attention chunk GEMMs are a small share of the prefill wall, and the cost of marshalling
operands into the NPU's fp16 tile layout is 2.4× the matmul time it saves.** That is why the offload is
perf-neutral at best. Routing these GEMMs through the int8 path instead (deliverable 2b) would not change
this: the `dqk`/`dv` densify+repack cost is dtype-independent and is the dominant term, and int8 would add
quantization on top. This is the same structural wall already recorded for SDP LUT ops (`ork_spine op
placement`) and the MoE dispatch floor (`MoE W4A4 prefill wall`).

---

## Tree state (all UNCOMMITTED, per instructions)

- `ggml/src/ggml-ork/ggml-ork.cpp` — modified
- `ggml/src/ggml-ork/ork-driver/src/npu.c` — modified (pure getter)
- `ggml/src/ggml-ork/ork-driver/include/ork_npu.h` — modified (declaration)
- `ggml/src/ggml-ork/ork-driver/tests/sbc_attest.txt` — refreshed (board `make test` ALL PASS)
- `ggml/src/ggml-ork/FP16_BMM_DOMAIN_WIP.md` — this doc
- board `~/llama.cpp` additionally carries pre-existing local edits to `ggml/include/ggml-ork.h` and
  `src/llama.cpp` (orkpack path derivation) that are NOT mine and NOT on the Mac — leave them.

## If someone resumes this

1. The domain fix is done and gated; nothing pending there.
2. To make `ORK_ATTN` shippable you must fix the FA handler math first. Fast loop: it reproduces on
   `-p 64` in one 4-minute run, and `ORK_ATTN_CPU=3` isolates it to pure CPU math, so it can be debugged
   without any NPU risk. Start at the `nkv` vs `nkvp` masked-column boundary in the
   `ORK_ATTN_SM_CPU` branch and at the output scatter.
3. Even after that, expect no prefill win (see the verdict). The lever that would matter is eliminating
   the `dqk`/`dv` densify+repack, not the matmul precision.
4. Consider arming `dom_dirty` for int8/fp16 doorbell drops (not just int4) as defence in depth — the
   co-domain fix removes the switch, but any future out-of-domain transient reopens the same 60 s cascade.

## Board-ops gotchas

- Wrap as `sudo timeout -s TERM <secs> ./x` (NOT `timeout N sudo ./x`). SIGTERM only, never SIGKILL.
- Single-stream: one board workload at a time.
- **Drop caches between big runs**: after several back-to-back 18 GB loads, `buff/cache` fills and weight
  import fails with `PRIME_FD_TO_HANDLE: Cannot allocate memory` + `CREATE FAIL errno=14`, which looks like
  a code bug and is not. `sudo sync && sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"`.
- A **teardown-only** kernel WARN is normal and pre-existing: `dma-iommu.c:691 __iommu_dma_unmap` via
  `rknpu_gem_object_destroy` → `drm_prime_gem_destroy` → `system_heap_unmap_dma_buf`, at process exit when
  the orkpack dma-buf imports are unmapped. Present in the baseline too; its own register dump can push the
  header out of the dmesg ring, making it look like a fresh crash. Strip hex lines to read it:
  `sudo dmesg | grep -vE '^\[[0-9. ]+\] +[0-9a-f]{4}\*? '`.
- Governors verified this session: dmc `2112000000`, cpu6 `2352000`, `performance`.
- `/tmp/ppl_corpus.txt` present (106 566 B); baseline PPL on it = 10.8332.

---

# ADDENDUM 2026-08-19b — the quality defect ROOT-CAUSED and FIXED (three bugs, none of them where this doc said)

**Everything above about the domain fix stands. The "REMAINING DEFECT: FA handler math" section above is WRONG
and is superseded by this addendum.** The FA handler's CPU-side math was always correct. The PPL blowup was in a
*different* handler, reached by a *different* tool, and the FA handler had its own (separate, silent) defect that
the PPL runs never even exercised.

## The mis-attribution, and how it happened

The prior pass reasoned: "`ORK_ATTN_CPU=3` forces BOTH FA matmuls to CPU and the NLL is byte-identical, therefore
the bug is in `ggml_backend_ork_flash_attn_ext`'s own CPU-side math." Both premises are true; the conclusion does
not follow, because of one line in the harness:

```
ork-driver/tools/re/ork_ppl.cpp:49
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;   // ggml-ork path (as ork_bench)
```

**`ork_ppl` runs with flash attention DISABLED.** So on the PPL harness there is no `GGML_OP_FLASH_ATTN_EXT` node
at all — attention arrives as the classic decomposition `mul_mat(k,q)` → `soft_max_ext` → `mul_mat(vᵗ,kq)`, and
`ORK_ATTN`'s *other* claim (batched/dynamic `MUL_MAT`, `ne[2]>1 && M>1`) routes both of those matmuls to
`ggml_backend_ork_bmm_fp16`. `ORK_ATTN_CPU` only moves the FA handler's internal matmuls, so of course it changed
nothing — the FA handler never ran. Conversely the `[attnprof] calls=60` / "bmm_fp16 is dead code" observations
were made under **`llama-bench`**, which *does* enable flash attention. Both observations were correct; they were
about different tools, and cross-attributing a quality number from one to the code path of the other is what sent
the investigation to the wrong function.

Lesson for the registry: a quality number is only evidence about the code path the *measuring tool* actually
executes. Check the harness's `flash_attn_type` before attributing an attention result.

## Method: a CPU-backend differential probe, not a model run

Two ~90-line standalone probes (in the session scratchpad, not committed; rebuild from this description if needed)
made every question a 20-second answer instead of a 6-minute model load:

- **`fa_probe <DK> <DV> <Hkv> <rk2> <n_kv> <nb>`** — builds ONE `GGML_OP_FLASH_ATTN_EXT` node with
  llama.cpp-realistic operands (F32 Q, F16 K/V, F16 causal mask padded to 64 rows), allocates it in a host
  buffer, computes it with `ggml_backend_graph_compute(cpu_backend, gf)` → saves as the oracle, poisons `dst`,
  then computes the SAME graph with `ggml_backend_graph_compute(ork_backend, gf)` and diffs elementwise plus
  per-head and per-query NRMSE. Calling the ork backend directly bypasses `supports_op`, so the handler under
  test always runs.
- **`bmm_probe <K> <N> <M> <Hkv> <rk2>`** — same shape for one batched `ggml_mul_mat` (`src0`[K,N,Hkv] F16 ×
  `src1`[K,M,H] F32), plus a **mechanism oracle**: it re-computes the product with a deliberately mis-indexed B
  and reports whether the NPU output matches that instead of the truth. That is what turned "wrong" into "wrong
  in exactly this way".

Build on the board:
`g++ -O2 -std=c++17 X.cpp -o X -I ~/llama.cpp/ggml/include -L ~/llama.cpp/build/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$HOME/llama.cpp/build/bin`

## BUG 1 — `ggml_backend_ork_bmm_fp16` fed the driver a TRANSPOSED weight  ← this is the PPL blowup

`ork_mm_pack` / `ork_mm_repack_f16` take `B[K,N]` **row-major**: `B[k*N + n] = weight(k,n)`. ggml stores element
`(k,n)` of a `[K,N]` tensor at `n*nb[1] + k*nb[0]` — `ne[0]`-contiguous — so `src0`'s raw buffer is `[N][K]`,
i.e. **Bᵀ**. The handler's fill was

```c
for (size_t j = 0; j < (size_t)K*N; j++) B[j] = (ork_f16) rds(src0, s0, j % K, j / K);   // == src0's raw order
```

which reproduces exactly that raw order. Every batched matmul therefore ran against the transpose of its weight.
Proven, not inferred: `bmm_probe`'s mechanism oracle — the emulated transposed-B product matched the NPU output to
**NRMSE 1.9e-4** (i.e. fp16 noise) at K=N=256, at K≠N, and at M=64 and M=512. Fix:

```c
for (size_t j = 0; j < (size_t)K*N; j++) B[j] = (ork_f16) rds(src0, s0, j / N, j % N);   // B[k*N+n] = src0(k,n)
```

## BUG 2 — `ork_mm_repack_f16` left the fp16 multi-core colsplit's CACHED weight copy stale (npu.c)

With BUG 1 fixed, `bmm_probe` went from 98 % of elements wrong to exactly the **second half of the heads** wrong
(NRMSE 1.42 → 1.00, first bad element at head 8 of 16 with `Hkv=2, rk2=8` — i.e. the first head that needs a
*different* `src0` slice). `ORK_NPU_MC=1` made it PASS. That named it:

the fp16 **multi-core colsplit does not read `w->Bb`**. It concatenates the `Sk` K-slice tiles into ONE contiguous
buffer (`w->Bbc` for `Sn==1`, `w->Bbc_ns[]` for `Sn>1`) — built **once**, behind a `*_valid` latch that was never
cleared (npu.c:11494, npu.c:5133). `ork_mm_repack_f16` refreshed `Bb` and `bsync`'d it, but the multi-core path
kept computing against the **first weight ever packed into that slot**, silently, forever. Single-core `run()` and
`ork_mm_run_stream_f16{,_chain}` read `Bb` directly and were always correct — which is exactly why the FA handler
(one `ork_w` per head + the chain primitive) never showed it while `ork_bmm_fp16` (one shared scratch slot
repacked per head + `ork_mm_run`) always did.

Fix (npu.c, `ork_mm_repack_f16`): refresh `Bbc` / `Bbc_ns[]` in place after refreshing `Bb`. `K,N` are unchanged
by contract, so the sizes match and the latches stay valid — no `bcreate`/`bdestroy` churn. `ork_mm_repack_i8`
was already coherent (it refreshes `Bf`; `Bbc` is fp16-only).

Board `make test`: **ALL TESTS PASSED**, `tests/sbc_attest.txt` refreshed and included in the change
(`make check-attest` OK on the Mac tree, `CORE_SHA=f377101c…`).

## BUG 3 — the FA handler exceeded the single-M-tile envelope of the fp16 stream primitive

Separate defect, in the actual named target, and invisible to `ork_ppl` (which never builds an FA node). With
`ORK_ATTN_CPU=3` the handler was bit-clean at every shape; with the NPU matmuls on it was garbage at ubatch 512:

| shape (DK=DV=256, H=16, Hkv=2) | NPU matmuls | `ORK_ATTN_CPU=3` |
|---|---|---|
| n_kv 512, nb 64 / 128 / 256 | PASS (NRMSE 2e-4) | PASS |
| n_kv 512, nb 384 / 512 | **FAIL, NRMSE 30–69, first bad at query 352** | PASS |
| n_kv 1024, nb 512 | **FAIL, NRMSE 153, first bad at query 0** | PASS |
| n_kv 512, nb 512, `ORK_ATTN_CPU=2` (A·V on CPU) | **PASS** | — |
| n_kv 512, nb 512, `ORK_ATTN_CPU=1` (QK^T on CPU) | FAIL | — |

`ORK_ATTN_CPU=1/2` isolates it to **A·V only**, and the shape dependence names the mechanism.
`ork_mm_run_stream_f16` / `_chain` synth **ONE regcmd program per task** — they are documented "Single M-tile"
and do not M-schedule. A program's rows are only computed against the right K-partition inside the 0x1040
K-reduction schedule's tile, `mg_max(K)*64`, where `mg_max` comes from `CBUF_CON0 = base − slope·(mg−1) ≥ 0x1b`
with `base/slope` derived from K (npu.c `synth()`; npu.c `run()` tiles to exactly `mg_max*64` and comments that
going past it "miscomputes"). For fp16:

| K | mg_max | legal M per program |
|---|---|---|
| 256 (`DK`, QK^T) | 11 | **704** — ubatch 512 fits, which is why QK^T was fine |
| 512 (`nkvp`, A·V at n_kv 512) | 5 | **320** — ubatch 512 does not fit |
| 1024 (`nkvp`, A·V at n_kv 1024) | 2 | **128** — wrong from row 0 |

The handler passed `M = n_tokens` straight through. Fix: `ork_f16_mtile(K)` + `ork_attn_chain_mtiled()` in
ggml-ork.cpp, mirroring `run()`'s own M-scheduler, applied at all three chain submits (QK^T, A·V, and the
transposed A·V of the `tnorm` path). Cost: 2 submits instead of 1 at ubatch 512 — irrelevant on a path that is
perf-neutral anyway.

The right long-term home for this is ork-driver: `ork_mm_run_stream_f16{,_chain}` should either M-tile internally
or `return -2` for an out-of-envelope M instead of silently miscomputing. Left as a follow-up (kept out of npu.c
here to keep the attest surface minimal).

## BUG 4 (not fixed — made OPT-IN) — the on-NPU softmax is quality-broken at some row counts

With the above fixed, the handler is clean under the CPU softmax at every shape. The **on-NPU softmax** (which was
the DEFAULT) is not:

- `MR = H·N = 8192` → NRMSE 2e-4 (indistinguishable from the CPU softmax)
- `MR = 6144` and `MR = 4096` → **NRMSE 0.55–0.69, wrong from row 0**
- byte-identical numbers with `ORK_ATTN_TNORM_OFF`, `ORK_ATTN_MAX_CPU` and `ORK_ATTN_SM_SUMCPU` all set

That last line excludes the transposed A·V, the int8 row-max and the reduce-matmul Σ, leaving
**`ork_npu_exp_i16`'s own row-count envelope** as the culprit — a separate op, not this handler.

So the on-NPU softmax is now **opt-in under `ORK_ATTN_SM_NPU`**, both for the FA-internal chain and for the
standalone `GGML_OP_SOFT_MAX` claim in `supports_op`. Consequences:

- plain `ORK_ATTN=1` is now correct end to end — including the **MoE expert-router softmax**, which the int16 exp
  LUT perturbs (worth ~3 % PPL: 70.06 → 67.86). That is what `ORK_ATTN_SM_CPU` used to be needed for.
- `ORK_ATTN_SM_CPU` is kept as an accepted alias meaning "CPU softmax", so existing invocations still work.

## Before / after

Probes (differential vs the ggml CPU backend, 0 submit failures in every run):

| probe | before | after |
|---|---|---|
| `fa_probe` DK=DV=256 H=16 Hkv=2, n_kv {512,1024,2048} × nb {64,128,256,384,512} | NRMSE 30–153 at nb≥384 | **NRMSE 2e-4, 0 elements out of tolerance, every shape** |
| `fa_probe` DK=DV=128 H=16 Hkv=4 n_kv=1024 nb=256 | NRMSE 0.55 | **NRMSE 2.1e-4** |
| `bmm_probe` K/N {256,512,1024} × M {64,512} × (Hkv,rk2) {(2,8),(1,16),(16,1)} | NRMSE 1.42, 98 % of elements wrong | **NRMSE 8e-4 – 1.7e-3** (fp16 rounding of the F32 operand) |

PPL (`ork_ppl <q8attn.gguf> /tmp/ppl_corpus.txt 1024 512`, window 1024, ubatch 512):

| config | PPL | mean NLL | `RKNPU_SUBMIT` failures |
|---|---|---|---|
| baseline (`ORK_ATTN` unset) | **10.8332** | 2.3826 | 0 |
| `ORK_ATTN=1 ORK_ATTN_SM_CPU=1` — **before** this change | 67.8630 | 4.2175 | 0 |
| `ORK_ATTN=1 ORK_ATTN_SM_CPU=1` — **after** | **10.7607** | 2.3759 | 0 |

**−0.67 % vs baseline** (gate: within ~1 %). Same corpus (106 566 B — regenerated after a reboot to the
identical byte count), same build, governors verified (dmc `2112000000`, cpu6 `2352000`, `performance`).
The baseline reproduced its documented 10.8332 / NLL 2.3826 exactly, so the two runs are directly comparable.

The offload really did execute — this is not a "silently declined, therefore identical" pass: with
`ORK_ATTN_TRACE=1` the run logs `[f16scratch] K=256 N=256 active_dom=0 -> ok` and `active_dom=1 -> ok`,
i.e. `ork_get_f16_scratch` (only reachable from `ggml_backend_ork_bmm_fp16`) was called and created a
co-domain scratch in both resident domains, with 0 submit failures.

The 0.67 % *improvement* over baseline is a precision shuffle, not a real quality gain: the attention
matmuls now run fp16×fp16 on the NPU instead of ggml's f16×f32 CPU kernel. Treat it as noise.

First failed attempt, for the record (do not read a defect into it): the first pair of 1024/512 runs — the
**baseline included** — died with `PRIME_FD_TO_HANDLE: Cannot allocate memory` / `CREATE FAIL errno=14 ...
size=10880KB dom=2` right after the ork-driver `make test`, with 13 GB free and CmaFree 250 MB. A
`drop_caches` did not clear it; `sudo reboot` did, and both runs then completed clean on the first try. Same
class as the documented "several back-to-back 18 GB loads" gotcha, but reachable from `make test` too —
**reboot before quoting a PPL number after a heavy NPU session.**

## Still open

1. **`ork_npu_exp_i16` row-count envelope** (BUG 4). Until that is diagnosed, the on-NPU softmax stays opt-in.
   `fa_probe … 256 256 2 8 512 384` is a 20-second reproducer.
2. **`ork_mm_run_stream_f16{,_chain}` should not silently accept an out-of-envelope M.** The ggml-ork-side
   `ork_f16_mtile()` fix protects this caller only; a `-2` refuse (or internal tiling) in npu.c protects all of
   them. Same class of hazard as BUG 2: a primitive that miscomputes instead of refusing.
3. **Perf verdict is unchanged: leave `ORK_ATTN` off by default.** Correctness was the goal here. The
   marshalling wall (densify/repack = ~2.4× the matmul time saved; NPU matmul = 11.6 % of the pp512 wall) is
   untouched by any of this, and `bmm_fp16`'s fixed B fill is now a *strided transposing* gather, i.e. slightly
   slower than the (wrong) straight copy it replaced.
4. The two probes are in the session scratchpad only. If this path is picked up again, the `fa_probe` /
   `bmm_probe` pattern (CPU backend as oracle on the identical graph, ork backend called directly to bypass
   `supports_op`) belongs in `tests/` — all four bugs were found in one afternoon with it, after a full pass
   using model runs found none of them.
