#!/usr/bin/env bash
# run_ork_bench.sh — run ork_bench while sampling RK3588 resource use, then print the AVG and PEAK of each
# resource over the run. Turns a bare tok/s number into an attributable one: you SEE whether RAM bandwidth /
# NPU / GPU / CPU was the wall. (Renamed from the ork-driver bench_monitored.sh wrapper and specialized to
# ork_bench + the orkd daemon path.) Sources match oRKLLM's dashboard (monitor.js):
#   RAM    : /proc/meminfo  (MemTotal-MemAvailable)/MemTotal
#   RAM BW : /sys/class/devfreq/dmc/load  ("<pct>@<freq>Hz")  <- DMC utilisation = RAM-bandwidth pressure
#   NPU    : /sys/kernel/debug/rknpu/load ("Core0: X%, ...")  mean across cores   [needs root]
#   GPU    : /sys/class/devfreq/*.gpu/load ("<pct>@<freq>Hz", Mali)
#   CPU    : /proc/stat aggregate busy% (100% = all cores saturated)
#   SWAP   : /proc/meminfo (SwapTotal-SwapFree)/SwapTotal
#
# The ggml-ork backend routes every NPU submit through the orkd daemon (>=1.0.0); the client auto-spawns the
# binary named by ORKD_BIN, which this script points at build/bin/orkd unless already set.
#
# Usage:
#   sudo ./run_ork_bench.sh [--interval-ms N] [--csv FILE] [--label STR] [--build DIR] -- <model.gguf> <promptfile> [P] [G] [ubatch]
# NPU sampling needs root (debugfs) — run the whole thing under sudo so sampler + workload both see it.
#   sudo ORK_PERSIST=~/qwen3-1.7b-q8_0.orkpack ./run_ork_bench.sh --label orkd -- ~/qwen3-1.7b-q8_0.gguf ~/bench_prompt.txt 128 64
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
INTERVAL_MS=200; CSV=""; LABEL="ork_bench"; SAMPLER_CPUS="0-3"   # RK3588/RK3576 little cores (A55/A53) = cpu0-3
while [ $# -gt 0 ]; do
  case "$1" in
    --interval-ms)  INTERVAL_MS="$2";  shift 2 ;;
    --csv)          CSV="$2";          shift 2 ;;
    --label)        LABEL="$2";        shift 2 ;;
    --build)        BUILD_DIR="$2";    shift 2 ;;
    --sampler-cpus) SAMPLER_CPUS="$2"; shift 2 ;;   # cpu list for the sampler (keep it off the workload's cores)
    --) shift; break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "error: no ork_bench args after -- (need at least <model.gguf> <promptfile>)" >&2; exit 2; }

ORK_BENCH="$BUILD_DIR/bin/ork_bench"
[ -x "$ORK_BENCH" ] || { echo "error: ork_bench not found/executable at $ORK_BENCH (build it, or pass --build DIR)" >&2; exit 2; }
# orkd is the committed NPU path: the client auto-spawns ORKD_BIN. Default it to this build's daemon.
if [ -z "${ORKD_BIN:-}" ]; then
  if [ -x "$BUILD_DIR/bin/orkd" ]; then export ORKD_BIN="$BUILD_DIR/bin/orkd"
  else echo "error: ORKD_BIN unset and no $BUILD_DIR/bin/orkd — build the orkd target" >&2; exit 2; fi
fi
echo "[monitor] ORKD_BIN=$ORKD_BIN" >&2

[ -n "$CSV" ] || CSV="$(mktemp /tmp/run_ork_bench_XXXX.csv)"
STATE="$(mktemp /tmp/run_ork_bench_cpu_XXXX)"
SLEEP=$(awk "BEGIN{printf \"%.3f\", $INTERVAL_MS/1000}")

# Overall CPU busy% via /proc/stat aggregate-line deltas (prev in $STATE).
cpu_tick() {
  awk -v state="$STATE" '
    BEGIN{ if((getline line < state)>0){split(line,a," ");pt=a[1];pi=a[2]} close(state) }
    /^cpu /{ idle=$5+$6; tot=0; for(i=2;i<=NF;i++)tot+=$i
             dt=tot-pt; di=idle-pi; busy=(dt>0)?100*(dt-di)/dt:0
             if(busy<0)busy=0; if(busy>100)busy=100
             printf "%d %d", tot, idle > state; close(state)
             printf "%.0f", busy; exit }' /proc/stat
}

sampler() {
  cpu_tick >/dev/null 2>&1   # prime the delta baseline
  echo "ts_ms,ram,rambw,npu,gpu,cpu,swap" > "$CSV"
  local gpudir; gpudir="$(ls -d /sys/class/devfreq/*gpu* 2>/dev/null | head -1)"
  while :; do
    local ts ram rambw npu gpu cpu swap n0 n1 n2 mt ma st sf
    ts=$(date +%s%3N)
    # RAM
    mt=$(awk '/^MemTotal/{print $2}' /proc/meminfo); ma=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
    ram=$(( mt>0 ? (100*(mt-ma))/mt : 0 ))
    # SWAP
    st=$(awk '/^SwapTotal/{print $2}' /proc/meminfo); sf=$(awk '/^SwapFree/{print $2}' /proc/meminfo)
    swap=$(( st>0 ? (100*(st-sf))/st : 0 ))
    # RAM BW (DMC utilisation %)
    rambw="$(cat /sys/class/devfreq/dmc/load 2>/dev/null | grep -oE '^[0-9]+')"; rambw=${rambw:-0}
    # NPU (mean across cores)
    read -r n0 n1 n2 <<<"$(grep -oE '[0-9]+%' /sys/kernel/debug/rknpu/load 2>/dev/null | tr -d '%' | tr '\n' ' ')"
    n0=${n0:-0}; n1=${n1:-0}; n2=${n2:-0}; npu=$(( (n0+n1+n2)/3 ))
    # GPU
    gpu=0; [ -n "$gpudir" ] && gpu="$(cat "$gpudir/load" 2>/dev/null | grep -oE '^[0-9]+')"; gpu=${gpu:-0}
    # CPU
    cpu="$(cpu_tick)"; cpu=${cpu:-0}
    echo "$ts,$ram,$rambw,$npu,$gpu,$cpu,$swap" >> "$CSV"
    sleep "$SLEEP"
  done
}

sampler & SAMPLER_PID=$!
# Pin the sampler (and the cat/awk/grep it forks, which inherit affinity) to the little cores so it can't
# steal cycles from the big-core workload or the big-core-pinned NPU-driver threads — otherwise the monitor
# perturbs the very numbers it reports.
taskset -cp "$SAMPLER_CPUS" "$SAMPLER_PID" >/dev/null 2>&1 || echo "[monitor] warn: taskset unavailable, sampler not pinned" >&2
cleanup(){ kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "[monitor] label=$LABEL interval=${INTERVAL_MS}ms csv=$CSV sampler_cpus=$SAMPLER_CPUS" >&2
echo "[monitor] === ork_bench START ===" >&2
"$ORK_BENCH" "$@"; RC=$?
echo "[monitor] === ork_bench END (rc=$RC) ===" >&2
cleanup; trap - EXIT INT TERM

# Summary: AVG and PEAK for each of the 6 resources.
awk -F, -v label="$LABEL" '
  NR==1{ for(i=2;i<=NF;i++) name[i]=$i; next }
  { n++; for(i=2;i<=NF;i++){ v=$i+0; sum[i]+=v; if(n==1||v>mx[i])mx[i]=v } }
  END{
    if(n==0){ print "[monitor] no samples"; exit }
    printf "\n===== RESOURCE USE (%s, %d samples) =====\n", label, n
    printf "%-8s %8s %8s\n", "metric", "avg%", "peak%"
    for(i=2;i<=NF;i++) printf "%-8s %8.1f %8.0f\n", name[i], sum[i]/n, mx[i]
    print  "==========================================="
  }' "$CSV"
echo "[monitor] csv: $CSV" >&2
rm -f "$STATE"
exit $RC
