#!/usr/bin/env bash
# Same-run, adjacent A/B of the loop vectorizer plus the gcc peer.
#
# The only reportable instrument on this laptop is an adjacent ratio: absolute ns swing by 2-3x with
# power state, thermal history and P/E-core placement. So all three arms run interleaved, ROUNDS
# times, and the minimum of each arm is taken per kernel.
#
#   bench/vecbench_ab.sh [rounds]
set -u
WK=${WK:-C:/Users/Quant/wkt/wk-vec/release/wukongc.exe}
CPEER=${CPEER:-C:/Users/Quant/wkt/hp-bin/vecbench_c.exe}
SRC=${SRC:-bench/vecbench.wk}
ROUNDS=${1:-3}

declare -A best_off best_on best_c
names=(1 2 3 4 5 6 7)

declare -A chks
run_into() {           # $1 = assoc array name, $2.. = command
  local -n dst=$1; shift
  # `tr -d '\r'`: the gcc peer is an MSYS-built Windows binary and prints CRLF, which makes the
  # numeric comparison below fail with "integer expression expected" on every line.
  local out; out=$("$@" 2>/dev/null | tr -d '\r' | paste - - -)
  while read -r tag ns chk; do
    [ -z "${tag:-}" ] && continue
    if [ -z "${dst[$tag]:-}" ] || [ "$ns" -lt "${dst[$tag]}" ]; then dst[$tag]=$ns; fi
    chks[$tag]="$chk"
  done <<< "$out"
}

# The arm order alternates by round. Running one arm always first gives it the cold cache and the
# other the warm one, which showed up as a 1.14x "speedup" on the kernel that this pass does not
# touch at all — i.e. as pure ordering bias, not as a result.
for r in $(seq 1 "$ROUNDS"); do
  "$WK" --run --backend=native -O2 "$SRC" >/dev/null 2>&1                          # warm
  WUKONG_NO_VECTORIZE=1 "$WK" --run --backend=native -O2 "$SRC" >/dev/null 2>&1    # warm
  if [ $((r % 2)) -eq 1 ]; then
    run_into best_off env WUKONG_NO_VECTORIZE=1 "$WK" --run --backend=native -O2 "$SRC"
    run_into best_on  "$WK" --run --backend=native -O2 "$SRC"
  else
    run_into best_on  "$WK" --run --backend=native -O2 "$SRC"
    run_into best_off env WUKONG_NO_VECTORIZE=1 "$WK" --run --backend=native -O2 "$SRC"
  fi
  run_into best_c   "$CPEER"
done

printf '%-4s %12s %12s %12s   %8s %8s\n' kern wukong-off wukong-on gcc-O3 'on/off' 'on/gcc'
for k in "${names[@]}"; do
  off=${best_off[$k]:-0}; on=${best_on[$k]:-0}; c=${best_c[$k]:-0}
  printf '%-4s %12d %12d %12d   %8.2f %8.2f\n' "$k" "$off" "$on" "$c" \
    "$(awk -v a=$off -v b=$on 'BEGIN{print (b?a/b:0)}')" \
    "$(awk -v a=$on -v b=$c 'BEGIN{print (b?a/b:0)}')"
done
