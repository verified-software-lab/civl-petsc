#!/usr/bin/env bash
set -euo pipefail

# Default rule if none provided
run_rule="${1:-small}"

# Optional: show usage if they pass “-h” or “--help”
if [[ "$run_rule" == "-h" || "$run_rule" == "--help" ]]; then
  cat <<EOF
Usage: $0 [run_rule]

  run_rule   which make target to invoke (e.g. "small" or "big"; defaults to "small")
EOF
  exit 0
fi

ROOT_DIR="$PWD"
TARGET_DIR="$ROOT_DIR/targets"

# Build the list of subdirectories
mapfile -t SUBDIRS < <(find "${TARGET_DIR}" \
    -mindepth 1 -maxdepth 1 -type d \
    -not -name ".*" -not -name "CIVLREP" \
    -exec basename {} \;)

# SUBDIRS=("VecMAXPBY" "VecMAXPY" "VecMDot" "VecMDot_MPI" "VecMTDot" "VecMTDot_MPI" "VecMaxPointwiseDivide" "VecMaxPointwiseDivide_MPI" "VecPointwiseMaxAbs" "VecSetValuesBlocked")

# Compute total before we start
total_count=${#SUBDIRS[@]}

# grab a human-readable timestamp
pretty_ts=$(date +'%m/%d/%Y_%H:%M:%S')

# sanitize "/" → "-" so we don’t create folders,
#    but keep the colons in the time if you like
safe_ts=${pretty_ts//\//-}

# Ensure Reports directory exists
mkdir -p "$ROOT_DIR/Reports"

# Create a log file with the timestamp
LOG="$ROOT_DIR/Reports/Summary_${safe_ts}_${run_rule}_${total_count}.log"
CSV="$ROOT_DIR/Reports/Function_Stat's_${safe_ts}_${run_rule}.csv"

# Record overall start time
start_script=$(date +%s)

{
  echo "============================================================================================================================"
  echo "                                               Verifying civl-petsc targets                                               "
  echo "============================================================================================================================"
} | tee -a "$LOG"

# Initialize CSV file with headers
echo "Function,Time (seconds),Status" > "$CSV"

# Counters and associative arrays
total_targets=0
pass_count=0
fail_count=0
declare -A func_times func_results

# Array to store function names for sorting
declare -A function_names

i=1
for d in "${SUBDIRS[@]}"; do
  total_targets=$((total_targets + 1))
  echo -e "========================================== Verifying [${i}/${total_count}] ${d} ======================================" | tee -a "$LOG"

  start_func=$(date +%s)
  output=$(cd "${TARGET_DIR}/$d" && make "${run_rule}" 2>&1)
  status=$?
  end_func=$(date +%s)
  duration=$((end_func - start_func))
  func_times["$d"]=$duration

  echo "$output" | tee -a "$LOG"

  if echo "$output" | grep -Eq "The program MAY NOT be correct|\.trace|Error\:|No rule to make target"; then
    func_results["$d"]="FAIL"
    fail_count=$((fail_count + 1))
    echo -e "\n$d: FAIL (took ${duration}s)" | tee -a "$LOG"
  else
    func_results["$d"]="PASS"
    pass_count=$((pass_count + 1))
    echo -e "\n$d: PASS (took ${duration}s)" | tee -a "$LOG"
  fi

  # Append function, time, and status to CSV
  echo "$d,$duration,${func_results[$d]}" >> "$CSV"

  # Store function names for sorting
  function_names["$d"]="$duration"
  
  echo "" | tee -a "$LOG"
  i=$((i + 1))
done

# Compute overall time
end_script=$(date +%s)
total_time=$((end_script - start_script))

# Final summary
{
  echo "============================================================================================================================"
  echo "Final Summary:"
  echo "Total Time Taken: ${total_time} second(s)"
  echo "Total Number of targets: ${total_targets}"
  echo "Number of targets Passed: ${pass_count}"
  echo "Number of targets Failed: ${fail_count}"
  echo ""
  echo "Individual Function Stats:"
  printf "  %-20s %-20s %s\n" "--------------------" "----------------------" "-------"
  printf "  %-20s %-20s   %s\n" "      Function"     "      Time Taken"         "Result"
  printf "  %-20s %-20s %s\n" "--------------------" "----------------------" "-------"

  for function in $(echo "${!function_names[@]}" | tr ' ' '\n' | sort); do
    printf "  %-20s : %-10d second(s) [%s]\n" \
      "$function" "${func_times[$function]}" "${func_results[$function]}"
  done
  echo "============================================================================================================================"
} | tee -a "$LOG"

echo "Detailed log saved to $LOG"
echo "CSV file with function times saved to $CSV"
