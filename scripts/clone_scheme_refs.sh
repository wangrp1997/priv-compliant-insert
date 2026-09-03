#!/usr/bin/env bash
# Clone/pull reference repos for tip-tracking / extrinsic dexterity schemes.
set -euo pipefail

REFS="/home/wangrenpeng/priv_compliant_insert/refs"
mkdir -p "$REFS"

# nros proxy
if [ -f /opt/nros_tools/nros_rc ]; then
  # shellcheck disable=SC1091
  source /opt/nros_tools/nros_rc
fi
if declare -F nros-proxy-on >/dev/null 2>&1; then
  nros-proxy-on || true
fi

clone_or_pull() {
  local url="$1"
  local dir="$2"
  if [ -d "$dir/.git" ]; then
    echo "[pull] $dir"
    git -C "$dir" pull --ff-only 2>&1 | tail -3 || git -C "$dir" fetch --depth 1 origin 2>&1 | tail -2
  elif [ -d "$dir" ]; then
    echo "[skip-non-git] $dir"
  else
    echo "[clone] $url -> $dir"
    git clone --depth 1 "$url" "$dir" 2>&1 | tail -5
  fi
}

export -f clone_or_pull
export REFS

# scheme repos (parallel)
repos=(
  "https://github.com/swri-robotics/ConnTact|ConnTact"
  "https://github.com/sangwkim/Tactile-Estimator-Controller|Tactile-Estimator-Controller"
  "https://github.com/Director-of-G/in_hand_manipulation_2|in_hand_manipulation_2"
  "https://github.com/ZhengtongXu/LeTac-MPC|LeTac-MPC"
  "https://github.com/imanlab/action_conditioned_tactile_prediction|action_conditioned_tactile_prediction"
  "https://github.com/imanlab/bgf|bgf"
  "https://github.com/Gabrieleenx/Slip-Aware-Object-Manipulation-with-Parallel-Grippers|Slip-Aware-Object-Manipulation"
  "https://github.com/benjaminalt/dpse|dpse"
  "https://github.com/avinash246813579/franka-peg-in-hole|franka-peg-in-hole"
  "https://github.com/ir-lab/irl_control|irl_control"
)

printf '%s\n' "${repos[@]}" | xargs -P 6 -I{} bash -c '
  IFS="|" read -r url name <<< "{}"
  clone_or_pull "$url" "'"$REFS"'/$name"
'

echo "DONE clone/pull under $REFS"
