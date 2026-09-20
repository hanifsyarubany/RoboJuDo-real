#!/usr/bin/env bash
# Bring the whole onboard G1 stack up in ONE detached tmux session, one pane per process,
# all visible at once -- and re-attach to it with the same command after an SSH/LAN drop.
#
#   ./scripts/start_robot_stack.sh              # create + launch, then attach (or just attach if already up)
#   ./scripts/start_robot_stack.sh --system-setup   # also run the one-per-boot sudo steps first
#   ./scripts/start_robot_stack.sh --run-policy      # also press Enter on the policy pane (see WARNING)
#   ./scripts/start_robot_stack.sh --no-attach       # launch in the background, don't attach
#   ./scripts/start_robot_stack.sh --kill            # tear the whole session down
#
# WHY tmux: the processes become children of the tmux server, not of your SSH session, so
# unplugging the LAN neither SIGHUPs them nor wedges them on a stdout pipe nobody is draining
# (that second one is what caused the 45s/82s control-loop freezes -- see
# RlPipeline.stop_with_guard_pose / robojudo/utils/logger.py's QueueHandler comment).
#
# SAFETY: the policy pane (the one that physically moves the robot via prepare()'s torque ramp) is
# pre-typed but NOT executed unless --run-policy is passed. Everything else starts on its own; you
# press Enter on the last pane when you are actually ready for the robot to move.

set -euo pipefail

SESSION="${G1_TMUX_SESSION:-g1}"

# --- paths / envs (override by exporting these) -------------------------------------------------
HOME_DIR="${HOME}"
NECK_DIR="${G1_NECK_DIR:-$HOME_DIR/main_workspace/dynamixel-neck-test}"
BALL_WS_DIR="${G1_BALL_WS_DIR:-$HOME_DIR/main_workspace/ball_ws}"
ROBOJUDO_DIR="${G1_ROBOJUDO_DIR:-$HOME_DIR/main_workspace/RoboJuDo-real}"
CALIB_YAML="${G1_CALIB_YAML:-$HOME_DIR/checkerboard_livox_ransac_hold_pose/camera_calibpose_to_livox_robust.yaml}"

# --- settle delays between stages (seconds) -- these exist because the stack is a pipeline: -----
# livox -> detector -> fusion -> bridge -> policy, and the policy BLOCKS on its first ball reading.
D_AFTER_YOLO="${G1_D_AFTER_YOLO:-6}"
D_AFTER_TRACKER="${G1_D_AFTER_TRACKER:-4}"
D_AFTER_LIVOX="${G1_D_AFTER_LIVOX:-5}"
D_AFTER_DETECTOR="${G1_D_AFTER_DETECTOR:-4}"
D_AFTER_FUSION="${G1_D_AFTER_FUSION:-3}"
D_AFTER_BRIDGE="${G1_D_AFTER_BRIDGE:-2}"

NOBLAS='export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1'

RUN_POLICY=0
DO_ATTACH=1
DO_SYSTEM_SETUP=0
for arg in "$@"; do
    case "$arg" in
        --run-policy)    RUN_POLICY=1 ;;
        --no-attach)     DO_ATTACH=0 ;;
        --system-setup)  DO_SYSTEM_SETUP=1 ;;
        --kill)
            tmux kill-session -t "$SESSION" 2>/dev/null && echo "killed session '$SESSION'" \
                || echo "no session '$SESSION' to kill"
            exit 0 ;;
        -h|--help)       sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

# --- already running? then this invocation is just "re-attach after the LAN came back" ----------
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION' is already up -- attaching (nothing relaunched)."
    [[ "$DO_ATTACH" == 1 ]] && exec tmux attach -t "$SESSION"
    exit 0
fi

# --- one-per-boot system setup (needs sudo, so do it BEFORE tmux so any prompt is interactive) ---
if [[ "$DO_SYSTEM_SETUP" == 1 ]]; then
    echo "== system setup (sudo) =="
    sudo nvpmodel -m 0
    sudo jetson_clocks
    [[ -e /dev/ttyUSB0 ]] && sudo chmod 666 /dev/ttyUSB0
    [[ -e /sys/bus/usb-serial/devices/ttyUSB0/latency_timer ]] \
        && echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer >/dev/null
    sudo bash -c 'for i in 232 310 156 152 130; do
                      [ -e /proc/irq/$i/smp_affinity_list ] && echo 0 > /proc/irq/$i/smp_affinity_list
                  done' || true
    echo "(gdm not stopped -- run 'sudo systemctl stop gdm' yourself if you want the GPU/CPU back)"
fi

# --- build the session --------------------------------------------------------------------------
FIRST=1
launch() {  # launch <pane-title> <settle-seconds> <command>
    local title="$1" settle="$2" cmd="$3" pane
    if [[ "$FIRST" == 1 ]]; then
        pane=$(tmux new-session -d -s "$SESSION" -n stack -P -F '#{pane_id}')
        FIRST=0
    else
        pane=$(tmux split-window -t "$SESSION:stack" -P -F '#{pane_id}')
        tmux select-layout -t "$SESSION:stack" tiled >/dev/null
    fi
    tmux select-pane -t "$pane" -T "$title"
    # Trailing || ... so a pane whose command never starts (bad path, missing env, typo) SAYS so
    # instead of just sitting at a clean prompt looking indistinguishable from a healthy one --
    # in a 7-pane tiled view a silently-dead process is very easy to miss. Also fires when you
    # Ctrl+C something, which is the same information you want.
    tmux send-keys -t "$pane" "{ $cmd ; } || echo '*** $title EXITED / FAILED TO START (see above) ***'" C-m
    echo "  [$title] started; settling ${settle}s"
    sleep "$settle"
}

echo "== bringing up '$SESSION' =="

launch yolo "$D_AFTER_YOLO" \
    "cd $HOME_DIR && conda activate yolo_zed && taskset -c 4-5 python3 yolo_360p_yolo11n_VGA100_optimized.py --publish-data --cpp-infer"

launch neck-tracker "$D_AFTER_TRACKER" \
    "cd $NECK_DIR && conda activate yolo_zed && taskset -c 6-7 python3 ball_tracker_projection_optimized_neck_bent_to_pelvis_optional.py --camera-livox-calib $CALIB_YAML --publish-base-heading --publish-pelvis --enable-servo-explore --confirm"

launch livox "$D_AFTER_LIVOX" \
    "taskset -c 6-7 ros2 launch livox_ros_driver2 msg_MID360_launch.py"

launch ball-detector "$D_AFTER_DETECTOR" \
    "conda deactivate 2>/dev/null; $NOBLAS && cd $BALL_WS_DIR && taskset -c 6-7 python3 scripts/40_ball_detector_node.py"

launch ball-fusion "$D_AFTER_FUSION" \
    "conda deactivate 2>/dev/null; $NOBLAS && cd $BALL_WS_DIR && taskset -c 6-7 python3 scripts/50_ball_fusion_node.py"

launch udp-bridge "$D_AFTER_BRIDGE" \
    "conda deactivate 2>/dev/null; cd $ROBOJUDO_DIR && taskset -c 6-7 /usr/bin/python3 scripts/foxy_ros2_ball_bridge.py"

# --- policy pane: pre-typed, Enter withheld unless --run-policy ----------------------------------
POLICY_CMD="cd $ROBOJUDO_DIR && conda activate robojudo && taskset -c 1-3 python scripts/run_pipeline_prepared.py -c g1_unified_loco_kick --live-ball --ball-source udp --ball-log-hz 1.0 --ball-log-decimals 3 --guard-arm-kp 40 --guard-arm-kd 4"
policy_pane=$(tmux split-window -t "$SESSION:stack" -P -F '#{pane_id}')
tmux select-layout -t "$SESSION:stack" tiled >/dev/null
tmux select-pane -t "$policy_pane" -T "POLICY"
if [[ "$RUN_POLICY" == 1 ]]; then
    tmux send-keys -t "$policy_pane" "$POLICY_CMD" C-m
    echo "  [POLICY] STARTED (--run-policy) -- robot will ramp to default pose"
else
    tmux send-keys -t "$policy_pane" "$POLICY_CMD"   # no C-m: operator presses Enter
    echo "  [POLICY] pre-typed, NOT started -- focus that pane and press Enter when ready"
fi
tmux select-pane -t "$policy_pane"

# Readable pane titles + scrollback + mouse, scoped to THIS session only. Deliberately not `set -g`:
# that writes tmux's GLOBAL options, which would silently reconfigure every other session on the
# same tmux server (and persist after this one is killed).
tmux set-option -t "$SESSION" mouse on >/dev/null 2>&1 || true
tmux set-option -t "$SESSION" history-limit 50000 >/dev/null 2>&1 || true
tmux set-option -w -t "$SESSION:stack" pane-border-status top >/dev/null 2>&1 || true
tmux set-option -w -t "$SESSION:stack" pane-border-format ' #{pane_index}:#{pane_title} ' >/dev/null 2>&1 || true

cat <<EOF

'$SESSION' is up: 7 panes, tiled, all in one window.
  attach / re-attach after a LAN drop :  $0
  detach                              :  Ctrl+B then D
  zoom one pane full-screen / back    :  Ctrl+B then Z
  move between panes                  :  Ctrl+B then arrow keys
  tear everything down                :  $0 --kill
EOF

[[ "$DO_ATTACH" == 1 ]] && exec tmux attach -t "$SESSION"
exit 0
