# Ref: Lecture 10 - feedback control.pdf, Page 7, 46 (Disturbance Handling)
# Theory: Constant disturbances like wind cause steady-state error in P-only controllers.
# Why: Setting WIND=True widens the PID integrator limits so the drone can accumulate
#      enough integral action to cancel out the persistent wind-induced position offset.
WIND = True
# WIND FLAG – must remain on Line 1 as required by the coursework marking script.

# =============================================================================
# AERO60492 Coursework 3 – Feedback Control
# Student    : Prathap Selvakumar  |  ID: 14354077
# Advanced method: Q-learning Reinforcement Learning with PID Gain Scheduling
#
# NOTE ON AI USE:
#   AI language models were used to assist with grammar and vocabulary in
#   comments only. All code logic, algorithm design, and implementation
#   are entirely the author's own work.
#
# NOTE ON IMPORTS:
#   Only Python standard library modules are used. No third-party packages
#   beyond matplotlib (already a simulator dependency) are imported.
# =============================================================================

# =============================================================================
# MODULE IMPORTS
# =============================================================================
import atexit           # Registers _FlightLogger._export() to run on interpreter exit,
                        # guaranteeing the CSV is saved even if the user presses Q.
import csv              # Reads target waypoints from targets.csv; writes flight log rows.
import math             # Provides sin, cos, atan2, sqrt for coordinate transforms and
                        # distance calculations throughout the controller.
import os               # Creates the logs/ directory and resolves file paths portably
                        # across Windows and Linux (university cluster support).
import random           # Provides the seeded RNG (random.Random(42)) used for
                        # epsilon-greedy exploration and experience replay sampling.
                        # Seeding with 42 makes results reproducible across runs.
from datetime import datetime  # Timestamps log filenames (YYYYMMDD_HHMMSS) so each
                                # flight produces a unique file and never overwrites previous data.

# =============================================================================
# MATPLOTLIB – optional, guarded against headless/marker machines
# =============================================================================
# Why "Agg" instead of the default "TkAgg"?
#   The previous version spawned a daemon thread that called plt.figure() and
#   plt.pause(). On Windows, Tk GUI windows MUST be created on the main thread.
#   The daemon thread violated this rule, causing a hard crash in the simulation
#   loop. "Agg" is a file-only backend: it renders entirely in memory and saves
#   to PNG on exit, with no GUI window and no thread restrictions whatsoever.
#   If matplotlib is absent on the marker's machine, _MATPLOTLIB_OK = False and
#   all plot calls are silently skipped — the controller never crashes.
# Ref: matplotlib documentation – backend selection.
try:
    import matplotlib
    matplotlib.use("Agg")               # File-only backend: renders to PNG, no GUI window.
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – required for 3-D projection.
    _MATPLOTLIB_OK = True               # matplotlib loaded successfully; plots will be saved.
except Exception:
    _MATPLOTLIB_OK = False              # matplotlib unavailable; plots silently skipped.

# =============================================================================
# CONTROL ACTION PROFILES – The RL Agent's 16 Discrete Actions
# =============================================================================
# WHY GAIN SCHEDULING VIA RL?
#   A single fixed PID tuning is a compromise. Gains tuned for fast long-range
#   approach (high kp, low kd) cause overshoot and oscillation when the drone
#   reaches the target. Gains tuned for precise hover (low kp, high kd) are
#   far too slow to cross 4 metres within the 10-second marking window.
#   Instead, the Q-learning agent selects one of these 16 profiles at every
#   20 Hz timestep. Over time it learns: use aggressive profiles when far away,
#   use gentle profiles when close. This is "gain scheduling via RL".
#   Ref: Lecture 10 – Advanced Control, gain scheduling rationale.
#
# COLUMN ORDER (all 11 values per profile):
#   kp_xy        – proportional gain for X/Y velocity commands.
#                  Ref: Lecture 10, p.8 – P term.
#   kp_z         – proportional gain for altitude (Z) velocity command.
#   kp_yaw       – proportional gain for yaw rate command.
#                  Raised to 1.20-1.50 in fine/mid profiles after testing
#                  revealed slow yaw convergence with lower values.
#   ki_xy        – integral gain for X/Y. Zero in coarse/mid profiles (0-7)
#                  because at long range the large persistent error would cause
#                  integral windup, producing a massive velocity spike on arrival.
#                  Small non-zero (0.01) in fine profiles (8-11) to remove the
#                  small residual bias that accumulates in the last 0.5 m.
#                  Ref: Lecture 10, p.10 – I term eliminates steady-state error.
#   ki_z         – integral gain for altitude. Raised to 0.05 in hover profiles
#                  (12-15) to cancel the ~3 cm gravity/wind droop observed during
#                  steady-state hover testing.
#   ki_yaw       – integral gain for yaw. Zero in all hover profiles (12-15)
#                  to prevent a slowly accumulating yaw integral from spinning
#                  the drone, which would violate the yaw_std < 0.001 rad criterion.
#   kd_xy        – derivative gain for X/Y. Acts as a velocity damper.
#                  TUNING FIX: Originally 0.50 in coarse profiles; at 1 m/s
#                  approach speed the derivative term was generating -0.50 m/s
#                  of counter-velocity, cutting effective speed in half 3 m from
#                  the target. Lowered to 0.10 in coarse profiles, raised to
#                  0.32-0.40 in fine profiles where braking is needed.
#                  Ref: Lecture 10, p.12 – D term damps rate of error change.
#   kd_z         – derivative gain for altitude.
#   v_xy_max     – hard velocity cap for X/Y commands (m/s).
#                  run.py clips all outputs to ±1.0 m/s at hardware level.
#   v_z_max      – hard velocity cap for Z (altitude) command (m/s).
#   yaw_rate_max – hard cap for yaw rate command (rad/s).
#                  run.py clips to ±1.745 rad/s (±100 deg/s) at hardware level.
#
# PROFILE TIERS:
#   0- 3  COARSE     dist > 1.5 m  : high kp, near-zero kd, zero ki, fast caps.
#   4- 7  MID-RANGE  0.5-1.5 m     : moderate kp/kd, zero ki, medium caps.
#   8-11  FINE       < 0.5 m       : high kd for braking, small ki, tight caps.
#  12-15  HOVER-HOLD < 0.08 m      : low kp, high kd, small ki, tiny caps.
# =============================================================================
_ACTION_PROFILES = (
    # -- Coarse approach (dist > 1.5 m) ----------------------------------------
    # kp_xy=1.20 saturates the ±1 m/s hardware cap immediately at long range.
    # kd_xy=0.10 provides minimal braking so the drone maintains full approach speed.
    # ki_xy=0.00 prevents integral windup during the long transit leg.
    (1.20, 1.10, 0.80, 0.00, 0.00, 0.03, 0.10, 0.10, 2.50, 1.50, 1.40),  # 0: Max speed
    (1.10, 1.00, 0.75, 0.00, 0.00, 0.03, 0.10, 0.10, 2.00, 1.20, 1.20),  # 1: Very fast
    (1.00, 0.90, 0.70, 0.00, 0.00, 0.04, 0.10, 0.10, 1.50, 1.00, 1.00),  # 2: Fast
    (0.90, 0.80, 0.65, 0.00, 0.00, 0.04, 0.10, 0.10, 1.20, 0.80, 0.90),  # 3: Moderate fast

    # -- Mid-range (0.5 m – 1.5 m) ---------------------------------------------
    # kp_xy raised (0.40→0.85), kd_xy lowered (0.60→0.20).
    # Drone stays at full speed until 1.5 m, then begins a controlled deceleration.
    # kp_yaw raised to 1.20 to correct heading while still closing the distance.
    (0.85, 0.75, 1.20, 0.00, 0.00, 0.05, 0.20, 0.15, 0.80, 0.60, 1.40),  # 4: Transition start
    (0.75, 0.65, 1.00, 0.00, 0.00, 0.05, 0.20, 0.15, 0.70, 0.55, 1.20),  # 5: Controlled
    (0.65, 0.55, 0.85, 0.00, 0.00, 0.06, 0.20, 0.15, 0.60, 0.50, 1.00),  # 6: Slowing
    (0.55, 0.45, 0.70, 0.00, 0.00, 0.06, 0.20, 0.15, 0.50, 0.45, 0.85),  # 7: Near fine zone

    # -- Fine approach (< 0.5 m) -----------------------------------------------
    # kd_xy raised (0.32-0.40) to actively brake in the final 0.5 m.
    # ki_xy=0.01 eliminates the small residual XY bias to achieve dist < 0.01 m.
    # kp_yaw=1.50 locks heading before the dwell phase begins.
    # Tight v_xy_max (0.12-0.25 m/s) prevents flying past the target.
    (0.45, 0.45, 1.50, 0.01, 0.01, 0.09, 0.40, 0.30, 0.25, 0.20, 1.50),  # 8:  High damping
    (0.40, 0.40, 1.40, 0.01, 0.01, 0.10, 0.38, 0.28, 0.20, 0.15, 1.40),  # 9:  Smooth brake
    (0.35, 0.35, 1.30, 0.01, 0.01, 0.11, 0.35, 0.25, 0.15, 0.12, 1.30),  # 10: Very slow
    (0.30, 0.30, 1.20, 0.01, 0.01, 0.12, 0.32, 0.22, 0.12, 0.10, 1.20),  # 11: Creeping

    # -- Hover-hold (< 0.08 m) -------------------------------------------------
    # Very small velocity caps (0.04-0.10 m/s) keep the drone stationary.
    # ki_z=0.05 (raised from 0.02) fully cancels gravity/wind altitude droop.
    # ki_yaw=0.00 prevents slow yaw drift over the 10 s measurement window.
    # High kd (0.40-0.55) is the primary stabilising force in hover mode.
    (0.35, 0.35, 0.50, 0.02, 0.05, 0.00, 0.55, 0.45, 0.10, 0.10, 0.25),  # 12: Firm hold
    (0.30, 0.30, 0.45, 0.02, 0.05, 0.00, 0.50, 0.42, 0.08, 0.08, 0.20),  # 13: Gentle hold
    (0.25, 0.25, 0.40, 0.02, 0.05, 0.00, 0.45, 0.40, 0.06, 0.06, 0.15),  # 14: Soft hold
    (0.20, 0.20, 0.35, 0.02, 0.05, 0.00, 0.40, 0.38, 0.04, 0.04, 0.10),  # 15: Minimal hold
)

# =============================================================================
# TOLERANCE AND REWARD CONSTANTS
# =============================================================================
_POS_TOL    = 0.08   # m   – drone must be within this 3-D radius to start dwell count.
_YAW_TOL    = 0.06   # rad – yaw must also be within tolerance simultaneously.
_HOLD_STEPS = 20     # steps – 20 × 0.02 s = 0.4 s of continuous holding required.

# All RL reward shaping coefficients in one place for transparency and easy tuning.
# Naming convention: all keys are lowercase strings describing the reward component.
# Ref: Lecture 10 – reward shaping rationale.
_REWARD_WEIGHTS = {
    "overshoot":      1500.0,  # Heavy penalty when close but moving away (prevents bouncing).
    "effort":         0.015,   # Light penalty for large velocity commands (discourages thrashing).
    "distance":       2.0,     # Continuous penalty proportional to dist² (drives to zero error).
    "dwell_enter":    20.0,    # Bonus on first step inside tolerance (rewards arrival).
    "dwell_complete": 30.0,    # Bonus after sustained hold (rewards stability).
}

# =============================================================================
# DATA LOGGER
# =============================================================================
class _FlightLogger:
    """
    Lightweight per-call telemetry recorder with automatic CSV and PNG export on exit.

    Design rationale:
        All data is accumulated in an in-memory list (_rows) during flight.
        No file I/O occurs during the 20 Hz control loop, so latency is zero.
        A single atexit hook flushes everything to disk when the simulator exits.
        A timestamped filename ensures flights never overwrite each other.
    """

    _LOG_DIR = "logs"   # All output files written here (created if absent).

    def __init__(self):
        self._rows            = []            # In-memory buffer: one tuple per controller() call.
        self._t               = 0.0           # Cumulative simulation time (sum of all dt values).
        self._fname           = None          # CSV path; set on first log() call to freeze timestamp.
        self._xs              = []            # X position history for post-flight path PNG.
        self._ys              = []            # Y position history.
        self._zs              = []            # Z position history.
        self._last_target_xyz = (2.0, 2.0, 2.0)  # Last recorded target XYZ for PNG annotation.
        atexit.register(self._export)         # Guarantee export even on abnormal exit.

    def log(self, dt, pos, euler, vel_cmd, target, wind, profile_idx):
        """
        Append one telemetry row per controller() call.

        Parameters
        ----------
        dt          : float – timestep (s).
        pos         : seq   – (x, y, z) world-frame position (m).
        euler       : seq   – (roll, pitch, yaw) Euler angles (rad).
        vel_cmd     : seq   – (vx, vy, vz) velocity COMMANDS output by controller (m/s).
                              Named vel_cmd (not vel_world) because these are commanded
                              outputs, not the actual pybullet world velocities.
        target      : seq   – (tx, ty, tz, tyaw) active setpoint.
        wind        : bool  – wind disturbance flag.
        profile_idx : int   – index 0-15 of the RL-selected gain profile.
        """
        if self._fname is None:
            os.makedirs(self._LOG_DIR, exist_ok=True)
            stamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._fname = os.path.join(self._LOG_DIR, f"flight_{stamp}.csv")

        self._t += dt

        x,  y,  z        = (float(v) for v in pos)
        roll, pitch, yaw = (float(v) for v in euler)
        vx_cmd, vy_cmd, vz_cmd = (float(v) for v in vel_cmd)  # _cmd suffix: velocity commands.
        tx, ty, tz, tyaw = (float(v) for v in target)

        # Derived metrics for post-flight analysis.
        dist_log    = math.sqrt((tx-x)**2 + (ty-y)**2 + (tz-z)**2)
        yaw_err_log = math.atan2(math.sin(tyaw - yaw), math.cos(tyaw - yaw))
        speed_log   = math.sqrt(vx_cmd**2 + vy_cmd**2 + vz_cmd**2)
        # Naming: *_log suffix marks variables computed only for logging,
        # distinguishing them from the control variables vx/vy/vz in controller().

        self._rows.append((
            round(self._t,       4),
            round(x,        4),  round(y,           4),  round(z,        4),
            round(roll,     4),  round(pitch,        4),  round(yaw,      4),
            round(vx_cmd,   4),  round(vy_cmd,       4),  round(vz_cmd,   4),
            round(tx,       4),  round(ty,           4),  round(tz,       4),  round(tyaw, 4),
            round(dist_log, 4),  round(yaw_err_log,  4),  round(speed_log,4),
            int(wind), profile_idx,
        ))

        self._xs.append(x)
        self._ys.append(y)
        self._zs.append(z)
        self._last_target_xyz = (tx, ty, tz)

    def _export(self):
        """Write accumulated rows to CSV; optionally save a 3-D path PNG via Agg."""
        if not self._rows or self._fname is None:
            return

        _COLS = [
            "time_s", "x", "y", "z",
            "roll_rad", "pitch_rad", "yaw_rad",
            "vx_cmd", "vy_cmd", "vz_cmd",
            "target_x", "target_y", "target_z", "target_yaw",
            "dist_m", "yaw_err_rad", "speed_ms",
            "wind_enabled", "rl_profile",
        ]
        try:
            with open(self._fname, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(_COLS)
                w.writerows(self._rows)
            print(f"\nINFO: Flight log saved -> {self._fname}  ({len(self._rows)} rows)")
        except Exception as exc:
            print(f"\nWARN: Could not save flight log: {exc}")

        if _MATPLOTLIB_OK and len(self._xs) > 1:
            try:
                png_path = self._fname.replace(".csv", "_path.png")
                fig = plt.figure(figsize=(7, 6))
                ax  = fig.add_subplot(111, projection="3d")
                ax.plot(self._xs, self._ys, self._zs, "b-", lw=1, alpha=0.6, label="Path")
                ax.plot([self._xs[-1]], [self._ys[-1]], [self._zs[-1]], "ro", ms=8, label="Final pos")
                ax.plot([self._last_target_xyz[0]],
                        [self._last_target_xyz[1]],
                        [self._last_target_xyz[2]], "g*", ms=14, label="Target")
                ax.plot([self._xs[-1],            self._xs[-1]],
                        [self._ys[-1],            self._ys[-1]],
                        [0,                       self._zs[-1]], "r--", lw=0.5, alpha=0.3)
                ax.plot([self._last_target_xyz[0], self._last_target_xyz[0]],
                        [self._last_target_xyz[1], self._last_target_xyz[1]],
                        [0,                        self._last_target_xyz[2]], "g--", lw=0.5, alpha=0.3)
                ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
                ax.set_title("Flight Path – RL+PID Controller")
                ax.legend(loc="upper left", fontsize=8)
                fig.tight_layout()
                fig.savefig(png_path, dpi=120)
                plt.close(fig)
                print(f"INFO: Path plot saved  -> {png_path}")
            except Exception as exc:
                print(f"WARN: Could not save path PNG: {exc}")


_logger = _FlightLogger()


# =============================================================================
# MATHEMATICAL HELPERS
# =============================================================================

def _clip(v, lo, hi):
    """Saturate value v to the closed interval [lo, hi]. Used for anti-windup and velocity caps."""
    return max(lo, min(hi, v))


def _state_key(dist, abs_yaw_err, wind):
    """
    Discretise the continuous (distance, yaw_error, wind) state into a hashable
    tuple that indexes the Q-table.

    Naming convention:
        dist        – 3-D Euclidean distance to target (m).
        abs_yaw_err – absolute yaw error (rad); always non-negative.
        wind        – bool converted to int (False→0, True→1).

    Returns (dist_band, yaw_band, wind_int) – all integers.
    96 total states: 8 dist_band × 6 yaw_band × 2 wind_int.
    Ref: Lecture 10 – state discretisation for tabular RL.
    """
    # dist_band: finer resolution close to target where precision matters.
    if   dist < 0.08: dist_band = 0
    elif dist < 0.20: dist_band = 1
    elif dist < 0.40: dist_band = 2
    elif dist < 0.70: dist_band = 3
    elif dist < 1.10: dist_band = 4
    elif dist < 1.80: dist_band = 5
    elif dist < 2.80: dist_band = 6
    else:             dist_band = 7

    # yaw_band: finer resolution near zero (aligned).
    if   abs_yaw_err < 0.04: yaw_band = 0
    elif abs_yaw_err < 0.10: yaw_band = 1
    elif abs_yaw_err < 0.25: yaw_band = 2
    elif abs_yaw_err < 0.50: yaw_band = 3
    elif abs_yaw_err < 1.00: yaw_band = 4
    else:                    yaw_band = 5

    wind_int = int(wind)   # False→0, True→1.
    return (dist_band, yaw_band, wind_int)


# =============================================================================
# CSV WAYPOINT LOADER
# =============================================================================
<<<<<<< HEAD

def _load_csv_targets(csv_path=None):
    """
    Load mission waypoints from targets.csv (free-fly development mode only).
    Falls back to the hard-coded four-corner square on any error.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "targets.csv")

=======
def _load_csv_targets(csv_path=None):
    """
    Loads a sequence of target waypoints from a CSV file.
    Provides a default square pattern if the file is missing or invalid.
    
    Args:
        csv_path (str): Path to the CSV file containing target coordinates.
                        If None, defaults to 'targets.csv' in the same directory as this file.
        
    Returns:
        tuple: A tuple of target tuples: ((x1, y1, z1, yaw1), (x2, y2, z2, yaw2), ...)
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "targets.csv")
    # Default fallback flight pattern (a simple 2x2 square at 2m altitude)
>>>>>>> 07d710f5bd881e1f2b5093dca36bf5f9b74103d5
    _DEFAULTS = (
        ( 2.0,  2.0, 2.0, 0.0),
        (-2.0,  2.0, 2.0, 1.57),
        (-2.0, -2.0, 2.0, 3.14),
        ( 2.0, -2.0, 2.0, 4.71),
    )

    if not os.path.isfile(csv_path):
        return _DEFAULTS

    try:
        targets = []
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            fields = [f.strip() for f in (reader.fieldnames or [])]

            def _col(axis):
                for c in (f"target_{axis}", axis):
                    if c in fields:
                        return c
                raise ValueError(f"No '{axis}' column in {fields}")

            cx, cy, cz = _col("x"), _col("y"), _col("z")
            cyaw = next(
                (c for c in ("target_yaw", "yaw", "heading", "psi") if c in fields),
                None,
            )
            if cyaw is None:
                raise ValueError("No yaw column found")

            for i, raw in enumerate(reader):
                row = {k.strip(): v.strip() for k, v in raw.items()}
                try:
                    targets.append((float(row[cx]), float(row[cy]),
                                    float(row[cz]), float(row[cyaw])))
                except (KeyError, ValueError) as e:
                    raise ValueError(f"Bad row {i+2}: {row}") from e

        return tuple(targets) if targets else _DEFAULTS
    except Exception:
        return _DEFAULTS


_CSV_TARGETS = _load_csv_targets()


# =============================================================================
# RL AGENT FUNCTIONS
# =============================================================================

def _get_errors(state, active_target):
    """
    [ADVANCED METHOD – State Estimator & Coordinate Frame Transformation]

    Computes all tracking errors needed by the PID controller and the RL cost.

    NAMING CONVENTION used throughout this function and all callers:
        ex_w, ey_w  – position errors in the World (inertial) frame (m).
        ex_b, ey_b  – position errors in the Body frame (m), after R_z(yaw).
        ez          – altitude error; identical in both frames for level flight.
        eyaw        – yaw error wrapped to (-π, π] (rad).
        dist        – 3-D Euclidean distance to target (m).
        cost        – scalar RL cost used to compute reward signal.

    WHY WORLD AND BODY FRAME ERRORS?
        PID integration and differentiation use world-frame errors (ex_w, ey_w)
        to avoid false spikes when the drone yaws. Body-frame errors (ex_b, ey_b)
        are stored in the return dict for reference but the main PID uses world frame.
        The final velocity commands are rotated back to body frame in controller()
        Step 8 before being returned to run.py.
        Ref: Lecture 6, p.14 – R_z(ψ) rotation matrix.

    ROTATION MATRIX (yaw only):
        [ ex_b ]   [ cos(yaw)  sin(yaw) ] [ ex_w ]
        [ ey_b ] = [-sin(yaw)  cos(yaw) ] [ ey_w ]

    COST FUNCTION:
        cost = dist² + 0.60·eyaw² + prec
        prec = 5·max(0, 0.10 − dist)²  (precision bonus for sub-10 cm accuracy)
        Ref: Lecture 10, p.15 – reward shaping.
    """
    x, y, z, _, _, yaw = (float(v) for v in state)
    tx, ty, tz, tyaw   = (float(v) for v in active_target)

    # World-frame position errors.
    ex_w = tx - x
    ey_w = ty - y
    ez   = tz - z

    # Yaw error wrapped to (-π, π]. Ref: Lecture 6, p.11 – angle wrapping.
    eyaw = math.atan2(math.sin(tyaw - yaw), math.cos(tyaw - yaw))

    # Body-frame XY errors via 2×2 yaw rotation matrix.
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    ex_b =  cos_yaw * ex_w + sin_yaw * ey_w
    ey_b = -sin_yaw * ex_w + cos_yaw * ey_w

    # RL cost function.
    dist = math.sqrt(ex_w**2 + ey_w**2 + ez**2)
    prec = 5.0 * max(0.0, 0.10 - dist) ** 2
    cost = dist**2 + 0.60 * eyaw**2 + prec

    # All keys use the naming convention documented above.
    return {
        "dist": dist, "cost": cost,
        "eyaw": eyaw, "ez":   ez,
        "ex_w": ex_w, "ey_w": ey_w,   # World-frame: used by PID integral/derivative.
        "ex_b": ex_b, "ey_b": ey_b,   # Body-frame:  stored for reference.
    }


def _td_update(mem, state_key, action_idx, reward, next_state_key):
    """
    [ADVANCED METHOD – Q-learning Temporal Difference Update]

    Naming convention for parameters:
        state_key      – hashable tuple (dist_band, yaw_band, wind_int) for current state.
        action_idx     – integer 0-15 index of the action taken.
        reward         – scalar reward received after the action.
        next_state_key – hashable tuple for the resulting next state.
        q_curr         – Q-value list for the current state (16 entries).
        q_next         – Q-value list for the next state (16 entries).

    UPDATE RULE (Ref: Lecture 10, p.19):
        Q(state_key, action_idx) +=
            alpha × [ reward + gamma × max(q_next) − Q(state_key, action_idx) ]
    """
    q_curr = mem["q_table"].setdefault(state_key,      [0.0] * len(_ACTION_PROFILES))
    q_next = mem["q_table"].setdefault(next_state_key, [0.0] * len(_ACTION_PROFILES))
    q_curr[action_idx] += mem["alpha"] * (
        reward + mem["gamma"] * max(q_next) - q_curr[action_idx]
    )


def _replay_update(mem):
    """
    [ADVANCED METHOD – Experience Replay Mini-Batch Update]

    Samples mem["replay_batch"] past transitions from the replay buffer and
    applies _td_update to each, using the same parameter naming convention:
        state_key, action_idx, reward, next_state_key.
    Ref: Lecture 10, p.21 – experience replay.
    """
    if len(mem["replay"]) < mem["replay_batch"]:
        return
    for state_key, action_idx, reward, next_state_key in \
            mem["rng"].sample(mem["replay"], mem["replay_batch"]):
        _td_update(mem, state_key, action_idx, reward, next_state_key)


def _update_learning_agent(mem, errs, dwell_bonus, state_key):
    """
    [ADVANCED METHOD – Online Q-learning with Reward Shaping]

    Naming convention:
        errs         – error dict from _get_errors (keys: dist, cost, eyaw, etc.).
        dwell_bonus  – float bonus from _handle_dwell (0.0 if not within tolerance).
        state_key    – current discretised state (dist_band, yaw_band, wind_int).
        reward       – total shaped reward for the PREVIOUS action.
        overshoot_penalty – penalty when close but cost is increasing (drone overshooting).

    All reward weight values come from _REWARD_WEIGHTS for full transparency.
    Update is throttled to every 5th step to save CPU for pybullet physics server.
    Ref: Lecture 10, p.18-21 – reward shaping, TD(0), experience replay.
    """
    if mem["prev_key"] is None:
        return   # No previous transition on the very first call.

    # Overshoot penalty: cost increase when close means the drone is moving away.
    overshoot_penalty = (
        _REWARD_WEIGHTS["overshoot"] * (errs["cost"] - mem["prev_cost"])
        if (errs["dist"] < 0.6 and errs["cost"] > mem["prev_cost"])
        else 0.0
    )

    reward = (
        (mem["prev_cost"] - errs["cost"])                          # Cost improvement.
        + dwell_bonus                                              # Arrival/hold bonus.
        - _REWARD_WEIGHTS["effort"]   * mem["prev_effort"]         # Effort penalty.
        - _REWARD_WEIGHTS["distance"] * errs["dist"] ** 2          # Distance penalty.
        - overshoot_penalty                                        # Overshoot penalty.
    )

    if mem["step_count"] % 5 == 0:
        _td_update(mem, mem["prev_key"], mem["prev_action"], reward, state_key)
        mem["replay"].append((mem["prev_key"], mem["prev_action"], reward, state_key))
        if len(mem["replay"]) > mem["replay_cap"]:
            mem["replay"].pop(0)
        _replay_update(mem)

    mem["epsilon"] = max(mem["eps_floor"], mem["epsilon"] * mem["eps_decay"])


def _handle_dwell(mem, errs, target_pos):
    """
    Manage the dwell-hold phase: count consecutive in-tolerance steps, award RL
    bonuses, reset integrators, and advance the waypoint index in free-fly mode.

    Naming convention:
        target_pos   – the raw argument from controller() (None in free-fly mode).
                       Named identically to the controller() parameter to make the
                       pass-through explicit: controller receives target_pos from
                       run.py and passes it here unchanged.
        dwell_bonus  – float returned to controller() and forwarded to
                       _update_learning_agent() as dwell_bonus.
        is_within    – bool: True if BOTH dist and yaw are inside tolerance.

    Ref: Coursework spec – 10 s stabilisation window; Lecture 10, p.16.
    """
    is_within   = errs["dist"] < _POS_TOL and abs(errs["eyaw"]) < _YAW_TOL
    dwell_bonus = 0.0

    if is_within:
        mem["dwell_steps"] += 1
        if mem["dwell_steps"] == 1:
            dwell_bonus = _REWARD_WEIGHTS["dwell_enter"]
        elif mem["dwell_steps"] >= _HOLD_STEPS:
            dwell_bonus         = _REWARD_WEIGHTS["dwell_complete"]
            mem["dwell_steps"]  = 0
            mem["integral"]     = [0.0, 0.0, 0.0]
            mem["integral_yaw"] = 0.0
            if target_pos is None:
                mem["wp_idx"] = (mem["wp_idx"] + 1) % len(_CSV_TARGETS)
    else:
        mem["dwell_steps"] = 0

    return dwell_bonus


def _choose_action(mem, state_key, force_exploit=False, mask_coarse=False, mask_fine=False):
    """
    [ADVANCED METHOD – ε-greedy Policy with Bi-directional Safety Masking]

    Naming convention:
        state_key    – current Q-table lookup key (dist_band, yaw_band, wind_int).
        q_vals       – raw Q-values for state_key from the Q-table (list of 16 floats).
        q_masked     – Q-values after safety masking (-1e9 for forbidden actions).
        action_idx   – integer 0-15 index of the selected action profile.
        eps          – effective epsilon (0.0 if force_exploit, else mem["epsilon"]).
        best_q       – maximum value in q_masked (used for greedy selection).
        tied_actions – list of action indices that all share best_q (tie-breaking).

    SAFETY MASKING (Ref: Lecture 10, p.22):
        mask_coarse (dist < 0.1 m): blocks profiles 0-3 (high-speed).
        mask_fine   (dist > 1.5 m): blocks profiles 8-15 (slow hover).
        Forbidden entries set to -1e9 so argmax never selects them.
    """
    q_vals = mem["q_table"].setdefault(state_key, [0.0] * len(_ACTION_PROFILES))

    if mask_coarse:
        q_masked = [v if i >= 4 else -1e9 for i, v in enumerate(q_vals)]
    elif mask_fine:
        q_masked = [v if i < 8  else -1e9 for i, v in enumerate(q_vals)]
    else:
        q_masked = q_vals

    eps = 0.0 if force_exploit else mem["epsilon"]

    if mem["rng"].random() < eps:
        if mask_coarse: return mem["rng"].randrange(4, 16)
        if mask_fine:   return mem["rng"].randrange(0, 8)
        return mem["rng"].randrange(0, 16)

    best_q       = max(q_masked)
    tied_actions = [i for i, v in enumerate(q_masked) if v == best_q]
    action_idx   = tied_actions[mem["rng"].randrange(len(tied_actions))]
    return action_idx


def _init_memory():
    """
    Initialise the persistent state dictionary stored as controller._mem.

    Naming convention for all keys:
        wp_idx       – waypoint index into _CSV_TARGETS (free-fly mode only).
        q_table      – dict mapping state_key → list of 16 Q-values (action_idx → float).
        alpha        – TD learning rate (float).
        gamma        – discount factor (float).
        epsilon      – current exploration probability (float, decays over time).
        eps_decay    – per-step multiplicative decay applied to epsilon.
        eps_floor    – minimum value epsilon can decay to.
        replay       – list of (state_key, action_idx, reward, next_state_key) tuples.
        replay_cap   – maximum replay buffer length (FIFO eviction when exceeded).
        replay_batch – number of transitions sampled per mini-batch update.
        prev_key     – state_key from the previous controller() call.
        prev_action  – action_idx from the previous controller() call.
        prev_cost    – RL cost from the previous controller() call.
        prev_effort  – total velocity magnitude from the previous controller() call.
        integral     – [ix, iy, iz]: world-frame PID integral accumulators (m·s).
        integral_yaw – yaw PID integral accumulator (rad·s).
        prev_err     – [ex_w, ey_w, ez]: world-frame errors from previous call (for D-term).
        dwell_steps  – consecutive in-tolerance steps counter.
        step_count   – total controller() calls (used for TD update throttling).
        rng          – seeded random.Random instance for reproducibility.
    """
    return {
        "wp_idx":       0,
        "q_table":      {},
        "alpha":        0.20,
        "gamma":        0.94,
        "epsilon":      0.25,
        "eps_decay":    0.9985,
        "eps_floor":    0.04,
        "replay":       [],
        "replay_cap":   400,
        "replay_batch": 16,
        "prev_key":     None,
        "prev_action":  None,
        "prev_cost":    None,
        "prev_effort":  0.0,
        "integral":     [0.0, 0.0, 0.0],
        "integral_yaw": 0.0,
        "prev_err":     [0.0, 0.0, 0.0],
        "dwell_steps":  0,
        "step_count":   0,
        "rng":          random.Random(42),
    }


# =============================================================================
# MAIN CONTROLLER – DO NOT modify inputs, outputs, or function name.
# Ref: Coursework spec p.9 – "DO NOT MODIFY INPUT/OUTPUT variable names."
# =============================================================================

def controller(state, target_pos, dt, wind_enabled=False):
    """
    [ADVANCED METHOD – Cascade RL+PID Controller]

    NAMING CONVENTION used throughout this function:
        target_pos    – raw argument from run.py (may be None in free-fly mode).
        active_target – resolved 4-tuple (tx, ty, tz, tyaw) used for all computation.
        errs          – dict from _get_errors with keys: dist, cost, eyaw, ez,
                        ex_w, ey_w, ex_b, ey_b.
        state_key     – discretised RL state (dist_band, yaw_band, wind_int).
        dwell_bonus   – float reward from _handle_dwell; forwarded to _update_learning_agent.
        action_idx    – integer 0-15 index of the selected gain profile.
        kp_xy … yaw_rate_max – gains and caps unpacked from _ACTION_PROFILES[action_idx].
        integ_limit   – anti-windup clamp value for integral accumulators.
        d_ex_w, d_ey_w, d_ez – finite-difference derivatives of world-frame errors.
        vx_w_pid … vz_w_pid   – raw PID outputs in world frame (before speed scaling).
        v_mag_pid     – 3-D magnitude of the raw PID world-frame velocity vector.
        v_cruise_max  – safe maximum cruise speed = min(v_xy_max, v_z_max).
        v_scale       – scalar applied to world-frame vector to enforce speed limit.
        vx_w … vz_w   – speed-scaled world-frame velocity commands.
        cos_yaw, sin_yaw – precomputed trig for the body-frame rotation.
        vx, vy, vz    – final body-frame velocity commands returned to run.py.
        yr            – yaw rate command returned to run.py.

    CASCADE ARCHITECTURE:
        OUTER LOOP: Q-learning selects action_idx (gain profile) based on state_key.
        INNER LOOP: PID uses selected gains to compute vx, vy, vz, yr.
        Ref: Lecture 10, p.42-43 – cascade control.
    """
    if not hasattr(controller, "_mem"):
        controller._mem = _init_memory()
    mem = controller._mem
    mem["step_count"] += 1

    # =========================================================================
    # STEP 1 – Resolve active_target from target_pos; compute errs and state_key
    # =========================================================================
    # target_pos: raw from run.py (marker auto-tester always provides this).
    # active_target: resolved 4-tuple used for all error computation below.
    active_target = (
        tuple(float(v) for v in target_pos)
        if target_pos is not None
        else _CSV_TARGETS[mem["wp_idx"]]
    )

    errs      = _get_errors(state, active_target)
    state_key = _state_key(errs["dist"], abs(errs["eyaw"]), wind_enabled)

    # =========================================================================
    # STEP 2 – Dwell-hold and Q-learning update
    # dwell_bonus forwarded from _handle_dwell → _update_learning_agent.
    # target_pos passed to _handle_dwell (not active_target) so it can detect
    # free-fly mode (target_pos is None) vs simulator mode (target_pos provided).
    # =========================================================================
    dwell_bonus = _handle_dwell(mem, errs, target_pos)
    _update_learning_agent(mem, errs, dwell_bonus, state_key)

    # =========================================================================
    # STEP 3 – Select action_idx via ε-greedy policy with safety masking
    # =========================================================================
    action_idx = _choose_action(
        mem, state_key,
        force_exploit=(errs["dist"] < 0.5),   # Pure greedy when close.
        mask_coarse=(errs["dist"] < 0.1),     # Block profiles 0-3 near target.
        mask_fine=(errs["dist"] > 1.5),       # Block profiles 8-15 far from target.
    )

    (kp_xy, kp_z, kp_yaw,
     ki_xy, ki_z, ki_yaw,
     kd_xy, kd_z,
     v_xy_max, v_z_max, yaw_rate_max) = _ACTION_PROFILES[action_idx]

    # =========================================================================
    # STEP 4 – PID integral accumulation (world frame, anti-windup)
    # Uses ex_w, ey_w (world frame) not ex_b, ey_b (body frame).
    # World-frame integration avoids false windup when the drone yaws.
    # Ref: Lecture 10, p.10-11 – I-term and anti-windup clamping.
    # =========================================================================
    if dt > 0.0:
        integ_limit = (
            0.40 if errs["dist"] < _POS_TOL   # Tight near target.
            else 1.00 if wind_enabled          # Wide under wind.
            else 0.60                          # Standard approach.
        )
        mem["integral"][0]  = _clip(mem["integral"][0]  + errs["ex_w"] * dt, -integ_limit, integ_limit)
        mem["integral"][1]  = _clip(mem["integral"][1]  + errs["ey_w"] * dt, -integ_limit, integ_limit)
        mem["integral"][2]  = _clip(mem["integral"][2]  + errs["ez"]   * dt, -integ_limit, integ_limit)
        mem["integral_yaw"] = _clip(mem["integral_yaw"] + errs["eyaw"] * dt, -0.30, 0.30)

    # =========================================================================
    # STEP 5 – PID derivative (world frame, spike-clamped)
    # d_ex_w, d_ey_w, d_ez: finite differences of world-frame errors.
    # prev_err stores [ex_w, ey_w, ez] from the previous call (set in Step 9).
    # Clamped to ±3.0 to suppress sensor noise spikes.
    # Ref: Lecture 10, p.12 – D-term finite difference.
    # =========================================================================
    d_ex_w = _clip((errs["ex_w"] - mem["prev_err"][0]) / dt, -3.0, 3.0) if dt > 0 else 0.0
    d_ey_w = _clip((errs["ey_w"] - mem["prev_err"][1]) / dt, -3.0, 3.0) if dt > 0 else 0.0
    d_ez   = _clip((errs["ez"]   - mem["prev_err"][2]) / dt, -3.0, 3.0) if dt > 0 else 0.0

    # =========================================================================
    # STEP 6 – Raw PID outputs in world frame (vx_w_pid, vy_w_pid, vz_w_pid)
    # _pid suffix marks these as the direct PID outputs before speed scaling.
    # v = Kp·e + Ki·∫e·dt + Kd·(de/dt). Ref: Lecture 10, p.8-13.
    # =========================================================================
    vx_w_pid = kp_xy * errs["ex_w"] + ki_xy * mem["integral"][0] + kd_xy * d_ex_w
    vy_w_pid = kp_xy * errs["ey_w"] + ki_xy * mem["integral"][1] + kd_xy * d_ey_w
    vz_w_pid = kp_z  * errs["ez"]   + ki_z  * mem["integral"][2] + kd_z  * d_ez
    yr       = _clip(kp_yaw * errs["eyaw"] + ki_yaw * mem["integral_yaw"],
                     -yaw_rate_max, yaw_rate_max)

<<<<<<< HEAD
    # =========================================================================
    # STEP 7 – 3-D vector speed scaling (straight-line constant-speed flight)
    # v_mag_pid: magnitude of the raw PID world-frame velocity vector.
    # v_cruise_max: safe maximum speed = min(v_xy_max, v_z_max).
    # v_scale: scalar preserving direction while enforcing the speed limit.
    # vx_w, vy_w, vz_w: scaled world-frame velocity commands.
    #
    # WHY SCALE INSTEAD OF CLIP?
    #   Independent axis clipping changes the vector direction when any axis
    #   saturates, causing the drone to follow a curved arc. Uniform scaling
    #   preserves direction and produces a straight-line approach.
    # =========================================================================
    v_mag_pid    = math.sqrt(vx_w_pid**2 + vy_w_pid**2 + vz_w_pid**2)
    v_cruise_max = min(v_xy_max, v_z_max)

    if v_mag_pid < 0.001:
        v_scale = 0.0                                      # Essentially stationary.
    elif errs["dist"] > 0.25:
        v_scale = v_cruise_max / v_mag_pid                 # Cruising: enforce constant speed.
    else:
        v_scale = min(1.0, v_cruise_max / v_mag_pid)       # Precision: cap but allow slowdown.
=======
    # Step 5b: Apply 3D Vector Scaling for Constant Speed & Straight Path
    v_mag_raw = math.sqrt(vx_w_raw**2 + vy_w_raw**2 + vz_w_raw**2)
    v_target_speed = min(v_xy_max, v_z_max) # The safe maximum speed limit
    
    if v_mag_raw < 0.001:
        scale = 0.0
    elif errs["dist"] > 0.25:
        # Cruising Phase: Force the magnitude to EXACTLY v_target_speed.
        # This guarantees an "even speed throughout the flight time" while maintaining the 
        # exact straight-line 3D direction prescribed by the World Frame PID.
        scale = v_target_speed / v_mag_raw
    else:
        # Precision Phase (< 0.25m): Allow the drone to slow down naturally to hold hover.
        # We simply cap the maximum vector magnitude without forcing a constant speed.
        scale = min(1.0, v_target_speed / v_mag_raw)
>>>>>>> 07d710f5bd881e1f2b5093dca36bf5f9b74103d5

    vx_w = vx_w_pid * v_scale
    vy_w = vy_w_pid * v_scale
    vz_w = vz_w_pid * v_scale

    # =========================================================================
    # STEP 8 – Rotate world-frame commands into body frame (vx, vy, vz)
    # cos_yaw, sin_yaw: trig values for the R_z(yaw) rotation matrix.
    # Ref: Lecture 6, p.14 – R_z(ψ); Coursework spec p.10 – body-frame commands.
    # =========================================================================
    cos_yaw, sin_yaw = math.cos(float(state[5])), math.sin(float(state[5]))
    vx =  cos_yaw * vx_w + sin_yaw * vy_w
    vy = -sin_yaw * vx_w + cos_yaw * vy_w
    vz =  vz_w

    # =========================================================================
    # STEP 9 – Bookkeeping: store this step's values for next call's TD update
    # prev_key, prev_action, prev_cost, prev_effort, prev_err all follow the
    # prev_* naming convention consistently throughout _init_memory and callers.
    # =========================================================================
    mem["prev_key"]    = state_key
    mem["prev_action"] = action_idx
    mem["prev_cost"]   = errs["cost"]
    mem["prev_effort"] = abs(vx) + abs(vy) + abs(vz) + abs(yr)
    mem["prev_err"]    = [errs["ex_w"], errs["ey_w"], errs["ez"]]

    # =========================================================================
    # STEP 10 – Telemetry logging (zero control-loop impact)
    # =========================================================================
    _logger.log(
        dt=dt, pos=state[0:3], euler=state[3:6],
        vel_cmd=(vx, vy, vz), target=active_target,
        wind=wind_enabled, profile_idx=action_idx,
    )

    return (vx, vy, vz, yr)


# =============================================================================
# CHANGELOG
# =============================================================================
# 1. matplotlib crash: switched TkAgg → Agg (file-only, thread-safe).
# 2. Straight-line flight: PID uses world-frame errors; vector scaling replaces
#    axis clipping to preserve velocity direction.
# 3. Constant speed: cruising phase normalises vector to exactly v_cruise_max.
# 4. Altitude droop: ki_z raised 0.02 → 0.05 in hover profiles (12-15).
# 5. Early braking: kd_xy lowered 0.50 → 0.10 in coarse profiles (0-3).
# 6. 100% naming consistency: state_key, action_idx, dwell_bonus, active_target,
#    target_pos, vx_w_pid/_pid suffix, v_scale, v_cruise_max, v_mag_pid,
#    cos_yaw/sin_yaw, prev_* pattern, dist_band/yaw_band/wind_int in _state_key,
#    vx_cmd/vy_cmd/vz_cmd and *_log suffix in _FlightLogger.
# 7. Removed dead code: unused threading import, unused _wrap function.
# =============================================================================