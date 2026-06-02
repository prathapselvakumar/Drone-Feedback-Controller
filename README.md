# Drone Feedback Controller: Q-Learning Gain Scheduling for PID Position Control

This repository contains my submission for Coursework 3 (Feedback Control) of AERO60492 Autonomous Mobile Robots. It is a position controller for a simulated Tello quadcopter that flies to a sequence of 3-D waypoints, each with a target heading, and holds position there, with or without simulated wind. The controller combines a PID law with a tabular Q-learning agent that chooses, at every control step, which of 16 pre-tuned sets of PID gains and speed limits to use.

## Problem

The simulator, supplied with the coursework, models the Tello in PyBullet at 1 kHz. Every 20 physics steps (50 Hz, a 0.02 s control period) it calls one function:

```python
controller(state, target_pos, dt, wind_enabled) -> (vx, vy, vz, yaw_rate)
```

| Input | Contents |
| :--- | :--- |
| `state` | Position (x, y, z) in metres and orientation (roll, pitch, yaw) in radians, world frame |
| `target_pos` | Target (x, y, z, yaw) |
| `dt` | Control period, 0.02 s |
| `wind_enabled` | Whether the wind disturbance is switched on |

The function returns body-frame velocity commands in m/s and a yaw-rate command in rad/s. The simulator clips each velocity to ±1 m/s and the yaw rate to ±1.745 rad/s (±100°/s), then passes them to the Tello's own cascaded velocity, attitude, and rate PID loops. Only `controller.py` is my work. The simulator (`run.py`), the inner loops (`src/`), and the drone model (`resources/`) come from the coursework.

A single fixed set of PID gains has to compromise between two regimes. Gains that cross several metres quickly tend to overshoot at the target, and gains that hold a precise hover are too slow over long distances. This controller addresses the conflict by letting a learning agent schedule the gains according to how far the drone is from the target.

## Method

The controller is a two-level cascade. The outer level is the Q-learning agent, which picks a gain profile. The inner level is a PID law that uses that profile to compute the velocity command.

### 1. Tracking errors

Position errors are computed in the world frame:

```
ex_w = tx − x,   ey_w = ty − y,   ez = tz − z
```

The yaw error is wrapped to (−π, π] with `atan2(sin(tyaw − yaw), cos(tyaw − yaw))`. The PID integral and derivative terms use the world-frame errors, so the drone turning on the spot does not create false changes in error. Commands are rotated into the body frame only at the last step (Section 5).

### 2. Q-learning gain scheduling

The agent observes a discrete state of 96 cells, formed from three variables:

| State variable | Bands |
| :--- | :--- |
| Distance to target (m) | < 0.08, < 0.20, < 0.40, < 0.70, < 1.10, < 1.80, < 2.80, ≥ 2.80 |
| Absolute yaw error (rad) | < 0.04, < 0.10, < 0.25, < 0.50, < 1.00, ≥ 1.00 |
| Wind | off, on |

The bands are narrower near the target, where precision matters most. Each of the 16 actions is a profile of 11 values: proportional, integral, and derivative gains for the horizontal and vertical axes, proportional and integral gains for yaw, and caps on horizontal speed, vertical speed, and yaw rate. The profiles fall into four groups of four:

| Profiles | Intended range | Character |
| :---: | :--- | :--- |
| 0 to 3 | Coarse, more than 1.5 m | High proportional gain, low damping, no XY integral, speed caps 0.8 to 1.5 m/s |
| 4 to 7 | Mid-range, 0.5 to 1.5 m | Moderate gains, stronger yaw correction |
| 8 to 11 | Fine, under 0.5 m | High damping for braking, small integral, caps 0.12 to 0.25 m/s |
| 12 to 15 | Hover hold, under 0.08 m | Low proportional gain, high damping, caps 0.04 to 0.10 m/s |

The agent learns online with one-step temporal-difference (TD) updates:

```
Q(s, a) ← Q(s, a) + α [ r + γ · max Q(s′, ·) − Q(s, a) ]
```

| Learning parameter | Value |
| :--- | :--- |
| Learning rate α | 0.20 |
| Discount factor γ | 0.94 |
| Exploration ε | starts at 0.25, multiplied by 0.9985 each step, floor 0.04 |
| Experience replay | buffer of 400 transitions, mini-batch of 16 |
| Update frequency | every fifth control step |
| Random seed | 42 |

The reward for each step is built from a cost, `cost = dist² + 0.6·eyaw² + 5·max(0, 0.10 − dist)²`:

```
r = (previous cost − current cost) + dwell bonus
    − 0.015 · previous command effort − 2.0 · dist²
    − overshoot penalty
```

The overshoot penalty is 1,500 times the rise in cost, applied only when the drone is within 0.6 m and the cost has increased, meaning it is moving away from a target it had nearly reached. The dwell bonus is 20 on entering the tolerance zone and 30 after holding it (Section 6).

Two safety rules constrain the agent. Within 0.5 m of the target it always takes the best-known action instead of exploring. Profiles 0 to 3 are blocked within 0.1 m, and profiles 8 to 15 are blocked beyond 1.5 m, so the agent cannot pick a high-speed profile next to the target or a hover profile far from it.

### 3. PID law

With the selected profile's gains, each axis gets a PID velocity command:

```
v = kp·e + ki·∫e dt + kd·de/dt
```

The integrals are clamped to ±0.40 inside the position tolerance, ±1.00 with wind on, and ±0.60 otherwise; the yaw integral is clamped to ±0.30. Derivatives are finite differences clamped to ±3.0 to suppress spikes. The yaw-rate command is proportional plus integral, capped at the profile's yaw-rate limit.

### 4. Speed scaling

The three world-frame velocity components are scaled together by one factor, so the direction of travel does not change when the speed limit is reached. Clipping each axis separately would bend the path whenever one axis saturated. The limit is the smaller of the profile's horizontal and vertical caps.

| Phase | Condition | Scaling |
| :--- | :--- | :--- |
| Cruise | distance > 0.25 m | Magnitude set exactly to the speed limit, giving a constant speed |
| Precision | distance ≤ 0.25 m | Magnitude capped at the speed limit, so the drone can slow down |

### 5. Body-frame output

The scaled world-frame velocity is rotated by the current yaw into the body frame:

```
vx =  cos(yaw)·vx_w + sin(yaw)·vy_w
vy = −sin(yaw)·vx_w + cos(yaw)·vy_w
vz =  vz_w
```

### 6. Dwell detection

The drone is in tolerance when it is within 0.08 m of the target and within 0.06 rad of the target yaw. After 20 consecutive in-tolerance steps (0.4 s) the controller awards the completion bonus and resets the integrators. If `target_pos` is `None`, it also advances to the next waypoint in `targets.csv`. The supplied simulator always passes a target, so in normal use waypoints are changed with the arrow keys.

### 7. Flight logging

Each call to `controller()` appends one row to an in-memory log. On exit, the log is written to `logs/flight_<timestamp>.csv` and a 3-D plot of the path to `logs/flight_<timestamp>_path.png`. Matplotlib uses the file-only `Agg` backend, because an earlier version crashed when it opened a plot window from a background thread on Windows. The CSV has these columns:

```
time_s, x, y, z, roll_rad, pitch_rad, yaw_rad, vx_cmd, vy_cmd, vz_cmd,
target_x, target_y, target_z, target_yaw, dist_m, yaw_err_rad, speed_ms,
wind_enabled, rl_profile
```

## Waypoints

`targets.csv` defines a 4 m square at 2 m altitude, turning a quarter turn at each corner:

| x (m) | y (m) | z (m) | yaw (rad) |
| ---: | ---: | ---: | ---: |
| 2 | 2 | 2 | 0 |
| −2 | 2 | 2 | 1.57 |
| −2 | −2 | 2 | 3.14 |
| 2 | −2 | 2 | 4.71 |

## Running the simulation

The simulator needs Python 3 with NumPy, PyBullet, and Matplotlib.

```bash
pip install -r requirements.txt
python run.py
```

| Key | Action |
| :---: | :--- |
| Right / left arrow | Next / previous waypoint |
| K | Toggle wind |
| R | Reset the drone, reload `controller.py` and `targets.csv` |
| Q | Quit and write the flight log |

## Repository structure

| Path | Contents |
| :--- | :--- |
| `controller.py` | The submitted controller |
| `run.py` | PyBullet simulator (supplied) |
| `src/` | Tello inner-loop controller, PID class, and wind model (supplied) |
| `resources/` | Tello URDF and meshes (supplied) |
| `targets.csv` | Waypoints |
| `Controller/` | Earlier versions of the controller, kept for reference |
| `logs/` | Flight logs and path plots from development runs |

## Limitations

- The Q-table starts empty on every run and is never saved, so the agent relearns from scratch each flight. Most of the behaviour at the start of a run comes from the safety masks and the hand-tuned profiles, not from learned values.
- The cruise speed limits of profiles 0 and 1 (1.5 and 1.2 m/s) exceed the simulator's ±1 m/s clip on each axis. When one component of the command is above 1 m/s and another is not, the clip shortens only that component, which changes the direction that the vector scaling was meant to keep.
- The `WIND` flag at the top of `controller.py` is not read by the control logic. The behaviour under wind depends only on the `wind_enabled` argument that the simulator passes in.
- Comments in `controller.py` describe the control rate as 20 Hz, but the simulator calls the controller every 20 ms, which is 50 Hz. The timing values in the code, such as the 0.4 s dwell, are correct for 50 Hz.
- The flight logs in `logs/` come from development runs in April and May 2026, before the final changes to `controller.py` on 2 June 2026. This repository does not include a quantitative evaluation of the final version.
- The simulator runs in real time, so results vary slightly between runs even though the agent's random number generator is seeded.
