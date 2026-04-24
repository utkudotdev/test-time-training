"""Low-level PD attitude controller for Skydio X2 drone.

Converts high-level commands [thrust, desired_roll, desired_pitch, desired_yaw_rate]
into individual motor commands. Allows the RL policy to focus on navigation
instead of motor-level stabilization.

Rotor layout (from x2.xml):
  motor1 (back-left,   CCW): pos=(-.14, -.18, .05), yaw gear = -0.0201
  motor2 (back-right,  CW ): pos=(-.14, +.18, .05), yaw gear = +0.0201
  motor3 (front-right, CCW): pos=(+.14, +.18, .08), yaw gear = -0.0201
  motor4 (front-left,  CW ): pos=(+.14, -.18, .08), yaw gear = +0.0201
"""

import numpy as np


# --- Action scaling (what policy outputs to physical quantities) ---
MAX_TILT = 0.4          # rad (~23 deg), max commanded roll/pitch angle
MAX_YAW = 1.0           # rad (~57 deg), max commanded yaw angle (absolute, world frame)
THRUST_DELTA = 2.0      # per-motor thrust range around hover (policy ±1 → ±THRUST_DELTA)

# --- PD gains ---
KP_ATT = 8.0     # roll/pitch angle error → torque
KD_RATE = 1.0    # angular rate damping (roll/pitch)
KP_YAW = 4.0     # yaw angle error → torque
KD_YAW = 0.8     # yaw rate damping


def quat_to_euler(quat):
    """MuJoCo quat (w, x, y, z) → (roll, pitch, yaw) in radians, ZYX order."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)

    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)

    return roll, pitch, yaw


def cascaded_control(quat, gyro, action, hover_thrust):
    """Convert high-level action → 4 motor commands via PD attitude control.

    Args:
      quat: MuJoCo (w, x, y, z) orientation of drone.
      gyro: body-frame angular velocity (rad/s), length-3 from body_gyro sensor.
      action: 4-D array in [-1, 1]:
        action[0] = thrust delta (−1 → hover−THRUST_DELTA, +1 → hover+THRUST_DELTA)
        action[1] = desired roll  (scaled by MAX_TILT)
        action[2] = desired pitch (scaled by MAX_TILT)
        action[3] = desired yaw angle, world frame (scaled by MAX_YAW)
                    Zero action → drone actively holds yaw = 0.
      hover_thrust: per-motor thrust that balances gravity.

    Returns:
      4 motor commands, each in [0, 13].
    """
    thrust_cmd = hover_thrust + action[0] * THRUST_DELTA
    roll_des = action[1] * MAX_TILT
    pitch_des = action[2] * MAX_TILT
    yaw_des = action[3] * MAX_YAW

    roll, pitch, yaw = quat_to_euler(quat)

    # Wrap yaw error to [-pi, pi] so shortest path is taken
    yaw_err = np.arctan2(np.sin(yaw_des - yaw), np.cos(yaw_des - yaw))

    # PD: angle error → torque, damped by current rate
    roll_torque = KP_ATT * (roll_des - roll) - KD_RATE * gyro[0]
    pitch_torque = KP_ATT * (pitch_des - pitch) - KD_RATE * gyro[1]
    yaw_torque = KP_YAW * yaw_err - KD_YAW * gyro[2]

    # Motor mixing. Signs derived from τ = r × F with each motor's (x, y) offset
    # and yaw gear term (±0.0201). Matches ZYX Euler convention in quat_to_euler.
    #   +roll  (+x torque): motors at +y (m2, m3) push up → m1-, m2+, m3+, m4-
    #   +pitch (+y torque): motors at -x (m1, m2) push up → m1+, m2+, m3-, m4-
    #   +yaw   (+z torque): CW motors (m2, m4) push up    → m1-, m2+, m3-, m4+
    m1 = thrust_cmd - roll_torque + pitch_torque - yaw_torque
    m2 = thrust_cmd + roll_torque + pitch_torque + yaw_torque
    m3 = thrust_cmd + roll_torque - pitch_torque - yaw_torque
    m4 = thrust_cmd - roll_torque - pitch_torque + yaw_torque

    return np.clip([m1, m2, m3, m4], 0.0, 13.0)
