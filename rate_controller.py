"""Low-level rate controller for Skydio X2 drone.

Converts high-level commands [thrust, desired_roll_rate, desired_pitch_rate, desired_yaw_rate]
into individual motor commands. Allows the RL policy to focus on navigation instead of motor-level stabilization.
"""

import numpy as np


# --- Action scaling ---
MAX_RATE = 3.0          
MAX_YAW_RATE = 1.5     
THRUST_DELTA = 2.0

# --- only P gain ---
KP_RATE = 1.5           # rate error → torque


def cascaded_control(quat, gyro, action, hover_thrust):
    """
    action[0] = thrust delta
    action[1] = desired roll rate   (body frame)
    action[2] = desired pitch rate  (body frame)
    action[3] = desired yaw rate    (body frame)
    """
    thrust_cmd = hover_thrust + action[0] * THRUST_DELTA
    roll_rate_des  = action[1] * MAX_RATE
    pitch_rate_des = action[2] * MAX_RATE
    yaw_rate_des   = action[3] * MAX_YAW_RATE

    # single P: rate error → torque
    roll_torque  = KP_RATE * (roll_rate_des  - gyro[0])
    pitch_torque = KP_RATE * (pitch_rate_des - gyro[1])
    yaw_torque   = KP_RATE * (yaw_rate_des   - gyro[2])

    # single P: rate error → torque
    m1 = thrust_cmd - roll_torque + pitch_torque - yaw_torque
    m2 = thrust_cmd + roll_torque + pitch_torque + yaw_torque
    m3 = thrust_cmd + roll_torque - pitch_torque - yaw_torque
    m4 = thrust_cmd - roll_torque - pitch_torque + yaw_torque

    return np.clip([m1, m2, m3, m4], 0.0, 13.0)