def scan(seq):
    return (seq
        .pose(upper_arm_yaw=180, duration=3)
        .pose(upper_arm_yaw=-180, duration=6)
        )