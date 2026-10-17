def confused_scan(seq):
    return (seq
        .pose(upper_arm_yaw=20, eye=1,  duration=0.2, additive=True)
        .pose(upper_arm_yaw=-20, eye=-1, duration=0.2, additive=True)
        .pose(upper_arm_yaw=20, eye=1,  duration=0.2, additive=True)
        .pose(upper_arm_yaw=-20, eye=-1, duration=0.2, additive=True)
        )
