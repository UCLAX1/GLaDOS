def curious_peek_horiz(seq):
    return (seq
        .pose(upper_arm_yaw=30, eye=1, duration=1.0, additive=True)
        .pose(upper_arm_yaw=-30, eye=-1, duration=1.0, additive=True)
        )
