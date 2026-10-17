def look_away(seq):
    return (seq
        .pose(upper_arm_yaw=-90, duration=0.3, additive=True)
        .pose(eye=-2, duration=1.0, lerp=False, additive=True)
        )
