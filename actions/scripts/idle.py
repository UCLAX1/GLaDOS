def idle(seq):
    return (seq
        .pose(nod=2,  upper_arm_yaw=4,  duration=3.0, lerp=True, additive=True)
        .pose(nod=-2, upper_arm_yaw=-8, duration=6.0, lerp=True, additive=True)
        .pose(nod=1,  upper_arm_yaw=3,  duration=3.0, lerp=True, additive=True)
        .pose(nod=-1, upper_arm_yaw=1,  duration=4.0, lerp=True, additive=True)
    )
