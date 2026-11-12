def ending_demo(seq):
    return (seq
        .pose(nod=30, lower_arm=80, duration=1)
        .pose(main_swivel=-30, tilt=-30, duration=1)
        .pose(nod=15.47, lower_arm=83.03, tilt=-15, duration=0.5)
        .pose(nod=-10, lower_arm=5, duration=1, additive=True)
        .pose(nod=20, lower_arm=-10, duration=1, additive=True)
        )