def idle(seq):
    return (seq
        .pose(nod=2,  main_swivel=4,  duration=3.0, lerp=True, additive=True)
        .pose(nod=-2, main_swivel=-8, duration=6.0, lerp=True, additive=True)
        .pose(nod=1,  main_swivel=3,  duration=3.0, lerp=True, additive=True)
        .pose(nod=-1, main_swivel=1,  duration=4.0, lerp=True, additive=True)
    )
