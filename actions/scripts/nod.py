def nod(seq):
    return (seq
        .pose(nod=10,  duration=0.3, lerp=True, additive=True)
        .pose(nod=-10, duration=0.5, lerp=True, additive=True)
        .pose(nod=5,   duration=0.3, lerp=True, additive=True)
        )
