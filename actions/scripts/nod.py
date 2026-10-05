def nod(seq):
    return (seq
        .pose(nod=20,  duration=0.3, lerp=True, additive=True)
        .pose(nod=-20, duration=0.5, lerp=True, additive=True)
        .pose(nod=10,  duration=0.3, lerp=True, additive=True)
        )
