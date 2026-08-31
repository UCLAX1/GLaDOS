def eye_extend(seq):
    return (seq
        .pose(eye=2, duration=0.2, additive=True)
        )