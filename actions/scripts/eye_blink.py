def eye_blink(seq):
    return (seq
        .pose(eye=2, duration=0.2, additive=True)
        .pose(eye=-2, duration=0.2, additive=True)
        )