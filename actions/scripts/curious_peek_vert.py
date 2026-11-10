def curious_peek_vert(seq):
    return (seq
        .pose(lower_arm=15, eye=1, duration=1.0, additive=True)
        .pose(lower_arm=-15, eye=-1, duration=1.0, additive=True)
        )
