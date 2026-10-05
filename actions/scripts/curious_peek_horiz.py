def curious_peek_horiz(seq):
    return (seq
        .pose(main_swivel=30, eye=1, duration=1.0, additive=True)
        .pose(main_swivel=-30, eye=-1, duration=1.0, additive=True)
        )
