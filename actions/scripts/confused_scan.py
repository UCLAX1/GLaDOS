def confused_scan(seq):
    return (seq
        .pose(main_swivel=20, eye=1,  duration=0.2, additive=True)
        .pose(main_swivel=-20, eye=-1, duration=0.2, additive=True)
        .pose(main_swivel=20, eye=1,  duration=0.2, additive=True)
        .pose(main_swivel=-20, eye=-1, duration=0.2, additive=True)
        )
