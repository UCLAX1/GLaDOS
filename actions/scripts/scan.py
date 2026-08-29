def scan(seq):
    return (seq
        .pose(main_swivel=180, duration=3)
        .pose(main_swivel=-180, duration=6)
        )