# Sample Mujoco Script, everything has already been set up so this is all you have to write for each function

def confused_scan(seq):
    return (seq
        .pose(main_swivel=20, duration = 0.2)
        .pose(eye=1, duration = 0.2)
        .pose(main_swivel=-20, duration = 0.2)
        .pose(eye=-1, duration = 0.2)

        .pose(main_swivel=20, duration = 0.2)
        .pose(eye=1, duration = 0.2)
        .pose(main_swivel=-20, duration = 0.2)
        .pose(eye=-1, duration = 0.2)
        )
