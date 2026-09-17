# Sample Mujoco Script, everything has already been set up so this is all you have to write for each function

def look_away(seq):
    return (seq
        .pose(main_swivel=-90, duration=0.3, additive=True)
        .pose(eye=-2, duration=1 lerp=False)
        )