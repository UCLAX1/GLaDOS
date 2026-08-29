# Sample Mujoco Script, everything has already been set up so this is all you have to write for each function

def nod(seq):
    return (seq
        .pose(nod=20,  duration=0.3, lerp=True)
        .pose(nod=-30, duration=0.6, lerp=True)
        )
