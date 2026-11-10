def look_away(seq):
    return (seq
        .pose(main_swivel=-90, duration=0.3, additive=True)
        .pose(eye=-2, duration=1.0, lerp=False, additive=True)
        )
