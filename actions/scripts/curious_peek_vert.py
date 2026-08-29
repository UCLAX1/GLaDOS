def curious_peek_vert(seq):
    return (seq
        .pose(lower_arm=30, eye=2, duration=1) # Depends where target is, 30 deg is arbitrary for now
        )