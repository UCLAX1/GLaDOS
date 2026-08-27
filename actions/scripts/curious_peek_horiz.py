def curous_peek_horiz(seq):
    return (seq
        .pose(main_swivel=30, eye=2, duration=1) # Depends where target is, 30 deg is arbitrary for now
        )