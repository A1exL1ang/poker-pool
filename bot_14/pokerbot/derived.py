"""Cheap relational features that help small offline trees generalize."""


def augment(f, q, m, opponents):
    board = f[11:11 + int(f[0])]
    made = f[1:7]
    board_made = f[33:39]
    pairs = [r for r in set(board) if board.count(r) >= 2]
    return tuple(f) + (q, m, opponents,
        f[2] - f[11], sum(r > f[2] for r in board),
        f[1] - f[33], int(made == board_made),
        int(f[7] in (f[2], f[3])) + int(f[8] in (f[2], f[3])),
        max(pairs, default=0), min(pairs, default=0),
        f[7] - f[11], f[8] - f[11],
        f[2] - f[34], f[3] - f[35])
