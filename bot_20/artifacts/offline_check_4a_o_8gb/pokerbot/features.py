"""Public-card feature extraction for the offline equity approximator.

No opponent private cards, simulation, network access, or randomness is used.
The feature schema is versioned because the compact model indexes it directly.
"""

from __future__ import annotations

from .evaluator import evaluate, STRAIGHT_HIGH

FEATURE_VERSION = 2
FEATURE_NAMES = (
    "board_count", "hole_high", "hole_low", "pocket_pair", "suited", "hole_gap",
    "category", "kicker_1", "kicker_2", "kicker_3", "kicker_4", "kicker_5",
    "board_category", "board_kicker_1", "board_kicker_2", "board_kicker_3", "board_kicker_4", "board_kicker_5",
    "board_rank_1", "board_rank_2", "board_rank_3", "board_rank_4", "board_rank_5",
    "board_distinct", "board_pairs", "board_trips", "board_quads",
    "board_suit_1", "board_suit_2", "board_suit_3", "board_suit_4",
    "hole_high_matches", "hole_low_matches", "hole_overcards", "ranks_above_pair",
    "max_flush_count", "flush_hole_count", "flush_hole_high", "flush_unseen_above",
    "board_straight_window", "combined_straight_window", "straight_out_ranks", "straight_hole_help",
    "board_straight_high", "combined_straight_high", "hand_plays_board",
    "hole_high_board_suit_count", "hole_low_board_suit_count", "nut_flush_blocker",
    "board_rank_mask", "combined_rank_mask",
    "board_count_2", "board_count_3", "board_count_4", "board_count_5", "board_count_6", "board_count_7",
    "board_count_8", "board_count_9", "board_count_T", "board_count_J", "board_count_Q", "board_count_K", "board_count_A",
    "same_category_as_board", "first_kicker_difference", "first_private_kicker", "available_cards_above_private_kicker",
    "board_pair_high", "board_pair_low", "board_trip_rank", "hole_matches_primary_rank", "hole_matches_secondary_rank",
    "board_singletons_above_primary", "board_pairs_above_primary", "board_pairs_above_secondary",
    "possible_straight_high", "single_card_straight_high", "straight_gap_to_nuts", "straight_gap_to_single_card_nuts",
    "single_card_higher_straight_outs", "two_card_higher_straight_combos", "single_card_straight_outs",
    "possible_flush_combos", "higher_full_house_pocket_pair_combos", "higher_full_house_single_card_outs",
    "primary_rank_minus_board_high", "secondary_rank_minus_board_high", "private_pair_above_board_pairs", "high_hole_minus_board_high",
)

STRAIGHT_MASKS = tuple(31 << offset for offset in range(9)) + (0x100F,)


def extract_features(hole, board) -> tuple:
    """Return reusable numeric features for a two-card hand and 3..5-card board."""
    a, b = hole
    ar, br = (a >> 2) + 2, (b >> 2) + 2
    if ar < br:
        a, b = b, a
        ar, br = br, ar
    board_counts = [0] * 13
    suit_counts = [0] * 4
    suit_masks = [0] * 4
    board_mask = 0
    for card in board:
        rank = card >> 2
        bit = 1 << rank
        board_counts[rank] += 1
        suit_counts[card & 3] += 1
        suit_masks[card & 3] |= bit
        board_mask |= bit
    if ar == br and (suit_counts[a & 3], suit_masks[a & 3]) < (suit_counts[b & 3], suit_masks[b & 3]):
        a, b = b, a
    ranks = sorted(((card >> 2) + 2 for card in board), reverse=True)
    ranks += [0] * (5 - len(ranks))
    combined_mask = board_mask | (1 << (ar - 2)) | (1 << (br - 2))
    total_suits = suit_counts[:]
    total_suits[a & 3] += 1
    total_suits[b & 3] += 1
    flush_suit = max(range(4), key=lambda s: (total_suits[s], suit_counts[s], (ar if a & 3 == s else 0), (br if b & 3 == s else 0), suit_masks[s]))
    flush_holes = int(a & 3 == flush_suit) + int(b & 3 == flush_suit)
    flush_high = ar if a & 3 == flush_suit else (br if b & 3 == flush_suit else 0)
    unseen_above = sum(1 for r in range(max(0, flush_high - 1), 13) if not suit_masks[flush_suit] & (1 << r)) if flush_high else 13
    nut_blocker = int(flush_holes > 0 and unseen_above == 0 and suit_counts[flush_suit] >= 2)
    board_window = combined_window = straight_help = 0
    out_mask = 0
    possible_straight = single_straight = 0
    single_missing = 0
    higher_single_missing = 0
    higher_two_combos = 0
    current_straight = STRAIGHT_HIGH[combined_mask]
    for window in STRAIGHT_MASKS:
        bc = (window & board_mask).bit_count()
        hc = (window & combined_mask).bit_count()
        board_window = max(board_window, bc)
        if hc > combined_window:
            combined_window = hc
            straight_help = hc - bc
        elif hc == combined_window:
            straight_help = max(straight_help, hc - bc)
        if hc == 4:
            out_mask |= window & ~combined_mask
        missing = window & ~board_mask
        needed = 5 - bc
        if needed <= 2:
            high = 5 if window == 0x100F else window.bit_length() + 1
            possible_straight = max(possible_straight, high)
            if needed <= 1:
                single_straight = max(single_straight, high)
                single_missing |= missing
                if high > current_straight:
                    higher_single_missing |= missing
            elif high > current_straight:
                first_bit = missing & -missing
                r1, r2 = first_bit.bit_length() + 1, (missing ^ first_bit).bit_length() + 1
                higher_two_combos += (4 - int(ar == r1) - int(br == r1)) * (4 - int(ar == r2) - int(br == r2))
    value = evaluate((*hole, *board))
    board_value = evaluate(board)
    category = value >> 20
    made_rank = (value >> 16) & 15
    kickers = tuple((value >> shift) & 15 for shift in (16, 12, 8, 4, 0))
    board_kickers = tuple((board_value >> shift) & 15 for shift in (16, 12, 8, 4, 0))
    counts = sorted(suit_counts, reverse=True)
    first_difference = next((i for i in range(5) if kickers[i] != board_kickers[i]), 5)
    private_kicker = kickers[first_difference] if first_difference < 5 else 0
    above_private = sum(4 - c - int(ar == r + 2) - int(br == r + 2) for r, c in enumerate(board_counts) if r + 2 > private_kicker)
    pair_ranks = [r + 2 for r, count in enumerate(board_counts) if count >= 2]
    board_pair_high = pair_ranks[-1] if pair_ranks else 0
    board_pair_low = pair_ranks[-2] if len(pair_ranks) > 1 else 0
    board_trip_rank = next((r + 2 for r in range(12, -1, -1) if board_counts[r] >= 3), 0)
    secondary = kickers[1]
    flush_combos = 0
    for s in range(4):
        count = suit_counts[s]
        if count >= 3:
            available = 13 - count - int(a & 3 == s) - int(b & 3 == s)
            if count == 3:
                flush_combos += available * (available - 1) // 2
            elif count == 5:
                remaining_cards = 50 - len(board)
                flush_combos += remaining_cards * (remaining_cards - 1) // 2
            else:
                # At least one card of this suit; board-only flush strength is
                # separately represented by the existing board-kicker fields.
                flush_combos += available * (50 - len(board) - available) + available * (available - 1) // 2
    pocket_full_combos = single_full_outs = 0
    if board_pair_high:
        for r, count in enumerate(board_counts):
            rank = r + 2
            available = 4 - count - int(ar == rank) - int(br == rank)
            if count == 1 and rank > made_rank:
                pocket_full_combos += available * (available - 1) // 2
            if count >= 2 and rank > made_rank:
                single_full_outs += available
    single_outs = 4 * single_missing.bit_count() - int(bool(single_missing & (1 << (ar - 2)))) - int(bool(single_missing & (1 << (br - 2))))
    higher_single_outs = 4 * higher_single_missing.bit_count() - int(bool(higher_single_missing & (1 << (ar - 2)))) - int(bool(higher_single_missing & (1 << (br - 2))))
    return (
        len(board), ar, br, int(ar == br), int(a & 3 == b & 3), ar - br,
        category, *kickers, board_value >> 20, *board_kickers, *ranks,
        board_mask.bit_count(), sum(c == 2 for c in board_counts), sum(c == 3 for c in board_counts), sum(c == 4 for c in board_counts),
        *counts, board_counts[ar - 2], board_counts[br - 2], int(ar > ranks[0]) + int(br > ranks[0]),
        sum(1 for i, count in enumerate(board_counts) if count and i + 2 > made_rank) if category in (1, 2, 3) else 0,
        total_suits[flush_suit], flush_holes, flush_high, unseen_above,
        board_window, combined_window, out_mask.bit_count(), straight_help,
        STRAIGHT_HIGH[board_mask], STRAIGHT_HIGH[combined_mask], int(len(board) == 5 and value == board_value),
        suit_counts[a & 3], suit_counts[b & 3], nut_blocker,
        board_mask, combined_mask, *board_counts,
        int(category == board_value >> 20), first_difference, private_kicker, above_private,
        board_pair_high, board_pair_low, board_trip_rank, int(ar == made_rank) + int(br == made_rank), int(ar == secondary) + int(br == secondary),
        sum(c == 1 and r + 2 > made_rank for r, c in enumerate(board_counts)),
        sum(c >= 2 and r + 2 > made_rank for r, c in enumerate(board_counts)), sum(c >= 2 and r + 2 > secondary for r, c in enumerate(board_counts)),
        possible_straight, single_straight, max(0, possible_straight - current_straight), max(0, single_straight - current_straight),
        higher_single_outs, higher_two_combos, single_outs,
        flush_combos, pocket_full_combos, single_full_outs,
        made_rank - ranks[0], secondary - ranks[0], int(ar == br and ar > board_pair_high), ar - ranks[0],
    )


FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
