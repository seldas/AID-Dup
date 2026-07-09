"""
Scoring a dedup run against a stored ground truth -- a direct port of
askMyFAERS backend/app/services/dedup_metrics.py (including the
degenerate-side conventions for duplicate-free series). Pure Python.

Both inputs are duplicate groupings expressed as lists of case-id groups.
Produces pair-level precision/recall/F1 plus a group-level comparison
(matched / partial / new / fully-missed groups with per-case
matched/new/missed tags) that the UI renders directly.
"""

from itertools import combinations


def _pairs(groups: list) -> set:
    out = set()
    for group in groups:
        for a, b in combinations(sorted(set(group)), 2):
            out.add((a, b))
    return out


def _restrict_groups(groups: list, universe: set) -> tuple:
    """Drops case ids outside `universe` from every group (then drops groups
    left with < 2 members). Returns (restricted_groups, excluded_case_ids)."""
    restricted = []
    excluded = set()
    for group in groups:
        kept = [c for c in group if c in universe]
        excluded.update(c for c in group if c not in universe)
        if len(kept) >= 2:
            restricted.append(kept)
    return restricted, sorted(excluded)


def compute_metrics(system_groups: list, gt_groups: list, universe: set = None) -> dict:
    """
    Args:
        system_groups: list[list[str]] -- the run's duplicate groups.
        gt_groups: list[list[str]] -- the ground-truth duplicate groups.
        universe: optional set of all case ids the system could have grouped;
            GT cases outside it are excluded from scoring (reported in
            "gt_cases_outside_universe").

    Returns {"pair_level": {...}, "group_level": {...}, "groups": [...]}.
    """
    system_groups = [sorted(set(str(c) for c in g)) for g in system_groups if len(set(g)) >= 2]
    gt_groups = [sorted(set(str(c) for c in g)) for g in gt_groups if len(set(g)) >= 2]

    excluded_gt_cases = []
    if universe is not None:
        universe = {str(c) for c in universe}
        gt_groups, excluded_gt_cases = _restrict_groups(gt_groups, universe)

    pairs_system = _pairs(system_groups)
    pairs_gt = _pairs(gt_groups)
    tp = len(pairs_system & pairs_gt)
    fp = len(pairs_system - pairs_gt)
    fn = len(pairs_gt - pairs_system)
    # Degenerate-side conventions (matter for duplicate-free series, where an
    # empty ground truth is valid): a system that correctly finds nothing
    # scores 1.0 across the board instead of 0.0.
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if fp == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    gt_sets = [set(g) for g in gt_groups]
    sys_sets = [set(g) for g in system_groups]

    groups_out = []
    matched_gt_indexes = set()
    n_exact = n_partial = n_new_groups = 0
    n_new_cases = n_missed_cases = 0

    for sys_index, sys_set in enumerate(sys_sets):
        best_gt_index, best_overlap = None, set()
        for gt_index, gt_set in enumerate(gt_sets):
            overlap = sys_set & gt_set
            if len(overlap) > len(best_overlap):
                best_gt_index, best_overlap = gt_index, overlap

        if best_gt_index is None:
            group_status = "new"
            n_new_groups += 1
            cases = [{"case_id": c, "tag": "new"} for c in sorted(sys_set)]
            gt_match = None
            missed_in_group = 0
        else:
            matched_gt_indexes.add(best_gt_index)
            gt_set = gt_sets[best_gt_index]
            union = sys_set | gt_set
            gt_match = {
                "gt_group_index": best_gt_index,
                "gt_group_size": len(gt_set),
                "jaccard": round(len(best_overlap) / len(union), 4) if union else 0.0,
                "recall_coverage": round(len(best_overlap) / len(gt_set), 4) if gt_set else 0.0,
                "precision_coverage": round(len(best_overlap) / len(sys_set), 4) if sys_set else 0.0,
            }
            cases = (
                [{"case_id": c, "tag": "matched"} for c in sorted(best_overlap)]
                + [{"case_id": c, "tag": "new"} for c in sorted(sys_set - gt_set)]
                + [{"case_id": c, "tag": "missed"} for c in sorted(gt_set - sys_set)]
            )
            missed_in_group = len(gt_set - sys_set)
            if sys_set == gt_set:
                group_status = "matched"
                n_exact += 1
            else:
                group_status = "matched"
                n_partial += 1

        new_in_group = sum(1 for c in cases if c["tag"] == "new")
        n_new_cases += new_in_group
        n_missed_cases += missed_in_group
        groups_out.append({
            "source": "experiment",
            "group_status": group_status,
            "group_id": sys_index,
            "tag": None,
            "exp_group_size": len(sys_set),
            "n_matched_cases": len(best_overlap),
            "n_new_cases": new_in_group,
            "n_missed_cases": missed_in_group,
            "gt_match": gt_match,
            "cases": cases,
        })

    n_fully_missed = 0
    for gt_index, gt_set in enumerate(gt_sets):
        if gt_index in matched_gt_indexes:
            continue
        n_fully_missed += 1
        n_missed_cases += len(gt_set)
        groups_out.append({
            "source": "ground_truth",
            "group_status": "fully_missed",
            "group_id": None,
            "tag": None,
            "exp_group_size": 0,
            "n_matched_cases": 0,
            "n_new_cases": 0,
            "n_missed_cases": len(gt_set),
            "gt_match": {
                "gt_group_index": gt_index,
                "gt_group_size": len(gt_set),
                "jaccard": 0.0,
                "recall_coverage": 0.0,
                "precision_coverage": 0.0,
            },
            "cases": [{"case_id": c, "tag": "missed"} for c in sorted(gt_set)],
        })

    return {
        "pair_level": {
            "true_positive_pairs": tp,
            "false_positive_pairs": fp,
            "false_negative_pairs": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "group_level": {
            "system_groups": len(sys_sets),
            "gt_groups": len(gt_sets),
            "exact_matches": n_exact,
            "partial_matches": n_partial,
            "new_groups": n_new_groups,
            "fully_missed_groups": n_fully_missed,
            "new_cases": n_new_cases,
            "missed_cases": n_missed_cases,
        },
        "gt_cases_outside_universe": excluded_gt_cases,
        "groups": groups_out,
    }
