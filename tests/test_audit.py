"""Headline asserts for the BDB 2027 combine audit - runnable via pytest or plain python.

    python -m pytest tests/ -q          # or:  python tests/test_audit.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import audit  # noqa: E402


def _findings():
    return audit.run(ROOT / "data")


def test_headline_counts():
    s = _findings()["sections"]
    assert s["C1.events"] == 6310                      # bounty brief: 6,310 drills
    assert s["C1.tracking_rows"] == 463189
    assert s["C1.players_tracked"] == 510              # bounty brief: 510 rookies
    assert s["C2.exact_dup_rows"] == 0 and s["C2.dup_event_time_entity"] == 0
    assert s["C2.dup_event_time"] == 32095             # dual-entity BALL/PLAYER slots


def test_duplicate_drills_and_gaps():
    s = _findings()["sections"]
    assert s["C4.multi_event_attempt_keys"] == 514     # F1: duplicate attempt labels
    assert s["C4.rows_in_multi_event_keys"] == 223772  # 48.3% of the file
    assert s["C4.forty_multi_event_keys"] == 0         # timed drills clean
    assert s["C5.frame_gaps_gt_0.15s"] == 0            # perfect 10 Hz inside events
    assert s["C5.groups_missing_attempt_1"] == 153 and s["C5.groups_interior_holes"] == 84
    assert s["C5.forty_stopwatch_no_sensor"] == 6      # F4: official time, no clip
    assert s["C5.three_cone_stopwatch_no_sensor"] == 30
    assert s["C5.shuttle_stopwatch_no_sensor"] == 41


def test_label_errors():
    s = _findings()["sections"]
    assert s["C6.objectid_rows"] == 136097             # F3: 2023 event_id format split
    assert s["C6.y_out_of_bounds_rows"] == 50360       # F6: documented bounds violated
    assert s["C6.dis_mismatch_frames"] == 426473       # F2: dis irreconcilable with x/y
    assert s["C6.speed_dis_mismatch_frames"] == 451932
    assert s["C6.drill_name_cross_type"] == 5
    header = (ROOT / "data" / "player_play.sample_20k.csv").read_text().splitlines()[0]
    assert "visitor_team_win_probility_added" in header  # official typo, G8


def test_referential_integrity_and_selftest():
    f = _findings()
    s = f["sections"]
    assert s["C7.id_sets_identical"] and s["C7.orphan_tracking_players"] == 0
    assert s["C3.tracking_null_cells"] == 0
    assert audit.selftest(f) == 0                      # every EXPECTED number holds


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if fails else 0)
