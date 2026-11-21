#!/usr/bin/env python3
"""Reproducible data audit of the NFL Big Data Bowl 2027 Combine tracking release.

Audits the 9 released CSVs (510 rookies, 6,310 Combine drill events, ~2.3 GB):
missing values, duplicate drills, tracking gaps, label errors. Every number in
AUDIT_REPORT.md is produced by this script; each finding carries a stable ID
(C1..C14, G1..G9) so the house can spot-check it.

Data placement
--------------
data/            Combine-scope files (shipped in this repo) - CORE sections run:
                 players.csv, combine_results.csv, player_career_successes.csv,
                 games.csv, combine_tracking.csv (+ *.sample_20k.csv headers).
data/ + Kaggle   Drop the four full game CSVs (player_play.csv,
                 game_tracking_{2023,2024,2025}.csv) next to them and the FULL
                 sections (G*) additionally run against the complete release.

Usage
-----
    python scripts/audit.py                    # audit data/, print report, write findings.json
    python scripts/audit.py --data DIR         # audit a different directory of CSVs
    python scripts/audit.py --selftest         # same, then assert every EXPECTED number (exit 1 on drift)

Dependencies: pandas, numpy only. No Kaggle token needed for the shipped data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Headline values verified on the full official release (2026-11-21 run).
# CORE: reproducible from the files shipped in data/. FULL: require the four full
# game CSVs from the Kaggle competition (checksums.json pins their identity).
EXPECTED = {
    "C1.tracking_rows": 463189, "C1.events": 6310, "C1.players_tracked": 510,
    "C2.exact_dup_rows": 0, "C2.dup_event_time_entity": 0, "C2.dup_event_time": 32095,
    "C2.ball_events": 551, "C2.player_only_slots_in_ball_events": 311,
    "C3.tracking_null_cells": 0, "C3.cr_forty_null": 86, "C3.cr_three_cone_null": 330,
    "C3.cr_short_shuttle_null": 309, "C3.cr_bench_null": 318, "C3.cr_vertical_null": 72,
    "C3.cr_broad_jump_null": 87, "C3.cr_ten_yd_null": 85, "C3.ngs_final_null": 1,
    "C3.players_udfa_null": 127,
    "C4.multi_event_attempt_keys": 514, "C4.rows_in_multi_event_keys": 223772,
    "C4.events_in_multi_event_keys": 3107, "C4.redundant_event_ids": 2593,
    "C4.max_events_per_attempt_key": 13, "C4.forty_multi_event_keys": 0,
    "C5.frame_gaps_gt_0.15s": 0, "C5.negative_time_deltas": 0, "C5.events_le_20_rows": 4,
    "C5.events_lt_1.5s": 3, "C5.attempt_groups": 1258, "C5.groups_missing_attempt_1": 153,
    "C5.groups_interior_holes": 84, "C5.forty_players_tracked": 418, "C5.forty_events": 793,
    "C5.forty_stopwatch_no_sensor": 6, "C5.three_cone_stopwatch_no_sensor": 30,
    "C5.three_cone_sensor_no_stopwatch": 10, "C5.shuttle_stopwatch_no_sensor": 41,
    "C5.shuttle_sensor_no_stopwatch": 10, "C5.ball_drill_names": 18, "C5.total_drill_names": 56,
    "C6.event_id_regex_rows": 327092, "C6.objectid_rows": 136097, "C6.objectid_events": 1966,
    "C6.time_frac_digit_variants": 1, "C6.y_out_of_bounds_rows": 50360, "C6.x_out_of_bounds_rows": 0,
    "C6.forty_y_min": -2.246992588, "C6.forty_y_max": 2.7,
    "C6.dis_mismatch_frames": 426473, "C6.consecutive_frames": 456328,
    "C6.speed_dis_mismatch_frames": 451932, "C6.drill_name_cross_type": 5,
    "C7.players_rows": 510, "C7.combine_results_rows": 510, "C7.career_rows": 510,
    "C7.games_rows": 1002, "C7.orphan_tracking_players": 0, "C7.draft_year_mismatch_rows": 0,
    "G8.player_play_parsed_rows": 314197, "G8.player_play_raw_lines": 327019,
    "G8.typo_column_present": 1, "G8.pp_dup_player_play_keys": 0, "G8.pp_orphan_games": 0,
    "G8.pp_wp_sum_violations": 0, "G8.pp_orphan_players": 0, "G8.embedded_newline_lines": 12822,
    "G8.games_season_type_counts": {"REG": 816, "PRE": 147, "POST": 39},
    "C7.games_dup_game_id": 0,
    "G9.gt2023_rows": 3665443, "G9.gt2024_rows": 7827944, "G9.gt2025_rows": 11160903,
    "G9.gt_orphan_games": 0, "G9.gt_season_file_mismatches": 0, "G9.gt_dup_frames": 0,
}

TS_EVENT_RE = re.compile(r"^\d{14}_\d{13}_\d{3}$")          # 2024/2025 event_id format
OBJECTID_RE = re.compile(r"^[0-9a-f]{24}-\d{4}$")           # 2023 event_id format (Mongo ObjectId + year suffix)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_line_count(path: Path) -> int:
    """Naive `wc -l` equivalent (counts \\n). Differs from parsed rows when a
    quoted field contains embedded newlines - see finding G8."""
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 26), b""):
            n += chunk.count(b"\n")
    return n - 1  # minus header line


def load(data_dir: Path):
    def rd(name, **kw):
        p = data_dir / name
        return pd.read_csv(p, **kw) if p.exists() else None
    return {
        "players.csv": rd("players.csv"),
        "combine_results.csv": rd("combine_results.csv"),
        "player_career_successes.csv": rd("player_career_successes.csv"),
        "games.csv": rd("games.csv"),
        "combine_tracking.csv": rd("combine_tracking.csv", dtype={"time": str}),
        "player_play.csv": rd("player_play.csv"),
        "game_tracking_2023.csv": rd("game_tracking_2023.csv", usecols=["game_id", "play_id", "nfl_id", "time", "x", "y", "event"]),
        "game_tracking_2024.csv": rd("game_tracking_2024.csv", usecols=["game_id", "play_id", "nfl_id", "time", "x", "y", "event"]),
        "game_tracking_2025.csv": rd("game_tracking_2025.csv", usecols=["game_id", "play_id", "nfl_id", "time", "x", "y", "event"]),
    }


def section_inventory(data_dir: Path, files: dict) -> dict:
    """Finding C0: exact inventory of the release (parsed row counts + sha256)."""
    out = {}
    for name, df in files.items():
        p = data_dir / name
        if df is None:
            continue
        out[name] = {"bytes": p.stat().st_size, "parsed_rows": int(len(df)),
                     "raw_lines_minus_header": raw_line_count(p), "sha256": sha256_file(p)}
    return out


def section_grain_and_keys(ct: pd.DataFrame) -> dict:
    """C1/C2: grain, duplicate rows/keys, dual-entity timestamp slots."""
    ct = ct.copy()
    ct["ts"] = pd.to_datetime(ct.time)
    keys_ent = ["event_id", "entity_type", "time"]
    ball_events = set(ct.loc[ct.entity_type == "BALL", "event_id"])
    be = ct[ct.event_id.isin(ball_events)]
    slots = be.groupby(["event_id", "ts"]).entity_type.nunique()
    return {
        "C1.tracking_rows": int(len(ct)),
        "C1.events": int(ct.event_id.nunique()),
        "C1.players_tracked": int(ct.nfl_id.nunique()),
        "C1.events_by_draft_year": {int(k): int(v) for k, v in ct.groupby("draft_year").event_id.nunique().items()},
        "C2.exact_dup_rows": int(ct.drop(columns="ts").duplicated().sum()),
        "C2.dup_event_time_entity": int(ct.duplicated(subset=keys_ent).sum()),
        "C2.dup_event_time": int(ct.duplicated(subset=["event_id", "time"]).sum()),
        "C2.ball_events": len(ball_events),
        "C2.player_only_slots_in_ball_events": int(((slots == 1) & (be.groupby(["event_id", "ts"]).entity_type.first() == "PLAYER")).sum()),
        "C2.ball_only_slots_in_ball_events": int(((slots == 1) & (be.groupby(["event_id", "ts"]).entity_type.first() == "BALL")).sum()),
    }


def section_missing_values(files: dict) -> dict:
    """C3: null counts per column (Combine scope)."""
    ct, cr, pl = files["combine_tracking.csv"], files["combine_results.csv"], files["players.csv"]
    career = files["player_career_successes.csv"]
    crn, pln = cr.isna().sum(), pl.isna().sum()
    return {
        "C3.tracking_null_cells": int(ct.isna().sum().sum()),
        "C3.combine_results_nulls": {k: int(v) for k, v in crn[crn > 0].items()},
        "C3.players_nulls": {k: int(v) for k, v in pln[pln > 0].items()},
        "C3.career_null_cells": int(career.isna().sum().sum()),
        "C3.cr_forty_null": int(crn["forty"]), "C3.cr_ten_yd_null": int(crn["ten_yd_split"]),
        "C3.cr_three_cone_null": int(crn["three_cone"]), "C3.cr_short_shuttle_null": int(crn["short_shuttle"]),
        "C3.cr_bench_null": int(crn["bench_reps"]), "C3.cr_vertical_null": int(crn["vertical"]),
        "C3.cr_broad_jump_null": int(crn["broad_jump"]), "C3.ngs_final_null": int(crn["ngs_final_score"]),
        "C3.players_udfa_null": int(pln["draft_round"]),
    }


def section_duplicate_drills(ct: pd.DataFrame) -> dict:
    """C4: same (player, year, drill_type, attempt) label recorded under multiple
    event_ids - duplicate drill labels with distinct session windows."""
    ct = ct.copy()
    ct["ts"] = pd.to_datetime(ct.time)
    key = ["nfl_id", "draft_year", "drill_type", "attempt"]
    ev_per_key = ct.groupby(key).event_id.nunique()
    multi = ev_per_key[ev_per_key > 1]
    rows_multi = int(ct.merge(multi.rename("n"), left_on=key, right_index=True).shape[0]) if len(multi) else 0
    win = ct.groupby("event_id").agg(t0=("ts", "min"), t1=("ts", "max"))
    gaps = []
    for _, grp in ct.merge(multi.rename("n"), left_on=key, right_index=True).groupby(key):
        fl = win.loc[grp.event_id.unique()]
        gaps.append((fl.t0.max() - fl.t1.min()).total_seconds() / 3600)
    gaps = pd.Series(gaps) if gaps else pd.Series(dtype=float)
    forty_keys = multi.reset_index().query("drill_type == 'FORTY_YARD_DASH'")
    return {
        "C4.distinct_attempt_keys": int(len(ev_per_key)),
        "C4.multi_event_attempt_keys": int(len(multi)),
        "C4.events_in_multi_event_keys": int(multi.sum()),
        "C4.redundant_event_ids": int(multi.sum() - len(multi)),
        "C4.rows_in_multi_event_keys": rows_multi,
        "C4.max_events_per_attempt_key": int(ev_per_key.max()),
        "C4.multi_keys_by_drill_type": {k: int(v) for k, v in multi.reset_index().drill_type.value_counts().items()},
        "C4.forty_multi_event_keys": int(len(forty_keys)),
        "C4.duplicate_session_gap_hours_min": round(float(gaps.min()), 2),
        "C4.duplicate_session_gap_hours_median": round(float(gaps.median()), 2),
        "C4.duplicate_session_gap_hours_max": round(float(gaps.max()), 2),
        "C4.duplicate_key_windows_overlapping": int((gaps < 0).sum()),
    }


def section_tracking_gaps(ct: pd.DataFrame, cr: pd.DataFrame) -> dict:
    """C5: frame continuity, truncated clips, attempt-number holes, sensor vs
    stopwatch drill coverage reconciliation, BALL tracking coverage."""
    ct = ct.copy()
    ct["ts"] = pd.to_datetime(ct.time)
    gaps = ct.groupby(["event_id", "entity_type"]).ts.diff().dt.total_seconds()
    ev = ct.groupby("event_id").agg(n=("ts", "size"), dur=("ts", lambda s: (s.max() - s.min()).total_seconds()))
    att = ct.groupby(["nfl_id", "draft_year", "drill_type"]).attempt.agg(lambda s: sorted(s.unique()))
    missing_first = int(att.apply(lambda a: a[0] != 1).sum())
    holes = int(att.apply(lambda a: a[0] == 1 and list(range(a[0], a[-1] + 1)) != a).sum())
    ply = ct[ct.entity_type == "PLAYER"]
    sensor_ids = {dt: set(g.nfl_id) for dt, g in ply.groupby("drill_type")}
    def recon(dt, col):
        s_ids, w_ids = sensor_ids.get(dt, set()), set(cr.loc[cr[col].notna(), "nfl_id"])
        return len(w_ids - s_ids), len(s_ids - w_ids)
    f_gap, f_rev = recon("FORTY_YARD_DASH", "forty")
    tc_gap, tc_rev = recon("THREE_CONE_DRILL", "three_cone")
    ss_gap, ss_rev = recon("SHORT_SHUTTLE", "short_shuttle")
    ball = ct[ct.entity_type == "BALL"]
    forty = ply[ply.drill_type == "FORTY_YARD_DASH"]
    return {
        "C5.frame_gaps_gt_0.15s": int((gaps > 0.15).sum()),
        "C5.negative_time_deltas": int((gaps < 0).sum()),
        "C5.events_le_20_rows": int((ev.n <= 20).sum()),
        "C5.events_lt_1.5s": int((ev.dur < 1.5).sum()),
        "C5.shortest_events": [
            {"event_id": str(i), "rows": int(r["n"]), "duration_s": round(float(r["dur"]), 1)}
            for i, r in ev.nsmallest(4, "n").iterrows()],
        "C5.attempt_groups": int(len(att)),
        "C5.groups_missing_attempt_1": missing_first,
        "C5.groups_interior_holes": holes,
        "C5.forty_players_tracked": int(forty.nfl_id.nunique()),
        "C5.forty_events": int(forty.event_id.nunique()),
        "C5.forty_stopwatch_no_sensor": f_gap, "C5.forty_sensor_no_stopwatch": f_rev,
        "C5.three_cone_stopwatch_no_sensor": tc_gap, "C5.three_cone_sensor_no_stopwatch": tc_rev,
        "C5.shuttle_stopwatch_no_sensor": ss_gap, "C5.shuttle_sensor_no_stopwatch": ss_rev,
        "C5.ball_rows": int(len(ball)), "C5.ball_events": int(ball.event_id.nunique()),
        "C5.ball_drill_names": int(ball.drill_name.nunique()),
        "C5.total_drill_names": int(ct.drill_name.nunique()),
        "C5.ball_drill_types": sorted(ball.drill_type.unique().tolist()),
    }


def section_label_errors(ct: pd.DataFrame) -> dict:
    """C6: spec/label problems - event_id format split, documented y-bounds
    violation, dis/s not reconcilable with x/y, drill_name crossing drill_type."""
    ct = ct.copy()
    pat = ct.event_id.str.match(TS_EVENT_RE)
    obj = ct.event_id.str.match(OBJECTID_RE)
    forty = ct[(ct.drill_type == "FORTY_YARD_DASH") & (ct.entity_type == "PLAYER")]
    fe = forty.groupby("event_id").agg(xmin=("x", "min"), xmax=("x", "max"), dis_sum=("dis", "sum"))
    fe["x_span"] = fe.xmax - fe.xmin
    ct_sorted = ct.sort_values(["event_id", "entity_type", "time"]).reset_index(drop=True)
    # exact 0.1 s consecutive pairs (no gaps): compare dis with hypot(dx,dy)
    ct_sorted["dx"] = ct_sorted.groupby(["event_id", "entity_type"]).x.diff()
    ct_sorted["dy"] = ct_sorted.groupby(["event_id", "entity_type"]).y.diff()
    dts = pd.to_datetime(ct_sorted.time).groupby([ct_sorted.event_id, ct_sorted.entity_type]).diff().dt.total_seconds()
    step = np.hypot(ct_sorted.dx, ct_sorted.dy)
    m = (dts == 0.1) & step.notna()
    diff = (ct_sorted.dis - step).abs()
    spd = (ct_sorted.s - ct_sorted.dis / 0.1).abs()
    name_types = ct.groupby("drill_name").drill_type.nunique()
    oob_y = ((ct.y < 0) | (ct.y > 53.3))
    return {
        "C6.event_id_regex_rows": int(pat.sum()),
        "C6.objectid_rows": int(obj.sum()),
        "C6.objectid_events": int(ct.loc[obj, "event_id"].nunique()),
        "C6.ts_format_events_2024": int(ct[ct.draft_year == 2024].event_id.nunique()),
        "C6.ts_format_events_2025": int(ct[ct.draft_year == 2025].event_id.nunique()),
        "C6.time_frac_digit_variants": int(ct.time.str.extract(r"\.(\d+)$")[0].str.len().nunique()),
        "C6.y_out_of_bounds_rows": int(oob_y.sum()),
        "C6.y_out_of_bounds_by_drill": {k: int(v) for k, v in ct.assign(o=oob_y).groupby("drill_type").o.sum().items() if v},
        "C6.x_out_of_bounds_rows": int(((ct.x < 0) | (ct.x > 120)).sum()),
        "C6.forty_y_min": float(forty.y.min()), "C6.forty_y_max": float(forty.y.max()),
        "C6.consecutive_frames": int(m.sum()),
        "C6.dis_mismatch_frames": int((m & (diff > 0.05)).sum()),
        "C6.dis_median_abs_error_yd": round(float(diff[m].median()), 3),
        "C6.speed_dis_mismatch_frames": int((m & (spd > 0.1)).sum()),
        "C6.forty_median_sum_dis_yd": round(float(fe.dis_sum.median()), 2),
        "C6.forty_median_x_span_yd": round(float(fe.x_span.median()), 2),
        "C6.drill_name_cross_type": int((name_types > 1).sum()),
        "C6.drill_name_cross_type_list": sorted(name_types[name_types > 1].index.tolist()),
    }


def section_referential(files: dict) -> dict:
    """C7: cross-table referential integrity inside the Combine scope."""
    ct, pl, cr = files["combine_tracking.csv"], files["players.csv"], files["combine_results.csv"]
    career = files["player_career_successes.csv"]
    mm = ct.merge(pl[["nfl_id", "draft_year"]], on="nfl_id", how="left", suffixes=("_trk", "_ply"))
    return {
        "C7.players_rows": int(len(pl)), "C7.players_dup_nfl_id": int(pl.nfl_id.duplicated().sum()),
        "C7.combine_results_rows": int(len(cr)), "C7.combine_results_dup_nfl_id": int(cr.nfl_id.duplicated().sum()),
        "C7.career_rows": int(len(career)), "C7.career_dup_nfl_id": int(career.nfl_id.duplicated().sum()),
        "C7.games_rows": int(len(files["games.csv"])), "C7.games_dup_game_id": int(files["games.csv"].game_id.duplicated().sum()),
        "C7.orphan_tracking_players": int((~ct.nfl_id.isin(pl.nfl_id)).sum()),
        "C7.players_without_tracking": int(len(set(pl.nfl_id) - set(ct.nfl_id))),
        "C7.draft_year_mismatch_rows": int((mm.draft_year_trk != mm.draft_year_ply).sum()),
        "C7.id_sets_identical": bool(set(pl.nfl_id) == set(cr.nfl_id) == set(career.nfl_id)),
    }


def section_game_scope(files: dict, data_dir: Path) -> dict:
    """G8/G9: full-release integrity (skipped unless the four big CSVs are present)."""
    out = {"status": "full game files absent - checks skipped (see checksums.json to fetch)"}
    pp, games = files["player_play.csv"], files["games.csv"]
    gts = {y: files[f"game_tracking_{y}.csv"] for y in (2023, 2024, 2025)}
    if all(v is not None for v in [pp, *gts.values()]):
        pp_path = data_dir / "player_play.csv"
        out.update({
            "G8.player_play_parsed_rows": int(len(pp)),
            "G8.player_play_raw_lines": raw_line_count(pp_path),
            "G8.embedded_newline_lines": raw_line_count(pp_path) - int(len(pp)),
            "G8.typo_column_present": int("visitor_team_win_probility_added" in pp.columns),
            "G8.pp_dup_player_play_keys": int(pp.duplicated(subset=["game_id", "play_id", "nfl_id"]).sum()),
            "G8.pp_orphan_games": int((~pp.game_id.isin(games.game_id)).sum()),
            "G8.pp_orphan_players": int((~pp.nfl_id.isin(files["players.csv"].nfl_id)).sum()),
            "G8.pp_wp_sum_violations": int(((pp.home_team_win_probability_added + pp.visitor_team_win_probility_added).abs() > 0.01).sum()),
            "G8.pp_cols_over_90pct_null": int((pp.isna().mean() > 0.9).sum()),
            "G8.pp_down_values": {int(k): int(v) for k, v in pp.down.value_counts(dropna=False).items()},
            "G8.games_season_type_counts": {k: int(v) for k, v in games.season_type.value_counts().items()},
        })
        season_map = games.set_index("game_id").season
        tot_orph, tot_seas, tot_dup = 0, 0, 0
        for y, gt in gts.items():
            out[f"G9.gt{y}_rows"] = int(len(gt))
            tot_orph += int((~gt.game_id.isin(games.game_id)).sum())
            tot_seas += int((gt.game_id.map(season_map) != y).sum())
            tot_dup += int(gt.duplicated(subset=["game_id", "play_id", "nfl_id", "time"]).sum())
            out[f"G9.gt{y}_x_oob"] = int(((gt.x < -1.5) | (gt.x > 121.5)).sum())
            out[f"G9.gt{y}_y_oob"] = int(((gt.y < -1.5) | (gt.y > 54.8)).sum())
        out.update({"G9.gt_orphan_games": tot_orph, "G9.gt_season_file_mismatches": tot_seas, "G9.gt_dup_frames": tot_dup})
        out["status"] = "full game files present - G8/G9 executed"
    return out


def run(data_dir: Path) -> dict:
    files = load(data_dir)
    missing_core = [n for n in ("combine_tracking.csv", "players.csv", "combine_results.csv", "player_career_successes.csv", "games.csv") if files[n] is None]
    if missing_core:
        sys.exit(f"[FATAL] missing core CSVs in {data_dir}: {missing_core}")
    findings = {"dataset": "NFL Big Data Bowl 2027 (Kaggle nfl-big-data-bowl-2027)", "sections": {}}
    findings["inventory"] = section_inventory(data_dir, files)
    findings["sections"].update(section_grain_and_keys(files["combine_tracking.csv"]))
    findings["sections"].update(section_missing_values(files))
    findings["sections"].update(section_duplicate_drills(files["combine_tracking.csv"]))
    findings["sections"].update(section_tracking_gaps(files["combine_tracking.csv"], files["combine_results.csv"]))
    findings["sections"].update(section_label_errors(files["combine_tracking.csv"]))
    findings["sections"].update(section_referential(files))
    findings["game_scope"] = section_game_scope(files, data_dir)
    return findings


def selftest(findings: dict) -> int:
    """Assert every EXPECTED headline number. G8/G9 keys are skipped (not failed)
    when the full game CSVs are absent from data/ - they need a Kaggle download."""
    flat = {**findings["sections"], **{k: v for k, v in findings["game_scope"].items() if not isinstance(v, str)}}
    failed, skipped = [], 0
    for key, want in EXPECTED.items():
        got = flat.get(key)
        if got is None and key.split(".")[0] in ("G8", "G9"):
            skipped += 1
            continue
        ok = np.isclose(got, want) if isinstance(want, float) and isinstance(got, (int, float)) else got == want
        if not ok:
            failed.append(f"{key}: expected {want!r}, got {got!r}")
    for f in failed:
        print("FAIL:", f)
    full_present = flat.get("G8.player_play_parsed_rows") is not None
    print(f"[selftest] {len(EXPECTED) - len(failed) - skipped}/{len(EXPECTED)} checks passed, {skipped} FULL-scope checks "
          + ("skipped (full game CSVs absent from data/)" if skipped else "executed"))
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "findings.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    findings = run(Path(args.data))
    with open(args.out, "w") as f:
        json.dump(findings, f, indent=1, default=str)
    print(json.dumps(findings, indent=1, default=str))
    if args.selftest:
        sys.exit(selftest(findings))


if __name__ == "__main__":
    main()
