# NFL Big Data Bowl 2027 — Combine Tracking Data Audit

**Scope:** the complete Kaggle release `nfl-big-data-bowl-2027` — 510 rookie prospects
(2023–2025 draft classes), 6,310 Combine drill events, 9 CSVs (~2.3 GB).
**Audit date:** 2026-11-21 (UTC). **Auditor:** DSP-AID.
**Bar:** every number below is produced by `scripts/audit.py` (single file, pandas only)
and asserted by `--selftest`; finding IDs (C1–C7, G8–G9) are stable keys in
[`findings_full_run.json`](findings_full_run.json).

## 0. Verdict

| Area | Result |
|---|---|
| Missing values | Tracking file: **0 null cells** (6,484,646 checked). Nulls elsewhere are opt-out/UDFA-shaped (§2) |
| Duplicate drills | **514 attempt labels spread over 3,107 event_ids (2,593 redundant copies) → 223,772 rows** collide under naive `(player, drill, attempt)` grouping (§3) |
| Tracking gaps | Intra-event frame gaps: **0**. But: 6+30+41 official times lack sensor clips; 153+84 attempt-number holes; 4 truncated clips; ball tracked in only 551/6,310 events (§4) |
| Label errors / spec breaks | `event_id` format split (136,097 rows); `dis`/`s` irreconcilable with `x`/`y` (93.5 % of frames); documented y-bounds violated on 50,360 rows; `player_play` `wc -l` trap +12,822; official column typo (§5) |
| Referential integrity | **Clean**: no duplicate rows/frames anywhere, no orphan FKs, id sets identical across the three 510-row tables (§6) |

Top-3 by blast radius for anyone building on this: **F1** (C4), **F2** (C6), **F3** (C6).

## 1. Inventory — the release as shipped [C0]

Row counts are parsed-CSV counts (pandas). `raw lines` = naive `wc -l`; it exceeds
parsed rows only where a quoted field embeds newlines (see G8). sha256 truncated here;
full digests + byte sizes in [`checksums.json`](checksums.json).

| File | Parsed rows | Raw lines | Bytes | sha256 (16) |
|---|---:|---:|---:|---|
| `players.csv` | 510 | 510 | 47,301 | `2cd7e0309e7aa405` |
| `combine_results.csv` | 510 | 510 | 41,020 | `e4773ef68167d508` |
| `player_career_successes.csv` | 510 | 510 | 13,801 | `0d4b8975debda4e2` |
| `games.csv` | 1,002 | 1,002 | 43,174 | `ca33543f6939af4e` |
| `combine_tracking.csv` | **463,189** | 463,189 | 83,788,174 | `541c69f6629f5f80` |
| `player_play.csv` | **314,197** | **327,019** | 146,879,042 | `68926a268c0a49e5` |
| `game_tracking_2023.csv` | 3,665,443 | 3,665,443 | 335,463,637 | `385e620b6b94e8cd` |
| `game_tracking_2024.csv` | 7,827,944 | 7,827,944 | 717,952,893 | `06b7f34304414306` |
| `game_tracking_2025.csv` | 11,160,903 | 11,160,903 | 1,022,575,093 | `419b6bc0388a9210` |

Headline reconciliation with the bounty brief: **510 rookies ✓** (identical id sets in
`players` / `combine_results` / `player_career_successes`), **6,310 drills ✓**
(`combine_tracking.event_id.nunique() == 6310`: 2023 → 1,966, 2024 → 2,171, 2025 → 2,173),
**9 CSVs ✓**.

```python
import pandas as pd
ct = pd.read_csv("data/combine_tracking.csv")
assert ct.event_id.nunique() == 6310 and len(pd.read_csv("data/players.csv")) == 510
```

**Provenance.** Official competition files need Kaggle login; the audit bytes were taken
from public mirror `prajwal02thakare/nfl-big-data-bowl-2027` (anonymous download, archive
size exactly equals Kaggle's reported `totalBytes` 512,718,152) and independently
cross-checked against the community data dictionary `maxquirk/big-data-bowl`
(parsed row counts match on every big file, incl. player_play 314,197 and the three
game-tracking files). If the house re-downloads from the competition and sha256s in
`checksums.json` differ, treat §G8/G9 as mirror-only and rerun `audit.py` — every other
finding is computed on combine-scope files shipped in `data/`.

## 2. Missing values [C3]

**`combine_tracking.csv`: zero nulls** — all 463,189 × 14 = 6,484,646 cells populated;
all timestamps parse as ISO-8601 with exactly 3 fractional digits (no format variants).

`combine_results.csv` — nulls are **opt-outs, not corruption** (players who skipped a
drill also have no stopwatch time):

| Column | Nulls (of 510) | Reading |
|---|---:|---|
| `three_cone` | 330 | only 180 official times exist; sensor tracked 160 (§4.3) |
| `bench_reps` | 318 | not tracked by NGS at all |
| `short_shuttle` | 309 | 201 official times; sensor 170 |
| `broad_jump` / `forty` / `ten_yd_split` / `vertical` | 87 / 86 / 85 / 72 | normal combine attendance pattern |
| `ngs_college_production_score`, `ngs_final_score` | 1 each | nfl_id 56040 (has 11 tracked events — score gap worth querying NFL, not a join bug) |

`players.csv`: `draft_round` / `draft_pick_within_round` / `draft_overall_pick` null on
**127** rows = undrafted free agents (documented pattern, safe to fill with sentinel).
`player_career_successes.csv`: 0 nulls. `player_play.csv`: 16 columns >90 % null —
position-stat semantics, worst `rush_yards` 314,175/314,197 (99.99 %); see §7.

```python
ct.isna().sum().sum() == 0                                # tracking: zero nulls
cr.isna().sum()["three_cone"] == 330                      # opt-out counts
pl.draft_round.isna().sum() == 127                        # UDFAs
```

## 3. Duplicate drills [C4] — finding F1, the one that breaks naive pipelines

The natural join/group key for drill-level work looks like
`(nfl_id, draft_year, drill_type, attempt)`. It is **not** unique: **514 of the 3,717
attempt keys map to multiple `event_id`s (up to 13)**, pulling in **223,772 rows —
48.3 % of the tracking file**. The duplicate windows are *separate sessions* on the same
combine day: median separation 0.69 h, range 0.08–1.42 h, zero overlap — i.e. re-runs or
re-captures recorded under the same attempt number, not two clocks on one run.

| drill_type | multi-event keys |
|---|---:|
| SKILL_DRILLS_DB | 147 |
| SKILL_DRILLS_WR | 114 |
| SKILL_DRILLS_DL | 112 |
| SKILL_DRILLS_OL | 88 |
| SKILL_DRILLS_TE | 46 |
| SKILL_DRILLS_LB | 7 |
| FORTY_YARD_DASH / SHORT_SHUTTLE / THREE_CONE_DRILL | 0 |

Good news: the three timed drills (40, shuttle, cone) are clean — there, attempt keys and
event_ids are 1:1. The trap is confined to the position skill drills.
**Rule: key drill attempts by `event_id`, never by `(player, drill, attempt)`.**

```python
key = ["nfl_id", "draft_year", "drill_type", "attempt"]
m = ct.groupby(key).event_id.nunique()
assert (m > 1).sum() == 514 and m.max() == 13
rows = ct.merge(m[m > 1].rename("n"), left_on=key, right_index=True)   # 223,772 rows
```

## 4. Tracking gaps [C5]

### 4.1 Frame continuity inside events — clean
Across all 456,328 consecutive frame pairs within `(event_id, entity_type)`: **0 gaps
> 0.15 s, 0 negative deltas**; every clip is a perfect 10 Hz run. Missing data does not
appear as dropped frames — it appears at clip and attempt boundaries (next three items).

### 4.2 Attempt-number holes [C5] — finding F5
Of **1,258** `(player, year, drill)` groups: **153 (12.2 %) never show attempt 1** (first
captured attempt is 2 or higher — the earlier run exists on film/stopwatch but has no
sensor clip), and **84 more have interior holes** (e.g. attempts `[1,2,3,5]`). Example:
nfl_id 55867 (2023) has a FORTY attempt-2 clip only; SHORT_SHUTTLE `[1,2,3,5]`. Anyone
computing "best of N attempts" per player will silently get fewer runs than they think.

### 4.3 Sensor vs stopwatch reconciliation [C5] — finding F4

| Drill | Official times | Players with sensor clip | Official **without** tracking | Tracked **without** official time |
|---|---:|---:|---:|---:|
| FORTY_YARD_DASH | 424 | 418 (793 attempt-clips, attempts up to 4) | **6** | 0 |
| THREE_CONE_DRILL | 180 | 160 | **30** | 10 |
| SHORT_SHUTTLE | 201 | 170 | **41** | 10 |

77 officially-timed runs have no sensor data (any combine→game movement model must
either drop them or fall back to stopwatch); 20 tracked runs have no official time
(unofficial re-runs — do not treat their sensor metrics as "official attempt" features).
Also: 74 players have FORTY sensor attempts ≥3 (13 with 4) while `combine_results` keeps
exactly one official `forty` — attempt-level sensor data is **not** 1:1 with the results
table.

### 4.4 Truncated clips [C5] — finding F9
**4 events ≤ 20 rows**, of which **3 shorter than 1.5 s** (a three-cone clip at 0.9 s /
10 frames cannot contain a real ~4–5 s run; two shuttle clips at 1.0–1.1 s). Event ids:
`20240303101000_1709502890626_000` (10 rows), `20240301100300_1709331900370_000` (11),
`20250228100500_1740793859122_000` (12), `20250302101000_1740950071992_000` (19).
Filter rule: drop or flag events under ~2 s duration.

### 4.5 Ball tracking coverage [C2/C5] — finding F7
BALL rows exist only for **551 of 6,310 events (8.7 %)**, confined to `SKILL_DRILLS_TE` /
`SKILL_DRILLS_WR` catch drills — **18 of 56 drill names** carry ball data. Ball windows
sit strictly *inside* player windows (0 starts before / ends after). The flip side is a
grain trap: BALL and PLAYER share identical `(event_id, time)` slots on all 32,095 ball
rows, so **`(event_id, time)` is not unique** — the file's key is
`(event_id, entity_type, time)`. A naive pivot/merge on `(event_id, time)` duplicates
32,095 rows; within ball events 311 frame-slots contain only the PLAYER row (ball
untracked mid-drill).

```python
assert ct.duplicated(["event_id", "time"]).sum() == 32095      # dual-entity slots
assert set(ct[ct.entity_type=="BALL"].drill_type) == {"SKILL_DRILLS_TE","SKILL_DRILLS_WR"}
```

## 5. Label errors & spec breaks [C6]

### 5.1 `event_id` format split — finding F3
2024/2025 ids match `^\d{14}_\d{13}_\d{3}$` (session-date + epoch-ms + seq): 327,092 rows.
All **1,966 events / 136,097 rows (29.4 %)** of the 2023 combine instead use Mongo ids:
`^[0-9a-f]{24}-\d{4}$`, e.g. `6414e5fde3d8e72dcdc8d394-2023`. Any code that derives the
drill date from `event_id[:8]` silently gets garbage (or a crash) on the whole 2023
cohort; parse the date from `time` instead.

### 5.2 Documented field bounds violated — finding F6
The data dictionary documents `x ∈ [0,120]`, `y ∈ [0,53.3]`. `x` holds (0 violations),
but **50,360 rows (10.9 %) have y outside [0, 53.3]** (min −11.46, max 66.28). Root cause
is a coordinate-origin convention, not sensor noise: the forty corridor is a straight
line at y ∈ **[−2.25, 2.70]** (per-event y std median 0.12) — combine lanes are indexed
off the sideline, not mapped onto game-field width. Bounds check by drill: FORTY 41,285,
WR 4,595, DB 2,949, TE 1,477, cone 24, OL 20, DL 10. Don't hard-code `0 ≤ y ≤ 53.3` on
combine data; normalize per drill/event instead.

### 5.3 `dis`, `s`, `a` are not step math — finding F2 (the modeling trap)
On the 456,328 exactly-consecutive frames: **426,473 (93.5 %) have |dis − hypot(Δx,Δy)| >
0.05 yd**, median absolute error **0.213 yd**; `s ≠ dis/0.1` on 451,932 frames (99 %).
Consequence: a full 40-yd dash accumulates only **36.74 yd** of `dis` (median) while the
x-span is **40.43 yd** (−9.1 %). `s/a/dis` are independent filtered sensor estimates —
internally plausible but irreconcilable with x/y. Pick one source for speed features and
stick with it; cross-source mixing produces inconsistent labels between combine and game
files (game `dis` **is** step-consistent in NGS convention).

### 5.4 Label/metadata errors
* **Column typo shipped in the release:** `player_play.csv` header contains
  `visitor_team_win_probility_added` (missing "a" in "probability") — real, official,
  reproducible from the shipped sample header; every downstream script must spell it wrong.
* **5 drill names appear under multiple position-group `drill_type`s** — `GAUNTLET_DRILL`
  (DB/TE/WR), `OVER_SHOULDER_ADJUST` (TE/WR), `PASS_RUSH_DRILL` (DL/LB),
  `SHORT_ZONE_BREAKS` (DL/LB), `BACK_PEDAL_AND_REACT` (DL/LB). `drill_type` is a *position
  group*, not a drill-family key; group analyses by `drill_name`.
* **`attempt` runs to 17** (skill-drill rep counts), and BALL rows inherit the player's
  attempt/drill labels — filtering `attempt <= 2` "official attempts only" is meaningless
  for skill drills.
* `draft_year` in tracking/results = draft class, consistent with `players.csv` (0
  mismatched rows); combine *session* date lives only in `time`.

## 6. Referential integrity [C7] — clean
* `players`, `combine_results`, `player_career_successes`: identical 510-id sets, no dupes.
* Tracking: 0 orphan `nfl_id`s; every one of the 510 players has ≥1 clip (min 6, max 22,
  mean 12.4 events/player).
* `games.csv`: 1,002 rows, no duplicate `game_id`; REG 816 / PRE 147 / POST 39 per §7.

## 7. Game-scope integrity [G8/G9] — full files required (checksums pinned)

* **G8 `player_play.csv`:** parsed **314,197** rows vs `wc -l` **327,019** →
  **12,822 lines are embedded newlines inside quoted `play_description`** — the release's
  biggest count-the-rows trap. Unique `(game_id, play_id, nfl_id)` ✓; 0 orphan games/
  players; `home_team_win_probability_added + visitor_team_win_probility_added` sums to 0
  on all 314,191 populated pairs (typo column notwithstanding); `down ∈ {0..4}` (961
  goal-to-go/OZ rows), quarter includes OT (5: 1,437).
* **G9 game tracking:** exact row counts per §1 table; **0** duplicate
  `(game_id, play_id, nfl_id, time)` frames; **0** orphan game_ids; **0** season/file
  mismatches (each `game_tracking_<year>.csv` contains exactly that season's games, incl.
  Jan-2024 playoff games in the 2023 file); mild sideline overshoot beyond ±1.5 yd of
  nominal bounds: x OOB 225 / 542 / 637, y OOB 5,894 / 12,183 / 15,873 (normal NGS
  behavior, clip if you assume strict bounds).

## 8. Guardrails for Big Data Bowl builders

1. Key drill attempts by `event_id`; `(player, drill, attempt)` collides on 48.3 % of rows [C4].
2. Use x/y displacement for distance/speed; never mix `dis`/`s` with x/y-derived features [C6].
3. Parse combine dates from `time`, never from `event_id` (two formats) [C6].
4. `(event_id, entity_type, time)` is the tracking grain; ball exists in 8.7 % of events only [C2/C5].
5. Drop events < 2 s and treat 77 stopwatch-only runs as missing-tracking by design [C5].
6. `0 ≤ y ≤ 53.3` documented bounds do not hold for combine coordinates (10.9 %) [C6].
7. Count `player_play` rows with a CSV parser, not `wc -l` (+12,822) [G8]; spell the win-prob column typo exactly as shipped [C6].

## Appendix — reproduction

Environment: Python 3.12.8, pandas 3.1.1, numpy 2.5.4 (any `pandas>=2.1` works; stable
APIs only). Full machine-generated run output: [`findings_full_run.json`](findings_full_run.json).

```bash
python scripts/audit.py --selftest          # bundle-only (shipped combine files): 59/74 + 15 skipped
# drop the 4 full game CSVs from Kaggle into data/, rerun:
python scripts/audit.py --selftest          # 74/74 checks passed, 0 skipped
python -m pytest tests/ -q                  # headline asserts, also runnable as plain python
```

Spot-check any single number above with the snippet in its section against `data/`
(combine scope) — every CORE finding is reproducible from the CSVs committed to this repo,
with no Kaggle token at verification time. FULL-scope (§7) numbers reproduce from the
official release pinned by sha256 in `checksums.json`.
