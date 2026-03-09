# Dilution Result Pack

- source campaign: `/Volumes/Untitled/publishing_data/dilution_publish_v1`
- promoted at: `2026-03-09T10:44:51.625030+00:00`
- source git commit: `03bde771122fe0b3cd33613294a8a059e31e4baf`
- includes raw data: `no`
- file count: `40`

## Scope

This pack snapshots the derived outputs of a dilution campaign into a stable folder that can be cited,
reviewed, or copied elsewhere without depending on the original USB layout.

## Contents

- campaign-level ledgers and Tc summaries
- Binder/FSS analysis outputs when available
- per-size derived tables, figures, and manifests
- raw files only when `--include-raw` is used

## Caveats

- A partial campaign is still promotable; inspect `campaign_status.csv` before treating the pack as final.
- Raw graph and sweep files are excluded by default to keep packs small.
- The inventory and checksums live in `pack_manifest.json`.
- The `size_N32` undiluted point (`p_removed = 0.0`) has been refreshed from the follow-up 8-realization refinement run stored under `size_N32/manifests/dilution/dilution_n32_r8_seed42.json`.
