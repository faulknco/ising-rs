# Dilution Result Pack

- source campaign: `/Volumes/Untitled/publishing_data/dilution_publish_v1`
- promoted at: `2026-03-09T01:41:13.077943+00:00`
- source git commit: `1f9aa478a4abb5c7714077f386c85440465e78e5`
- includes raw data: `no`
- file count: `39`

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
- `size_N32` contains a noisy undiluted (`p_removed = 0.0`) susceptibility-peak estimate; use the Binder/FSS summaries and realization-level tables together before treating that single `Tc` mean as final.
