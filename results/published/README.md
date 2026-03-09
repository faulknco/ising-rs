# Published Result Packs

Each subdirectory in this tree should represent one versioned result pack.

A pack should contain:

- raw inputs or references to raw inputs
- manifests
- derived tables
- generated figures
- a short README describing scope, provenance, and limitations

Use `analysis/scripts/promote_validation_result_pack.py` or
`analysis/scripts/promote_dilution_result_pack.py` to snapshot a workflow into this directory with
a pack README and a checksum manifest.
