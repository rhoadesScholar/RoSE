# Pre-commit Hooks

This directory contains custom pre-commit hooks for the RoSE project.

## update-version.py

This hook automatically updates the version in `CITATION.cff` when committing to the main branch.

### What it does

- Generates a PEP440-compliant date-based version tag (format: YYYY.M.D.HHMM)
- Updates the `version` field in `CITATION.cff`
- Updates the `date-released` field in `CITATION.cff`
- Stages the updated `CITATION.cff` file

### When it runs

- Only on commits to the `main` branch
- Skips if commit message contains `[skip ci]`
- Runs during the pre-commit stage

This replaces the previous CI/CD workflow behavior where version updates were pushed back to the repository after the fact. Now version updates happen as part of the commit process itself.
