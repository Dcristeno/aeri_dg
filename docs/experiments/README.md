# Experiment Records

This folder stores structured experiment records.

## Files

- `runs.csv`: early R1-oriented run ledger. It predates the later merge sweeps, so use `docs/CURRENT_STATUS.md` and `docs/merge/*RESULTS*.md` for current merge conclusions.

## Recording Rule

New structured runs should include:

- date
- branch
- run name
- training objective
- sampler
- best epoch
- R1/R5/R10/RSum/mAP/mINP
- checkpoint/config path
- SwanLab URL when available
- short conclusion
