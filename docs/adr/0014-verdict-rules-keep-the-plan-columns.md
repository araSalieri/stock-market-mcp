# Each Verdict rule names its stored columns; the columns keep their names

The signals table's `long_stop`, `long_target`, `short_stop` and `short_target` hold different levels per Verdict: a buy's default Trade Plan, a dip's picked Stop and Target (ADR-0012), a cut's Cut Stop and Buyback Price (ADR-0010), and a hold's two default Targets. Each Verdict's rule in `verdicts.py` names which columns it stores and scores against, so only the rules know the mapping and the columns keep their names.

## Considered Options

- Rename the columns to each Signal's own stop and target, migrated on open. Rejected: a migration and an ADR-0012 reversal to rename something only the Verdict rules read.

## Consequences

- A new Verdict states its columns in its rule; it doesn't need new columns while its levels fit a Stop and a Target.
- `seed_signals` keeps its `long` and `short` arguments.
