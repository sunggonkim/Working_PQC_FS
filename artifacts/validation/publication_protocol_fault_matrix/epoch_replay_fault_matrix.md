# Epoch Replay Fault Matrix

- Generated: `2026-06-28T06:59:44Z`
- Overall pass: `true`

## replay_normal
- Pass: `true`
- Mutation: `None`
- Remount read matches: `true`
- Last compaction rc: `0`
- Journal repair records max: `0`

## replay_torn_tail
- Pass: `true`
- Mutation: `torn_tail`
- Remount read matches: `true`
- Last compaction rc: `0`
- Journal repair records max: `0`

## replay_duplicate_generation
- Pass: `true`
- Mutation: `duplicate_generation`
- Remount read matches: `false`
- Last compaction rc: `-17`
- Journal repair records max: `0`

## replay_journal_loss
- Pass: `true`
- Mutation: `journal_loss`
- Remount read matches: `true`
- Last compaction rc: `0`
- Journal repair records max: `4`

## Negative Claim Guard

This matrix proves mounted remount compaction for journal-backed committed epoch prefixes, torn-tail ignore, and duplicate generation rejection. It also proves recovery-time journal repair from a committed epoch prefix after deliberate journal-sidecar loss. It does not prove physical power-loss certification, throughput improvement, group-commit amortization, or rollback resistance.
