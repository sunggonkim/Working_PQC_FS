# O4 energy/thermal result

Overall pass: `True`

This is not an energy-efficiency claim, not a fixed-clock proof, and not a replacement for per-mode energy experiments.

## Representative rows

| Row | Samples | Mean VIN W | Max tj C | Estimated VIN J |
| --- | ---: | ---: | ---: | ---: |
| `sqlite_qos_methodology` | 250 | 29.14 | 48.3 | 728.4 |
| `keyplane_rekey_methodology` | 987 | 27.52 | 48.3 | 2716.5 |
| `frozen_aegisq_contract` | 840 | 28.41 | 49.0 | 11932.8 |
| `frozen_dmcrypt_contract` | 839 | 27.38 | 42.6 | 11487.8 |
| `frozen_gocryptfs_contract` | 4116 | 27.94 | 48.6 | 11498.2 |
| `frozen_plaintext_contract` | 4114 | 27.42 | 47.3 | 11282.4 |

## Checks

- `sqlite_log_present`: `True`
- `keyplane_log_present`: `True`
- `frozen_rows_present`: `True`
- `paper_mentions_o4`: `True`
- `paper_no_energy_efficiency_win_claim`: `True`
- `checklist_o4_done`: `True`
