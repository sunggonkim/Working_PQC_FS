# TPM provisioning probe

This artifact records TPM/TCTI/provisioning state only. It does not establish PCR sealing, monotonic freshness, or hardware-backed recovery.

## Configuration

- NV index: `0x01500010`
- Requested TCTI: `<default>`
- Used sudo: `True`

## Tool availability

- tpm2_nvreadpublic: `True`
- tpm2_nvread: `True`
- tpm2_nvwrite: `True`
- tpm2_getcap: `True`

## Probe results

### `sudo -S tpm2_getcap properties-fixed`

- TCTI: `<default>`
- Return code: `0`
- `TPM2_PT_FAMILY_INDICATOR: / raw: 0x322E3000 / value: "2.0"`
- `TPM2_PT_MANUFACTURER: / raw: 0x58595A20 / value: "XYZ "`
- `TPM2_PT_VENDOR_STRING_2: / raw: 0x6654504D / value: "fTPM"`
- `TPM2_PT_PCR_COUNT: / raw: 0x18 / TPM2_PT_PCR_SELECT_MIN:`
- `TPM2_PT_NV_INDEX_MAX: / raw: 0x800 / TPM2_PT_MEMORY:`
- `TPM2_PT_MODES: / raw: 0x1 / value: TPMA_MODES_FIPS_140_2`
- First stderr line: `[sudo] password for thor:`

### `sudo -S tpm2_nvreadpublic 0x01500010`

- TCTI: `<default>`
- Return code: `0`
- `0x1500010:`
- `name: 000bc7029adb2cd158ef9cb07fd317d316fd1baec2a43cea7cfcf6f32c7060c1353a`
- `hash algorithm:`
- `friendly: sha256`
- `friendly: ownerwrite|ownerread`
- `attributes:`
- `value: 0xB`
- `value: 0x20002`
- `size: 88`

### `sudo -S tpm2_pcrread sha256:0,1,2,3,4,5,6,7`

- TCTI: `<default>`
- Return code: `0`
- PCR rows: `8`

## Conservative interpretation

- A zero return code from `tpm2_nvreadpublic` records that the NV index exists and exposes owner read/write attributes.
- A zero return code from `tpm2_pcrread` records current PCR values only; it is not PCR binding or PCR-drift rejection.
- No monotonic freshness update or recovery verdict is claimed by this probe.
