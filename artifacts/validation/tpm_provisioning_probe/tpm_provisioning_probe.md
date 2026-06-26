# TPM provisioning probe

This artifact records TPM/TCTI/provisioning state only. It does not establish PCR sealing, monotonic freshness, or hardware-backed recovery.

## Configuration

- NV index: `0x01500010`
- Requested TCTI: `<default>`
- Used sudo: `False`

## Tool availability

- tpm2_nvreadpublic: `True`
- tpm2_nvread: `True`
- tpm2_nvwrite: `True`
- tpm2_getcap: `True`

## Probe results

### `tpm2_getcap properties-fixed`

- TCTI: `<default>`
- Return code: `1`
- First stderr line: `ERROR:tcti:src/tss2-tcti/tcti-device.c:451:Tss2_Tcti_Device_Init() Failed to open specified TCTI device file /dev/tpmrm0: Permission denied `

### `tpm2_nvreadpublic 0x01500010`

- TCTI: `<default>`
- Return code: `1`
- First stderr line: `ERROR:tcti:src/tss2-tcti/tcti-device.c:451:Tss2_Tcti_Device_Init() Failed to open specified TCTI device file /dev/tpmrm0: Permission denied `

### `tpm2_pcrread sha256:0,1,2,3,4,5,6,7`

- TCTI: `<default>`
- Return code: `1`
- First stderr line: `ERROR:tcti:src/tss2-tcti/tcti-device.c:451:Tss2_Tcti_Device_Init() Failed to open specified TCTI device file /dev/tpmrm0: Permission denied `

## Conservative interpretation

- A zero return code from `tpm2_nvreadpublic` records that the NV index exists and exposes owner read/write attributes.
- A zero return code from `tpm2_pcrread` records current PCR values only; it is not PCR binding or PCR-drift rejection.
- No monotonic freshness update or recovery verdict is claimed by this probe.
