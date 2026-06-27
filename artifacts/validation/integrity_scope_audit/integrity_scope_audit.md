# Integrity Scope Audit

- Overall pass: `true`
- Paper scope pass: `true`
- Source scope pass: `true`
- Integrity parity bench pass: `true`

## High-Risk Paper Mentions
- `Paper/3_Design.tex:7` `per_file_content_merkle_tree` scoped=`true`: The integrity helper computes SHA-256/Merkle material for parity tests and for the committed-prefix anchor root; AEGIS-Q does not persist a per-file content Merkle tree. The freshness plane publishes an authenticated checkpoint and can target a pre-provisioned TPM NV index; this revision evaluates only fai
- `Paper/7_Implementation_Details.tex:6` `per_file_content_merkle_tree` scoped=`true`: These tests check functional equality for the helper used by the committed-prefix anchor root; they do not mean the filesystem persists a per-file content Merkle tree or provides side-channel protection. \begin{table}[t] \centering \caption{Implementation boundaries that are easy to misread without an ex

## Required Scope Phrases
- `committed-prefix anchor root` found=`true`
- `does not persist a per-file content Merkle tree` found=`true`
- `do not mean the filesystem persists a per-file content Merkle tree` found=`true`

## Integrity Bench
- Command: `/home/thor/skim/pqc_encrpyted_fs/build/bench_gpu_integrity --only-tests`
- Return code: `0`
- Stdout: `artifacts/validation/integrity_scope_audit/bench_gpu_integrity.stdout.txt`
- Stderr: `artifacts/validation/integrity_scope_audit/bench_gpu_integrity.stderr.txt`
