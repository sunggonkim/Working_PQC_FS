#!/usr/bin/env python3
"""Build the D1 mount-key KDF verdict and claim guard.

This script inspects the production mount-key derivation path, records local
Argon2id/scrypt implementation availability, and checks that paper-facing text
matches the implemented KDF boundary.  It does not upgrade any security claim
beyond the production code and mounted-path smoke evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "kdf_crypto_plane"

KEYRING_C = CODE / "crypto" / "pqc_keyring.c"
RUNTIME_C = CODE / "runtime" / "pqc_runtime.c"
KEYRING_H = CODE / "crypto" / "pqc_keyring.h"
DESIGN_TEX = ROOT / "Paper" / "3_Design.tex"
DISCUSSION_TEX = ROOT / "Paper" / "10_Discussion_and_Limitations.tex"

SOURCE_COMMENT_EXTS = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".h",
    ".hpp",
    ".py",
    ".sh",
}


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def iter_doc_paths() -> list[Path]:
    paths = sorted((ROOT / "Paper").glob("*.tex"))
    paths.append(ROOT / "README.md")
    docs = ROOT / "docs"
    if docs.exists():
        paths.extend(sorted(docs.rglob("*.md")))
    return [path for path in paths if path.exists()]


def iter_source_comment_lines(path: Path) -> list[tuple[int, str]]:
    if path.suffix in {".py", ".sh"}:
        rows: list[tuple[int, str]] = []
        for line_no, raw in enumerate(read(path).splitlines(), 1):
            stripped = raw.lstrip()
            if stripped.startswith("#!") or not stripped.startswith("#"):
                continue
            rows.append((line_no, stripped[1:].strip()))
        return [(line_no, line) for line_no, line in rows if line]

    lines: list[tuple[int, str]] = []
    in_block = False
    for line_no, raw in enumerate(read(path).splitlines(), 1):
        text = raw
        while text:
            if in_block:
                end = text.find("*/")
                if end < 0:
                    lines.append((line_no, text.strip()))
                    text = ""
                else:
                    lines.append((line_no, text[:end].strip()))
                    text = text[end + 2:]
                    in_block = False
                continue

            slash = text.find("//")
            block = text.find("/*")
            if slash < 0 and block < 0:
                break
            if slash >= 0 and (block < 0 or slash < block):
                lines.append((line_no, text[slash + 2:].strip()))
                break

            end = text.find("*/", block + 2)
            if end < 0:
                lines.append((line_no, text[block + 2:].strip()))
                in_block = True
                break
            lines.append((line_no, text[block + 2:end].strip()))
            text = text[end + 2:]
    return [(line_no, line) for line_no, line in lines if line]


def iter_claim_lines() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in iter_doc_paths():
        for line_no, line in enumerate(read(path).splitlines(), 1):
            rows.append({
                "path": relpath(path),
                "line": line_no,
                "text": line.strip(),
                "kind": "document",
            })

    for path in sorted(CODE.rglob("*")):
        if not path.is_file() or path.suffix not in SOURCE_COMMENT_EXTS:
            continue
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        for line_no, line in iter_source_comment_lines(path):
            rows.append({
                "path": relpath(path),
                "line": line_no,
                "text": line.strip(),
                "kind": "source_comment",
            })
    return rows


def line_is_guarded(text: str) -> bool:
    return bool(re.search(
        r"\b(no|not|never|without|cannot|does not|do not|is not|are not|"
        r"rather than|only|legacy|compatib|historical|future|requires|"
        r"before|until|unless|open|blocked|unavailable|boundary|scope|"
        r"scoped|non-claim|limitation|not claimed)\b|아직|주장하지|"
        r"완전.*아니|제한",
        text,
        re.IGNORECASE,
    ))


def run_capture(cmd: list[str], *, cwd: Path = ROOT,
                input_text: str | None = None) -> dict[str, Any]:
    start = time.perf_counter_ns()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    end = time.perf_counter_ns()
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_ms": (end - start) / 1_000_000.0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def compile_probe(source: str, libs: list[str]) -> dict[str, Any]:
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return {
            "available": False,
            "reason": "no_c_compiler",
            "command": [],
            "returncode": None,
            "stderr": "",
        }
    with tempfile.TemporaryDirectory(prefix="aegisq_kdf_probe_") as tmp_s:
        tmp = Path(tmp_s)
        src = tmp / "probe.c"
        exe = tmp / "probe"
        src.write_text(source, encoding="utf-8")
        cmd = [cc, str(src), "-o", str(exe), *libs]
        result = run_capture(cmd)
        result["available"] = result["returncode"] == 0
        result["reason"] = "compile_ok" if result["available"] else "compile_failed"
        return result


def kdf_benchmark() -> dict[str, Any]:
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return {"available": False, "reason": "no_c_compiler", "rows": []}
    source = r'''
#include <openssl/evp.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int main(int argc, char **argv) {
    unsigned char out[32];
    const char *password = "aegisq-kdf-probe-password";
    const unsigned char salt[] = "PQC_FUSE_SALT_NIST";
    if (argc != 2)
        return 2;
    uint64_t start = now_ns();
    int ok = 0;
    if (strcmp(argv[1], "pbkdf2") == 0) {
        ok = PKCS5_PBKDF2_HMAC(password, strlen(password), salt,
                               sizeof(salt) - 1, 600000, EVP_sha256(),
                               sizeof(out), out);
    } else if (strcmp(argv[1], "scrypt") == 0) {
        ok = EVP_PBE_scrypt(password, strlen(password), salt, sizeof(salt) - 1,
                            32768, 8, 1, 64ull * 1024ull * 1024ull,
                            out, sizeof(out));
    } else {
        return 2;
    }
    uint64_t end = now_ns();
    if (ok != 1)
        return 1;
    printf("%s %.6f\n", argv[1], (double)(end - start) / 1000000.0);
    return 0;
}
'''
    with tempfile.TemporaryDirectory(prefix="aegisq_kdf_bench_") as tmp_s:
        tmp = Path(tmp_s)
        src = tmp / "kdf_bench.c"
        exe = tmp / "kdf_bench"
        src.write_text(source, encoding="utf-8")
        build = run_capture([cc, str(src), "-o", str(exe), "-lcrypto"])
        rows: list[dict[str, Any]] = []
        if build["returncode"] != 0:
            return {
                "available": False,
                "reason": "compile_failed",
                "build": build,
                "rows": rows,
            }
        for name in ("pbkdf2", "scrypt"):
            samples: list[float] = []
            failures: list[dict[str, Any]] = []
            for _ in range(3):
                run = run_capture([str(exe), name])
                if run["returncode"] != 0:
                    failures.append(run)
                    continue
                parts = run["stdout"].strip().split()
                if len(parts) == 2:
                    samples.append(float(parts[1]))
            rows.append({
                "kdf": name,
                "samples_ms": samples,
                "median_ms": sorted(samples)[len(samples) // 2] if samples else None,
                "failures": failures,
            })
        return {
            "available": True,
            "reason": "compile_ok",
            "build": build,
            "rows": rows,
            "parameters": {
                "pbkdf2": {
                    "digest": "SHA256",
                    "iterations": 600000,
                    "salt_len": len("PQC_FUSE_SALT_NIST"),
                    "output_len": 32,
                },
                "scrypt_probe": {
                    "N": 32768,
                    "r": 8,
                    "p": 1,
                    "maxmem_bytes": 64 * 1024 * 1024,
                    "output_len": 32,
                },
            },
        }


def detect_current_kdf() -> dict[str, Any]:
    keyring = read(KEYRING_C)
    runtime = read(RUNTIME_C)
    fmt = read(ROOT / "code" / "common" / "pqc_format.h")
    salt_match = re.search(
        r'const\s+uint8_t\s+salt\[\]\s*=\s*"([^"]+)"',
        keyring,
    )
    pbkdf2_call = "PKCS5_PBKDF2_HMAC" in keyring
    scrypt_call = "EVP_PBE_scrypt" in keyring
    metadata_format = "pqc_kdf_metadata_t" in fmt
    rand_metadata_salt = "RAND_bytes(meta->salt" in keyring
    iterations = None
    digest = None
    output_len = None
    call_match = re.search(
        r"PKCS5_PBKDF2_HMAC\([\s\S]{0,260?}?\n\s*(\d+),\s*EVP_(sha\d+)\(\),\s*(\d+),\s*g_master_key",
        keyring,
    )
    if call_match:
        iterations = int(call_match.group(1))
        digest = call_match.group(2).upper()
        output_len = int(call_match.group(3))
    else:
        if "600000" in keyring:
            iterations = 600000
        if "EVP_sha256()" in keyring:
            digest = "SHA256"
        if re.search(r"\b32,\s*g_master_key", keyring):
            output_len = 32

    if scrypt_call and metadata_format and rand_metadata_salt:
        algorithm = "scrypt-with-PBKDF2-legacy-compat"
        salt_scope = "per_filesystem_metadata"
        per_filesystem_salt_present = True
        salt_literal = None
        salt_len = 32
    else:
        algorithm = (
            "PBKDF2-HMAC-SHA256"
            if pbkdf2_call and digest == "SHA256" else "unknown"
        )
        salt_scope = "fixed_literal_in_source" if salt_match else "unknown"
        per_filesystem_salt_present = (
            "PQC_KDF_SALT" in keyring or "kdf_salt" in keyring or
            "salt_xattr" in keyring
        )
        salt_literal = salt_match.group(1) if salt_match else None
        salt_len = len(salt_match.group(1)) if salt_match else None

    return {
        "source": relpath(KEYRING_C),
        "runtime_source": relpath(RUNTIME_C),
        "function": "pqc_keyring_derive_master_key",
        "algorithm": algorithm,
        "pbkdf2_call_present": pbkdf2_call,
        "scrypt_call_present": scrypt_call,
        "kdf_metadata_format_present": metadata_format,
        "digest": digest,
        "iterations": iterations,
        "salt_literal": salt_literal,
        "salt_len": salt_len,
        "salt_scope": salt_scope,
        "per_filesystem_salt_present": per_filesystem_salt_present,
        "random_metadata_salt_present": rand_metadata_salt,
        "output_len": output_len,
        "runtime_requires_password": "PQC_MASTER_PASSWORD is required" in runtime,
        "runtime_log_pbkdf2": "PBKDF2-SHA256" in runtime,
        "runtime_reports_kdf_name": "pqc_keyring_kdf_name" in runtime,
    }


def toolchain_probe() -> dict[str, Any]:
    argon2_src = "#include <argon2.h>\nint main(void){return 0;}\n"
    scrypt_src = (
        "#include <openssl/evp.h>\n"
        "int main(void){unsigned char out[32];"
        "return EVP_PBE_scrypt(\"p\",1,(const unsigned char*)\"s\",1,"
        "32768,8,1,67108864,out,sizeof(out)) == 1 ? 0 : 1;}\n"
    )
    return {
        "cc": shutil.which("cc") or shutil.which("gcc"),
        "pkg_config_libargon2": run_capture(
            ["pkg-config", "--modversion", "libargon2"]
        ) if shutil.which("pkg-config") else {
            "returncode": None,
            "stdout": "",
            "stderr": "pkg-config missing",
        },
        "argon2_header_compile": compile_probe(argon2_src, ["-largon2"]),
        "openssl_scrypt_compile": compile_probe(scrypt_src, ["-lcrypto"]),
        "openssl_version": run_capture(["openssl", "version", "-a"])
        if shutil.which("openssl") else {
            "returncode": None,
            "stdout": "",
            "stderr": "openssl missing",
        },
    }


def scan_claim_text() -> dict[str, Any]:
    specs: list[tuple[str, str]] = [
        ("argon2id_deployment",
         r"\bArgon2id\b|\bArgon2\b"),
        ("hardware_backed_credential_release",
         r"hardware[- ]backed credential|hardware[- ]bound credential|"
         r"hardware credential release|hardware-released"),
        ("full_offline_attack_resistance",
         r"full offline|offline[- ]attack resistance|offline attack "
         r"resistance|offline[- ]guessing resistance|resists offline|"
         r"prevents offline|eliminates offline"),
        ("d1_or_deployment_closure",
         r"\bD1\b.{0,24}\b(closed|closure|complete)|security[- ]ready|"
         r"ready for deployment|deployment-ready"),
        ("pbkdf2_as_current_default",
         r"Mount-key derivation uses PBKDF2|uses PBKDF2 with "
         r"HMAC-SHA256|PBKDF2 is the default|default PBKDF2"),
    ]

    candidates: list[dict[str, Any]] = []
    for row in iter_claim_lines():
        text = row["text"]
        if not text:
            continue
        for name, pattern in specs:
            if not re.search(pattern, text, re.IGNORECASE):
                continue
            guarded = line_is_guarded(text)
            if name == "pbkdf2_as_current_default":
                guarded = guarded or bool(re.search(
                    r"legacy|compatib|not the default|pre[- ]metadata|"
                    r"existing roots|previous verdict|before changing",
                    text,
                    re.IGNORECASE,
                ))
            item = {
                "kind": row["kind"],
                "category": name,
                "path": row["path"],
                "line": row["line"],
                "text": text,
                "guarded": guarded,
            }
            candidates.append(item)

    return {
        "candidate_count": len(candidates),
        "unguarded_count": sum(1 for item in candidates
                               if not item["guarded"]),
        "candidates": candidates,
    }


def scan_paper_kdf_text() -> dict[str, Any]:
    paper_text = "\n".join(read(path) for path in sorted((ROOT / "Paper").glob("*.tex")))
    design_text = read(DESIGN_TEX)
    discussion_text = read(DISCUSSION_TEX) if DISCUSSION_TEX.exists() else ""
    checks = {
        "design_states_openssl_scrypt":
            bool(re.search(r"OpenSSL\s+scrypt", design_text, re.IGNORECASE)),
        "design_states_versioned_kdf_record":
            "\\texttt{.pqc\\_kdf}" in design_text,
        "design_states_random_32_byte_per_filesystem_salt":
            bool(re.search(
                r"random\s+32-byte\s+per-filesystem\s+salt",
                design_text,
                re.IGNORECASE,
            )),
        "design_states_scrypt_n":
            bool(re.search(r"N\s*=\s*32768", design_text)),
        "design_states_scrypt_r":
            bool(re.search(r"r\s*=\s*8", design_text)),
        "design_states_scrypt_p":
            bool(re.search(r"p\s*=\s*1", design_text)),
        "design_states_scrypt_maxmem":
            bool(re.search(r"64\s*~?MiB|67108864", design_text)),
        "design_states_pbkdf2_legacy_boundary":
            bool(re.search(
                r"PBKDF2-HMAC-SHA256\s+compatibility|legacy.*PBKDF2|"
                r"PBKDF2.*legacy|PBKDF2.*compatibility",
                design_text,
                re.IGNORECASE,
            )),
        "design_states_pbkdf2_not_new_root_default":
            bool(re.search(
                r"PBKDF2\s+is\s+not\s+the\s+new-root\s+default",
                design_text,
                re.IGNORECASE,
            )),
        "paper_states_password_or_offline_guessing_limit":
            bool(re.search(
                r"password entropy|offline[- ]guessing|weak-password|"
                r"weak password",
                paper_text,
                re.IGNORECASE,
            )),
        "discussion_keeps_no_hardware_credential_release":
            bool(re.search(
                r"no hardware-backed credential release|never hardware-released",
                discussion_text,
                re.IGNORECASE,
            )),
        "no_stale_pbkdf2_default_sentence":
            "Mount-key derivation uses PBKDF2 with HMAC-SHA256" not in paper_text,
    }
    return {
        "checks": checks,
        "overall_pass": all(checks.values()),
        "failed_checks": [key for key, value in checks.items() if not value],
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    current = payload["current_kdf"]
    tc = payload["toolchain"]
    verdict = payload["verdict"]
    paper = payload["paper_kdf_text"]
    claims = payload["claim_scan"]
    lines = [
        "# KDF Current-State Verdict",
        "",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Verdict: `{verdict['decision']}`",
        f"- Parent D1 closed: `{str(payload['parent_d1_closed']).lower()}`",
        "",
        "## Current Production KDF",
        "",
        f"- Algorithm: `{current['algorithm']}`",
        f"- Iterations: `{current['iterations']}`",
        f"- Salt scope: `{current['salt_scope']}`",
        f"- Salt length: `{current['salt_len']}`",
        f"- Output length: `{current['output_len']}`",
        "",
        "## Toolchain",
        "",
        f"- Argon2 header compile: `{str(tc['argon2_header_compile']['available']).lower()}`",
        f"- OpenSSL scrypt compile: `{str(tc['openssl_scrypt_compile']['available']).lower()}`",
        "",
        "## Paper-Facing KDF Text",
        "",
        f"- Paper text pass: `{str(paper['overall_pass']).lower()}`",
        f"- Failed paper checks: `{', '.join(paper['failed_checks']) if paper['failed_checks'] else 'none'}`",
        "",
        "## Claim Guard",
        "",
        f"- Guard candidates: `{claims['candidate_count']}`",
        f"- Unguarded candidates: `{claims['unguarded_count']}`",
        "",
        "## Next Production Step",
        "",
        f"- {verdict['next_production_step']}",
        "",
        "## Boundary",
        "",
        payload["boundary"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_payload() -> dict[str, Any]:
    current = detect_current_kdf()
    toolchain = toolchain_probe()
    bench = kdf_benchmark()
    claims = scan_claim_text()
    paper_kdf_text = scan_paper_kdf_text()

    argon2_available = (
        toolchain["argon2_header_compile"].get("available") is True and
        toolchain["pkg_config_libargon2"].get("returncode") == 0
    )
    scrypt_available = toolchain["openssl_scrypt_compile"].get("available") is True

    scrypt_implemented = (
        current["algorithm"] == "scrypt-with-PBKDF2-legacy-compat"
    )
    if scrypt_implemented:
        source_checks = {
            "mount_key_function_found": current["function"] in read(KEYRING_C),
            "scrypt_derivation_detected": current["scrypt_call_present"] is True,
            "kdf_metadata_format_detected":
                current["kdf_metadata_format_present"] is True,
            "per_filesystem_salt_present":
                current["per_filesystem_salt_present"] is True,
            "random_metadata_salt_present":
                current["random_metadata_salt_present"] is True,
            "legacy_pbkdf2_compatibility_present":
                current["pbkdf2_call_present"] is True,
            "runtime_requires_password": current["runtime_requires_password"] is True,
            "runtime_reports_kdf_name":
                current["runtime_reports_kdf_name"] is True,
        }
    else:
        source_checks = {
            "mount_key_function_found": current["function"] in read(KEYRING_C),
            "pbkdf2_hmac_sha256_detected":
                current["algorithm"] == "PBKDF2-HMAC-SHA256",
            "iterations_detected": current["iterations"] == 600000,
            "fixed_salt_detected":
                current["salt_scope"] == "fixed_literal_in_source",
            "per_filesystem_salt_absent":
                current["per_filesystem_salt_present"] is False,
            "runtime_requires_password":
                current["runtime_requires_password"] is True,
        }
    toolchain_checks = {
        "argon2id_probe_completed":
            toolchain["argon2_header_compile"].get("returncode") is not None,
        "scrypt_probe_completed":
            toolchain["openssl_scrypt_compile"].get("returncode") is not None,
        "replacement_path_available": scrypt_available or argon2_available,
    }
    benchmark_checks = {
        "benchmark_compiled": bench.get("available") is True,
        "pbkdf2_sampled": any(
            row.get("kdf") == "pbkdf2" and row.get("samples_ms")
            for row in bench.get("rows", [])
        ),
        "scrypt_sampled_if_available": (
            (not scrypt_available) or any(
                row.get("kdf") == "scrypt" and row.get("samples_ms")
                for row in bench.get("rows", [])
            )
        ),
    }
    claim_checks = {
        "no_unguarded_d1_dangerous_claims":
            claims["unguarded_count"] == 0,
    }
    paper_checks = {
        "paper_kdf_text_matches_production_boundary":
            paper_kdf_text["overall_pass"] is True,
    }

    if scrypt_implemented:
        decision = "scrypt_metadata_implemented_check_paper_guard"
        next_step = (
            "Proceed to D2 only after the mounted smoke, paper KDF text, and "
            "negative-claim guard all pass."
        )
    elif scrypt_available:
        decision = "replace_pbkdf2_with_openssl_scrypt_next"
        next_step = (
            "Implement a production scrypt mount-key path using OpenSSL "
            "EVP_PBE_scrypt, a per-filesystem random salt stored in explicit "
            "root KDF metadata, and a compatibility/migration decision for "
            "existing PBKDF2 metadata."
        )
    elif argon2_available:
        decision = "replace_pbkdf2_with_argon2id_next"
        next_step = (
            "Implement a production Argon2id mount-key path with per-filesystem "
            "salt and retained mount-latency/memory parameters."
        )
    else:
        decision = "pbkdf2_temporarily_scoped_no_local_replacement_header"
        next_step = (
            "Keep PBKDF2 scoped temporarily and install Argon2id/scrypt "
            "development support before changing the production key path."
        )

    checks = {
        **source_checks,
        **toolchain_checks,
        **benchmark_checks,
        **claim_checks,
        **paper_checks,
        "next_step_named": bool(next_step),
    }
    parent_closed = (
        scrypt_implemented and
        all(source_checks.values()) and
        all(toolchain_checks.values()) and
        all(benchmark_checks.values()) and
        all(claim_checks.values()) and
        all(paper_checks.values())
    )
    if parent_closed:
        decision = "scrypt_metadata_paper_guard_complete"
        next_step = (
            "Move to D2-S0: add mounted-path crypto-plane route evidence so "
            "data-plane AES-GCM and key-plane PQC remain visibly separated."
        )
    return {
        "schema_version": 2,
        "generated_by": "code/experiments/build_kdf_current_state_verdict.py",
        "generated_utc": now_utc(),
        "current_kdf": current,
        "toolchain": toolchain,
        "benchmark": bench,
        "claim_scan": claims,
        "paper_kdf_text": paper_kdf_text,
        "source_checks": source_checks,
        "toolchain_checks": toolchain_checks,
        "benchmark_checks": benchmark_checks,
        "claim_checks": claim_checks,
        "paper_checks": paper_checks,
        "verdict": {
            "decision": decision,
            "argon2id_available_for_build": argon2_available,
            "scrypt_available_for_build": scrypt_available,
            "next_production_step": next_step,
            "smallest_safe_change": (
                "Keep the versioned KDF metadata record and derive the mount "
                "key from per-filesystem salt before making any stronger "
                "security claim."
            ),
        },
        "checks": checks,
        "overall_pass": all(checks.values()),
        "parent_d1_closed": parent_closed,
        "boundary": (
            "This verdict allows only the implemented OpenSSL scrypt new-root "
            "path, the explicit PBKDF2 legacy compatibility boundary, and the "
            "password/offline-guessing limitation.  It does not claim "
            "Argon2id deployment, hardware-backed credential release, full "
            "offline-attack resistance, or broader security readiness."
        ),
        "source_files": [relpath(KEYRING_C), relpath(KEYRING_H), relpath(RUNTIME_C)],
        "paper_file_observed": relpath(DESIGN_TEX),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    json_path = out_dir / "kdf_current_state_verdict.json"
    md_path = out_dir / "kdf_current_state_verdict.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    write_markdown(payload, md_path)
    print(json.dumps({
        "overall_pass": payload["overall_pass"],
        "decision": payload["verdict"]["decision"],
        "parent_d1_closed": payload["parent_d1_closed"],
        "json": relpath(json_path),
        "failed_checks": [
            key for key, value in payload["checks"].items() if not value
        ],
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
