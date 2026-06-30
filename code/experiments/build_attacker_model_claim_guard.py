#!/usr/bin/env python3
"""Build the D4 attacker-model and non-claim guard.

The production source of truth is ``code/crypto/pqc_trust_boundary.c``.  This
guard checks that the trust-boundary table covers the required attacker-model
subjects and that paper-facing text does not imply defenses that the table marks
as non-claims.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "attacker_model_claim_guard"

TRUST_C = CODE / "crypto" / "pqc_trust_boundary.c"
TRUST_H = CODE / "crypto" / "pqc_trust_boundary.h"
SELFTEST_C = CODE / "support" / "pqc_selftest.c"
MAIN_C = CODE / "frontend" / "pqc_main.c"
CRYPTO_SOURCES = CODE / "crypto" / "sources.cmake"

REQUIRED_SUBJECTS = {
    "PQC_TRUST_SUBJECT_DAEMON_PROCESS": "daemon-process",
    "PQC_TRUST_SUBJECT_KERNEL_DRIVER_FUSE": "kernel-driver-fuse-stack",
    "PQC_TRUST_SUBJECT_BACKING_STORAGE": "backing-storage",
    "PQC_TRUST_SUBJECT_PRIVILEGED_LOCAL_ATTACKER": "privileged-local-attacker",
    "PQC_TRUST_SUBJECT_MULTI_TENANT_GPU": "multi-tenant-gpu",
    "PQC_TRUST_SUBJECT_GPU_SIDE_CHANNEL": "gpu-side-channel",
    "PQC_TRUST_SUBJECT_DEPLOYMENT_READINESS": "deployment-readiness",
}

REQUIRED_NON_CLAIMS = {
    "PQC_TRUST_SUBJECT_KERNEL_DRIVER_FUSE": ["kernel", "FUSE"],
    "PQC_TRUST_SUBJECT_PRIVILEGED_LOCAL_ATTACKER": ["privileged"],
    "PQC_TRUST_SUBJECT_MULTI_TENANT_GPU": ["multi-tenant"],
    "PQC_TRUST_SUBJECT_GPU_SIDE_CHANNEL": ["side-channel"],
    "PQC_TRUST_SUBJECT_DEPLOYMENT_READINESS": [
        "ready for deployment",
        "general-purpose filesystem",
    ],
}

REQUIRED_PAPER_PHRASES = {
    "trusts_kernel_fuse_driver_stack":
        "cannot obtain the mount credential or compromise the running kernel, FUSE daemon, CUDA driver, or cryptographic libraries",
    "privileged_attacker_nonclaim":
        "We make no claim against a privileged local attacker",
    "gpu_side_channel_nonclaim":
        "GPU side channel & No protection mechanism claimed",
    "limitations_privileged_sidechannel":
        "privileged-attacker defense, or side-channel evaluation",
    "deployment_nonclaim":
        "not deployed-filesystem or peak-throughput superiority",
    "conclusion_nonclaims":
        "GPU constant-time behavior",
}

SCAN_ROOTS = [
    ROOT / "Paper",
    ROOT / "README.md",
    ROOT / "docs",
    CODE,
]

DANGEROUS_PATTERNS = [
    ("side_channel", re.compile(r"side[- ]channel(?: protection| defense| resistance)?", re.I)),
    ("gpu_constant_time", re.compile(r"GPU constant[- ]time", re.I)),
    ("multi_tenant", re.compile(r"multi[- ]tenant(?: GPU)?(?: isolation| defense| scheduling)?", re.I)),
    ("privileged_attacker", re.compile(r"privileged(?: local)? attacker(?: defense)?", re.I)),
    ("compromised_kernel", re.compile(r"compromis(?:ed|e) (?:the )?(?:running )?kernel|compromised[- ]kernel", re.I)),
    ("fuse_stack_defense", re.compile(r"FUSE[- ]stack defense|FUSE stack defense|compromised .*FUSE", re.I)),
    ("deployment_ready", re.compile(r"deployment[- ]ready|ready for deployment|ready for production", re.I)),
    ("general_purpose_fs", re.compile(r"general[- ]purpose (?:secure )?filesystem", re.I)),
]

NEGATION_TERMS = (
    "no ",
    "not ",
    "not a ",
    "not as ",
    "does not ",
    "do not ",
    "must not ",
    "without ",
    "unless ",
    "out of scope",
    "non-claim",
    "nonclaim",
    "excluded",
    "exclude",
    "lacks ",
    "lacking ",
    "cannot ",
    "never ",
    "is not ",
    "are not ",
    "remains unsupported",
    "future work",
    "limitation",
    "limitations",
    "does not establish",
    "says nothing about",
    "requires platform isolation outside",
)

BENIGN_SUBSTRINGS = (
    "privileged benchmark",
    "privileged cache",
    "privileged drop_caches",
    "privileged loop probes",
    "privileged setup",
)


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_trust_rows(src: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"\{\s*\.subject\s*=\s*(PQC_TRUST_SUBJECT_[A-Z_]+),([\s\S]*?)\n\s*\}")
    for match in pattern.finditer(src):
        subject = match.group(1)
        body = match.group(2)
        row: dict[str, Any] = {"subject": subject}
        for field in (
            "name",
            "trusted_component",
            "implemented_boundary",
            "excluded_attacker",
            "failure_boundary",
            "non_claim_guard",
        ):
            strings = re.findall(rf"\.{field}\s*=\s*((?:\"[^\"]*\"\s*)+)", body)
            if strings:
                row[field] = " ".join(re.findall(r"\"([^\"]*)\"", strings[0]))
            else:
                row[field] = ""
        status = re.search(r"\.status\s*=\s*(PQC_TRUST_STATUS_[A-Z_]+)", body)
        claims_defense = re.search(r"\.claims_defense\s*=\s*([01])", body)
        deployment_ready = re.search(r"\.deployment_ready\s*=\s*([01])", body)
        row["status"] = status.group(1) if status else ""
        row["claims_defense"] = int(claims_defense.group(1)) if claims_defense else -1
        row["deployment_ready"] = int(deployment_ready.group(1)) if deployment_ready else -1
        rows.append(row)
    return rows


def iter_scan_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = relpath(path)
            if "/__pycache__/" in rel or rel.endswith((".pdf", ".png", ".jpg", ".o")):
                continue
            if rel.startswith("code/experiments/") and path.name != "build_attacker_model_claim_guard.py":
                continue
            if path.suffix.lower() not in {
                ".c", ".h", ".cu", ".py", ".md", ".tex", ".bib", ".cmake", ".txt",
            } and path.name not in {"README.md", "CMakeLists.txt"}:
                continue
            files.append(path)
    return sorted(set(files))


def is_guarded(context: str) -> bool:
    lower = context.lower()
    if any(item in lower for item in BENIGN_SUBSTRINGS):
        return True
    return any(term in lower for term in NEGATION_TERMS)


def scan_claims() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    unguarded: list[dict[str, Any]] = []
    for path in iter_scan_files():
        try:
            lines = read(path).splitlines()
        except UnicodeDecodeError:
            continue
        for idx, line in enumerate(lines, start=1):
            for name, pattern in DANGEROUS_PATTERNS:
                if not pattern.search(line):
                    continue
                start = max(0, idx - 4)
                end = min(len(lines), idx + 4)
                context = " ".join(lines[start:end])
                source_of_truth = path in {TRUST_C, Path(__file__).resolve()}
                item = {
                    "kind": name,
                    "path": relpath(path),
                    "line": idx,
                    "text": line.strip(),
                    "guarded": source_of_truth or is_guarded(context),
                }
                candidates.append(item)
                if not item["guarded"]:
                    unguarded.append(item)
    return candidates, unguarded


def build_report() -> dict[str, Any]:
    trust_src = read(TRUST_C)
    trust_hdr = read(TRUST_H)
    selftest_src = read(SELFTEST_C)
    main_src = read(MAIN_C)
    crypto_sources = read(CRYPTO_SOURCES)
    paper = "\n".join(read(path) for path in ROOT.glob("Paper/*.tex"))

    rows = parse_trust_rows(trust_src)
    rows_by_subject = {row["subject"]: row for row in rows}
    candidates, unguarded = scan_claims()

    source_checks = {
        "all_required_subjects_present":
            set(rows_by_subject) == set(REQUIRED_SUBJECTS),
        "all_names_match_subjects": all(
            rows_by_subject.get(subject, {}).get("name") == expected
            for subject, expected in REQUIRED_SUBJECTS.items()
        ),
        "all_rows_have_scope_fields": all(
            all(row.get(field) for field in (
                "trusted_component",
                "implemented_boundary",
                "excluded_attacker",
                "failure_boundary",
                "non_claim_guard",
            ))
            for row in rows
        ),
        "required_subjects_are_non_claims": all(
            rows_by_subject.get(subject, {}).get("status") ==
            "PQC_TRUST_STATUS_NON_CLAIM" and
            rows_by_subject.get(subject, {}).get("claims_defense") == 0 and
            rows_by_subject.get(subject, {}).get("deployment_ready") == 0
            for subject in REQUIRED_NON_CLAIMS
        ),
        "required_non_claim_guards_present": all(
            all(term in rows_by_subject.get(subject, {}).get("non_claim_guard", "")
                for term in terms)
            for subject, terms in REQUIRED_NON_CLAIMS.items()
        ),
        "trust_self_test_checks_required_non_claims":
            "require_non_claim(PQC_TRUST_SUBJECT_KERNEL_DRIVER_FUSE" in trust_src and
            "require_non_claim(PQC_TRUST_SUBJECT_PRIVILEGED_LOCAL_ATTACKER" in trust_src and
            "require_non_claim(PQC_TRUST_SUBJECT_MULTI_TENANT_GPU" in trust_src and
            "require_non_claim(PQC_TRUST_SUBJECT_GPU_SIDE_CHANNEL" in trust_src and
            "require_non_claim(PQC_TRUST_SUBJECT_DEPLOYMENT_READINESS" in trust_src,
        "self_test_wired_to_binary":
            "pqc_selftest_trust_boundary" in selftest_src and
            "PQC-FUSE trust boundary self-test" in main_src,
        "build_includes_trust_boundary_source":
            "crypto/pqc_trust_boundary.c" in crypto_sources,
        "header_exposes_single_trust_boundary_api":
            "pqc_trust_boundary_entries" in trust_hdr and
            "pqc_trust_boundary_find" in trust_hdr,
    }

    paper_checks = {
        key: phrase in paper
        for key, phrase in REQUIRED_PAPER_PHRASES.items()
    }

    claim_checks = {
        "dangerous_claim_candidates_scanned": len(candidates) > 0,
        "unguarded_d4_claim_count_zero": len(unguarded) == 0,
    }

    close_conditions = {
        "production_trust_boundary_table_complete": all(source_checks.values()),
        "paper_states_attacker_model_and_non_claims": all(paper_checks.values()),
        "dangerous_claim_lint_passes": all(claim_checks.values()),
    }

    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": all(close_conditions.values()),
        "scope": (
            "D4 attacker-model and non-claim guard. It verifies the production "
            "trust-boundary table and rejects unguarded claims about compromised "
            "kernel/driver/FUSE defense, privileged local attackers, multi-tenant "
            "GPU isolation, GPU side-channel protection, deployment readiness, "
            "and general-purpose filesystem status."
        ),
        "trust_boundary_rows": rows,
        "source_checks": source_checks,
        "paper_checks": paper_checks,
        "claim_checks": claim_checks,
        "close_conditions": close_conditions,
        "candidate_count": len(candidates),
        "unguarded_count": len(unguarded),
        "claim_candidates": candidates,
        "unguarded_claims": unguarded,
        "source_artifacts": [
            relpath(TRUST_C),
            relpath(TRUST_H),
            relpath(SELFTEST_C),
            relpath(MAIN_C),
            relpath(CRYPTO_SOURCES),
        ],
        "non_claims": [
            "no compromised kernel/driver/FUSE-stack defense",
            "no privileged local attacker defense",
            "no multi-tenant GPU isolation or cross-tenant defense",
            "no GPU side-channel protection",
            "no deployment-ready or ready-for-production claim",
            "no general-purpose filesystem status",
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Attacker-Model Claim Guard",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Scope: {report['scope']}",
        f"- Candidate count: `{report['candidate_count']}`",
        f"- Unguarded count: `{report['unguarded_count']}`",
        "",
        "## Close Conditions",
        "",
    ]
    for key, value in report["close_conditions"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    for section in ("source_checks", "paper_checks", "claim_checks"):
        lines.extend(["", f"## {section}", ""])
        for key, value in report[section].items():
            lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(["", "## Trust Boundary Rows", ""])
    for row in report["trust_boundary_rows"]:
        lines.extend(
            [
                f"### {row['name']}",
                "",
                f"- Subject: `{row['subject']}`",
                f"- Status: `{row['status']}`",
                f"- Trusted component: {row['trusted_component']}",
                f"- Excluded attacker: {row['excluded_attacker']}",
                f"- Failure boundary: {row['failure_boundary']}",
                f"- Non-claim guard: {row['non_claim_guard']}",
                "",
            ]
        )
    if report["unguarded_claims"]:
        lines.extend(["## Unguarded Claims", ""])
        for item in report["unguarded_claims"]:
            lines.append(
                f"- `{item['path']}:{item['line']}` `{item['kind']}`: "
                f"{item['text']}"
            )
    lines.extend(["## Non-Claims", ""])
    for item in report["non_claims"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = out / "attacker_model_claim_guard.json"
    md_path = out / "attacker_model_claim_guard.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_pass": report["overall_pass"],
                "json": relpath(json_path),
                "markdown": relpath(md_path),
                "candidate_count": report["candidate_count"],
                "unguarded_count": report["unguarded_count"],
                "close_conditions": report["close_conditions"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_complete and not report["overall_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
