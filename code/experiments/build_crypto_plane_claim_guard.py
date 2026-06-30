#!/usr/bin/env python3
"""Build the D2 crypto-plane paper claim guard.

The guard is intentionally narrow: it verifies that paper-facing text keeps
bulk block encryption on AES-GCM, limits ML-KEM/Kyber to the key/session/envelope
plane, and does not imply that key-plane rekey work is on the ordinary mounted
read/write critical path.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "crypto_plane_separation"
TRACE_SMOKE = DEFAULT_OUT / "crypto_plane_trace_smoke.json"
SOURCE_EXTS = {".c", ".h", ".cu", ".cpp", ".hpp", ".py", ".sh"}


@dataclass
class Finding:
    path: str
    line: int
    kind: str
    pattern: str
    text: str
    guarded: bool


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def iter_document_paths() -> list[Path]:
    paths = sorted((ROOT / "Paper").glob("*.tex"))
    for path in (ROOT / "README.md",):
        if path.exists():
            paths.append(path)
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

    rows: list[tuple[int, str]] = []
    in_block = False
    for line_no, raw in enumerate(read(path).splitlines(), 1):
        text = raw
        while text:
            if in_block:
                end = text.find("*/")
                if end < 0:
                    rows.append((line_no, text.strip()))
                    text = ""
                else:
                    rows.append((line_no, text[:end].strip()))
                    text = text[end + 2:]
                    in_block = False
                continue

            slash = text.find("//")
            block = text.find("/*")
            if slash < 0 and block < 0:
                break
            if slash >= 0 and (block < 0 or slash < block):
                rows.append((line_no, text[slash + 2:].strip()))
                break

            end = text.find("*/", block + 2)
            if end < 0:
                rows.append((line_no, text[block + 2:].strip()))
                in_block = True
                break
            rows.append((line_no, text[block + 2:end].strip()))
            text = text[end + 2:]
    return [(line_no, line) for line_no, line in rows if line]


def iter_claim_lines() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in iter_document_paths():
        for line_no, line in enumerate(read(path).splitlines(), 1):
            rows.append({
                "path": relpath(path),
                "line": line_no,
                "kind": "document",
                "text": line.strip(),
            })

    code_root = ROOT / "code"
    for path in sorted(code_root.rglob("*")):
        if not path.is_file() or path.suffix not in SOURCE_EXTS:
            continue
        if "experiments" in path.parts:
            continue
        for line_no, line in iter_source_comment_lines(path):
            rows.append({
                "path": relpath(path),
                "line": line_no,
                "kind": "source_comment",
                "text": line.strip(),
            })
    return rows


NEGATION_RE = re.compile(
    r"\b(no|not|never|without|cannot|does not|do not|is not|are not|"
    r"rather than|only|optional|prototype|diagnostic|future|requires|"
    r"before|until|unless|boundary|scope|scoped|non-claim|limitation|"
    r"anti-pattern|fallback|not claimed|doesn't)\b|주장하지|아니|제한",
    re.IGNORECASE,
)


DANGEROUS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "bulk_data_direct_pqc",
        re.compile(
            r"\b(?:bulk|ordinary|file|block|data)[-\w ]{0,45}"
            r"(?:encrypt|encrypted|encryption)[-\w ]{0,45}"
            r"(?:PQC|ML-KEM|Kyber|ML-DSA)\b|"
            r"\b(?:PQC|ML-KEM|Kyber|ML-DSA)[-\w ]{0,45}"
            r"(?:encrypts|encrypted|encryption)[-\w ]{0,45}"
            r"(?:bulk|ordinary|file data|file blocks|blocks|data)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "gpu_crypto_mandatory_data_blocks",
        re.compile(
            r"\b(?:GPU|CUDA)[-\w ]{0,40}(?:mandatory|required|must|only|fixed)"
            r"[-\w ]{0,40}(?:AES-GCM|data plane|data blocks|file blocks)\b|"
            r"\b(?:data plane|data blocks|file blocks)[-\w ]{0,40}"
            r"(?:mandatory|required|must|only|fixed)[-\w ]{0,40}(?:GPU|CUDA)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "keyplane_rekey_rw_critical_path",
        re.compile(
            r"\b(?:rekey|key-plane|key plane|ML-KEM|Kyber)[-\w ]{0,50}"
            r"(?:ordinary|normal|read/write|write path|read path|critical path)"
            r"|"
            r"\b(?:ordinary|normal|read/write|write path|read path|critical path)"
            r"[-\w ]{0,50}(?:rekey|key-plane|key plane|ML-KEM|Kyber)\b",
            re.IGNORECASE,
        ),
    ),
]


def scan_claims() -> list[Finding]:
    findings: list[Finding] = []
    for row in iter_claim_lines():
        text = row["text"]
        if not text:
            continue
        for name, pattern in DANGEROUS_PATTERNS:
            if pattern.search(text):
                findings.append(Finding(
                    path=row["path"],
                    line=int(row["line"]),
                    kind=row["kind"],
                    pattern=name,
                    text=text,
                    guarded=bool(NEGATION_RE.search(text)),
                ))
    return findings


def load_trace_smoke() -> dict[str, Any]:
    if not TRACE_SMOKE.exists():
        return {"exists": False}
    payload = json.loads(read(TRACE_SMOKE))
    verdict = payload.get("verdict", {})
    ordinary = payload.get("runs", {}).get("ordinary", {}).get("trace", {})
    forced = payload.get("runs", {}).get("forced_keyplane", {}).get("trace", {})
    return {
        "exists": True,
        "path": relpath(TRACE_SMOKE),
        "overall_pass": verdict.get("overall_pass") is True,
        "source_plane_boundary_present":
            verdict.get("source_plane_boundary_present") is True,
        "ordinary_aes_encrypt_blocks":
            ordinary.get("data_aes_gcm_encrypt_blocks", 0),
        "ordinary_aes_decrypt_blocks":
            ordinary.get("data_aes_gcm_decrypt_blocks", 0),
        "ordinary_keyplane_batches": ordinary.get("keyplane_batches", 0),
        "ordinary_keyplane_refreshed_files":
            ordinary.get("keyplane_refreshed_files", 0),
        "forced_keyplane_batches": forced.get("keyplane_batches", 0),
        "forced_keyplane_refreshed_files":
            forced.get("keyplane_refreshed_files", 0),
        "ordinary_read_write_aes_gcm_only":
            verdict.get("ordinary_read_write_aes_gcm_only") is True,
        "forced_rekey_is_keyplane_only":
            verdict.get("forced_rekey_is_keyplane_only") is True,
    }


def paper_requirements() -> dict[str, Any]:
    paper = "\n".join(read(path) for path in sorted((ROOT / "Paper").glob("*.tex")))
    requirements = {
        "paper_says_data_plane_aes_gcm":
            bool(re.search(r"data plane is an AES-256-GCM block format", paper)),
        "paper_says_mlkem_limited_to_keyplane_envelope":
            bool(re.search(r"key-plane workflow contains optional batched ML-KEM envelope refresh", paper)),
        "paper_says_mlkem_does_not_encrypt_blocks":
            bool(re.search(r"does not encrypt ordinary file blocks", paper)),
        "paper_says_aes_data_writes_cpu_first":
            bool(re.search(r"AES-GCM data writes are latency-sensitive and remain CPU-first", paper)),
        "paper_says_rekey_not_hardware_lifecycle":
            bool(re.search(r"not a persistent KEM hierarchy or hardware-backed credential lifecycle", paper)),
    }
    return {
        "requirements": requirements,
        "all_present": all(requirements.values()),
    }


def build_report() -> dict[str, Any]:
    findings = scan_claims()
    unguarded = [finding for finding in findings if not finding.guarded]
    trace = load_trace_smoke()
    paper = paper_requirements()
    checks = {
        "trace_smoke_exists": trace.get("exists") is True,
        "trace_smoke_overall_pass": trace.get("overall_pass") is True,
        "ordinary_io_has_aes_blocks":
            int(trace.get("ordinary_aes_encrypt_blocks") or 0) > 0 and
            int(trace.get("ordinary_aes_decrypt_blocks") or 0) > 0,
        "ordinary_io_has_no_keyplane_batches":
            int(trace.get("ordinary_keyplane_batches") or 0) == 0 and
            int(trace.get("ordinary_keyplane_refreshed_files") or 0) == 0,
        "forced_rekey_records_keyplane_work":
            int(trace.get("forced_keyplane_batches") or 0) > 0 and
            int(trace.get("forced_keyplane_refreshed_files") or 0) > 0,
        "paper_crypto_plane_boundary_present": paper["all_present"],
        "no_unguarded_d2_overclaims": len(unguarded) == 0,
    }
    parent_d2_closed = all(checks.values())
    return {
        "generated_at": now_utc(),
        "generated_by": "code/experiments/build_crypto_plane_claim_guard.py",
        "trace_smoke": trace,
        "paper": paper,
        "checks": checks,
        "findings": [finding.__dict__ for finding in findings],
        "unguarded_findings": [finding.__dict__ for finding in unguarded],
        "unguarded_count": len(unguarded),
        "overall_pass": parent_d2_closed,
        "parent_d2_closed": parent_d2_closed,
        "negative_claim_guard": (
            "Bulk file data must remain described as AES-GCM block data. "
            "ML-KEM/Kyber may be described only as key/session/envelope-plane "
            "work, and ordinary read/write claims must not put rekey on the "
            "critical path unless mounted trace evidence changes."
        ),
    }


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "crypto_plane_claim_guard.json"
    md_path = out_dir / "crypto_plane_claim_guard.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")

    lines = [
        "# Crypto Plane Claim Guard",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Parent D2 closed: `{str(report['parent_d2_closed']).lower()}`",
        f"- Unguarded D2 overclaims: `{report['unguarded_count']}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{str(value).lower()}`")
    lines.extend(["", "## Negative Claim Guard", "", report["negative_claim_guard"], ""])
    if report["unguarded_findings"]:
        lines.extend(["", "## Unguarded Findings", ""])
        for finding in report["unguarded_findings"]:
            lines.append(
                f"- `{finding['path']}:{finding['line']}` "
                f"{finding['pattern']}: {finding['text']}"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = build_report()
    write_report(report, args.out_dir)
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "parent_d2_closed": report["parent_d2_closed"],
        "unguarded_count": report["unguarded_count"],
        "json": relpath(args.out_dir / "crypto_plane_claim_guard.json"),
        "md": relpath(args.out_dir / "crypto_plane_claim_guard.md"),
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
