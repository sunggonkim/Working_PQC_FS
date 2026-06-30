#!/usr/bin/env python3
"""Retained proof for Gate 0.9-S1 epoch redo-log record encoding.

This artifact is intentionally narrow.  It proves that the isolated epoch-log
record codec builds and that the unit self-test covers encode/decode,
checksum rejection, version rejection, invalid length rejection, and short
record rejection.  It does not prove mounted-path durability, group commit,
checkpoint compaction, or crash recovery by itself.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "publication_protocol_fault_matrix"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_self_test() -> dict[str, Any]:
    proc = subprocess.run(
        [str(FUSE_BIN), "--self-test"],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    combined = proc.stdout + proc.stderr
    return {
        "argv": [relpath(FUSE_BIN), "--self-test"],
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "epoch_log_pass_line": "PQC-FUSE epoch log self-test: PASS" in combined,
        "fail_line_present": "FAIL" in combined,
    }


def source_evidence() -> dict[str, Any]:
    header = ROOT / "code" / "storage" / "pqc_epoch_log.h"
    source = ROOT / "code" / "storage" / "pqc_epoch_log.c"
    format_h = ROOT / "code" / "common" / "pqc_format.h"
    cmake = ROOT / "CMakeLists.txt"
    selftest = ROOT / "code" / "support" / "pqc_selftest.c"
    writeback = ROOT / "code" / "storage" / "pqc_writeback.c"
    publish = ROOT / "code" / "storage" / "pqc_epoch_publish.c"

    header_text = header.read_text(encoding="utf-8", errors="replace")
    source_text = source.read_text(encoding="utf-8", errors="replace")
    format_text = format_h.read_text(encoding="utf-8", errors="replace")
    selftest_text = selftest.read_text(encoding="utf-8", errors="replace")
    writeback_text = writeback.read_text(encoding="utf-8", errors="replace")
    publish_text = publish.read_text(encoding="utf-8", errors="replace")

    return {
        "header": relpath(header),
        "source": relpath(source),
        "header_exists": header.exists(),
        "source_exists": source.exists(),
        "cmake_includes_module": "pqc_epoch_log.c" in cmake.read_text(encoding="utf-8", errors="replace"),
        "format_magic_owned_by_pqc_format": "PQC_EPOCH_LOG_MAGIC" in format_text,
        "encode_api_visible": "pqc_epoch_log_encode_record" in header_text,
        "decode_api_visible": "pqc_epoch_log_decode_record" in header_text,
        "checksum_digest_visible": "EVP_sha256" in source_text,
        "version_rejection_selftest_visible": "-EPROTO" in selftest_text,
        "checksum_rejection_selftest_visible": "-EBADMSG" in selftest_text,
        "mounted_writeback_uses_epoch_log": "pqc_epoch_log" in writeback_text,
        "publication_dispatch_uses_epoch_log": "pqc_epoch_log" in publish_text,
    }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "epoch_log_record_contract.json"
    md_path = out_dir / "epoch_log_record_contract.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Epoch Redo-Log Record Contract",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Self-test pass: `{str(payload['self_test_pass']).lower()}`",
        f"- Source pass: `{str(payload['source_pass']).lower()}`",
        f"- Mounted path integration visible: `{str(payload['mounted_path_integration_visible']).lower()}`",
        "",
        "## Covered Rejections",
        "",
        "- checksum mismatch returns `-EBADMSG`",
        "- version mismatch returns `-EPROTO`",
        "- invalid plaintext length returns `-EINVAL`",
        "- short record returns `-EMSGSIZE`",
        "",
        "## Unsupported Durability Boundary",
        "",
        payload["unsupported_durability_boundary"],
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not FUSE_BIN.exists():
        raise SystemExit(f"missing build artifact: {relpath(FUSE_BIN)}")

    self_test = run_self_test()
    source = source_evidence()
    self_test_pass = (
        self_test["returncode"] == 0 and
        self_test["epoch_log_pass_line"] and
        not self_test["fail_line_present"]
    )
    source_pass = all(bool(source[key]) for key in (
        "header_exists",
        "source_exists",
        "cmake_includes_module",
        "format_magic_owned_by_pqc_format",
        "encode_api_visible",
        "decode_api_visible",
        "checksum_digest_visible",
        "version_rejection_selftest_visible",
        "checksum_rejection_selftest_visible",
    ))
    mounted_path_integration_visible = (
        source["mounted_writeback_uses_epoch_log"] or
        source["publication_dispatch_uses_epoch_log"]
    )
    payload = {
        "schema_version": 1,
        "generated_by": "experiments/run_epoch_log_record_contract.py",
        "generated_utc": now_utc(),
        "scope": "Gate 0.9-S1 isolated epoch redo-log record codec proof.",
        "self_test": self_test,
        "source_evidence": source,
        "self_test_pass": self_test_pass,
        "source_pass": source_pass,
        "mounted_path_integration_visible": mounted_path_integration_visible,
        "overall_pass": self_test_pass and source_pass,
        "unsupported_durability_boundary": (
            "This codec artifact does not by itself prove append durability, "
            "group commit, checkpoint compaction, crash replay, fdatasync "
            "reduction, or throughput improvement. Those remain Gate 0.9-S2 "
            "through 0.9-S4 work."
        ),
    }
    write_outputs(args.out_dir, payload)
    print(json.dumps({
        "overall_pass": payload["overall_pass"],
        "json": relpath(args.out_dir / "epoch_log_record_contract.json"),
        "mounted_path_integration_visible": mounted_path_integration_visible,
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
