#!/usr/bin/env python3
"""Build a reproducibility bundle for the current paper artifact set.

This script does not invent results or regenerate unsupported measurements.
It packages the checked-in paper sources, figure assets, experiment scripts,
and artifact logs into a single manifest with sha256 hashes.  Optionally it
can rebuild the paper PDF and verify that the final document still has exactly
12 pages.

The intended use is:

  python3 experiments/build_repro_bundle.py --out-dir artifacts/repro_bundle

The bundle is intentionally conservative: it records what exists today and
the commands needed to reproduce the paper entry points, but it does not
pretend that every unsupported benchmark has been rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "repro_bundle"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pdf_page_count(pdf_path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"could not determine page count for {pdf_path}")


def collect_files() -> list[Path]:
    roots = [
        ROOT / "Paper",
        ROOT / "Paper" / "Figures",
        ROOT / "artifacts",
        ROOT / "experiments",
    ]
    explicit = [
        ROOT / "CMakeLists.txt",
        ROOT / "cuda_aead.cu",
        ROOT / "cuda_aead.h",
        ROOT / "cuda_integrity.cu",
        ROOT / "cuda_integrity.h",
        ROOT / "cuda_pqc.cu",
        ROOT / "cuda_pqc.h",
        ROOT / "pqc_admission.c",
        ROOT / "pqc_admission.h",
        ROOT / "pqc_anchor.c",
        ROOT / "pqc_anchor.h",
        ROOT / "pqc_block_job.h",
        ROOT / "pqc_file_key.c",
        ROOT / "pqc_file_key.h",
        ROOT / "pqc_fuse.c",
        ROOT / "README.md",
        ROOT / "SUBMISSION_CHECKLIST.md",
    ]

    allowed_suffixes = {
        ".tex", ".bib", ".pdf", ".png", ".jpg", ".jpeg", ".json", ".jsonl",
        ".csv", ".txt", ".log", ".bt", ".cu", ".c", ".h", ".py", ".sh",
        ".cpp", ".hpp", ".md", ".yaml", ".yml",
    }

    files: set[Path] = set()
    for path in explicit:
        if path.is_file():
            files.add(path)

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.is_relative_to(ROOT / "Paper" / "Previous paper"):
                continue
            if path == ROOT / "Paper" / "previous paper.pdf":
                continue
            if path.is_relative_to(ROOT / "artifacts" / "repro_bundle"):
                continue
            if path.suffix.lower() not in allowed_suffixes and path.name not in {
                "Makefile",
                "Dockerfile",
            }:
                continue
            files.add(path)

    return sorted(files)


def maybe_rebuild_paper() -> None:
    subprocess.run(["cmake", "--build", "build", "--parallel", "2"], cwd=ROOT, check=True)
    paper_dir = ROOT / "Paper"
    for cmd in [
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
    ]:
        subprocess.run(cmd, cwd=paper_dir, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--rebuild-paper",
        action="store_true",
        help="Rebuild Paper/main.pdf before hashing and verify that it is 12 pages.",
    )
    args = parser.parse_args()

    if args.rebuild_paper:
        maybe_rebuild_paper()

    paper_pdf = ROOT / "Paper" / "main.pdf"
    pages = pdf_page_count(paper_pdf)
    if pages != 12:
        raise SystemExit(f"Paper/main.pdf has {pages} pages, expected 12")

    files = collect_files()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_files = []
    sha_lines = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        entry = {"path": rel, "sha256": digest, "size_bytes": size}
        manifest_files.append(entry)
        sha_lines.append(f"{digest}  {rel}")

    commands = {
        "paper_build": [
            "cd Paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex",
        ],
        "microbench": [
            "python3 experiments/run_verified_microbench.py --runs 3 --out artifacts/validation/microbench",
            "python3 experiments/plot_verified_microbench.py",
        ],
        "qos": [
            "python3 experiments/run_m5_admission_sweep.py",
            "python3 experiments/run_qos_gpu.py",
            "PQC_SUDO_PASSWORD=<password> python3 experiments/run_qos_cupti_pm_fuse_bridge.py --out-dir artifacts/validation/qos_cupti_pm_fuse_bridge --samples 8 --cupti-iterations 600 --sampling-interval 1000000 --max-samples 512",
        ],
        "crash_recovery": [
            "python3 experiments/run_crash_replay_e8.py",
            "python3 experiments/run_fuse_tamper_rejection.py",
            "python3 experiments/run_app_recovery_bundle.py",
            "PQC_SUDO_PASSWORD=<password> python3 experiments/run_combined_durability_bundle.py --out-dir artifacts/validation/combined_durability_bundle",
            "PQC_SUDO_PASSWORD=<password> python3 experiments/run_sqlite_syscall_crash_tpm.py --out-dir artifacts/validation/sqlite_syscall_crash_tpm --when 1 2 3",
        ],
    }

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "paper_pages": pages,
        "paper_pdf": {
            "path": "Paper/main.pdf",
            "sha256": sha256_file(paper_pdf),
        },
        "commands": commands,
        "files": manifest_files,
    }

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out_dir / "sha256sums.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    (args.out_dir / "paper_pages.txt").write_text(f"{pages}\n", encoding="utf-8")

    print(json.dumps({
        "out_dir": str(args.out_dir),
        "files": len(manifest_files),
        "paper_pages": pages,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
