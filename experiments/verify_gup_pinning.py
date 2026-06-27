#!/usr/bin/env python3
"""Development-only helper for pinning / GUP trace experiments.

This file intentionally does not fabricate trace output.  It is a reminder of
the intended trace shape and a safety check that the current repository does
not treat GUP tracing as verified evidence.
"""

from __future__ import annotations

import shutil


def main() -> int:
    print("[*] Development-only helper: no trace output is fabricated here.")
    print("[*] The repository treats GUP tracing as unverified until a real root trace is captured.")
    if shutil.which("bpftrace") is None:
        print("[!] bpftrace is not installed.")
        return 1
    print("[*] A real run would look like:")
    print("    sudo bpftrace -e <trace> -c ./build/repro_malloc_register")
    print("[*] This helper does not execute the trace automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
