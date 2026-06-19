#!/usr/bin/env python3
"""
Create a minimal placeholder GGUF file for Cactus backend infrastructure
testing.  This is NOT a real model — it only allows the test binary to
locate a file path.  Real integration tests require a genuine GGUF model
(see generate_model.sh).
"""

import os
import struct
import sys


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3


def create_placeholder_gguf(output_path: str) -> None:
    print(f"Creating placeholder GGUF at: {output_path}")
    with open(output_path, "wb") as f:
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<I", GGUF_VERSION))   # version
        f.write(struct.pack("<Q", 0))               # tensor count
        f.write(struct.pack("<Q", 0))               # metadata kv count
    print(f"Placeholder created ({os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_cactus_model.gguf"
    try:
        create_placeholder_gguf(output)
        with open("model_path.txt", "w") as f:
            f.write(output + "\n")
        print(f"model_path.txt written: {output}")
        print("Note: This is a placeholder for infrastructure testing only.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
