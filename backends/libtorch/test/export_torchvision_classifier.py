"""Export the existing classifier fixture with the pinned runtime input contract."""
import argparse
from pathlib import Path
import sys

import torch
import torchvision.models as models

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib.torchscript_metadata import save_torchscript


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    versions = dict(line.split("=", 1) for line in (ROOT / "versions.env").read_text().splitlines()
                    if "=" in line and not line.startswith("#"))
    expected = versions["PYTORCH_VERSION"].strip('"\'')
    if torch.__version__.split("+", 1)[0] != expected:
        raise RuntimeError(f"Fixture export requires the pinned PyTorch {expected}; found {torch.__version__}")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
    example_input = torch.rand(1, 3, 224, 224)
    traced = torch.jit.trace(model, example_input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = (args.output_dir / "resnet18_traced.pt").resolve()
    save_torchscript(traced, model_path, (example_input,))
    (args.output_dir / "model_path.txt").write_text(str(model_path) + "\n")
    print(model_path)


if __name__ == "__main__":
    main()
