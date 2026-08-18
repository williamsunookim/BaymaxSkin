"""Export the BatchNorm-free KIMLAB point-cloud model to ONNX."""

from argparse import ArgumentParser
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn


INPUT_SIZE = 16
OUTPUT_SIZE = 5568


class JointPointCloud(nn.Module):
    """MLP from ``verify_series_model.py`` with every BatchNorm removed."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(INPUT_SIZE, 8),
            nn.SiLU(),
            nn.Linear(8, 16),
            nn.SiLU(),
            nn.Linear(16, 64),
            nn.SiLU(),
            nn.Linear(64, 1024),
            nn.SiLU(),
            nn.Linear(1024, OUTPUT_SIZE),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(inputs)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("KIMLAB_figure_model_best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test/static/models"),
    )
    parser.add_argument("--stem", default="KIMLAB_figure_model_best")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = JointPointCloud()
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    fp32_path = args.output_dir / f"{args.stem}.onnx"
    quantized_path = args.output_dir / f"{args.stem}_quant.onnx"

    example_input = torch.zeros(1, INPUT_SIZE, dtype=torch.float32)
    torch.onnx.export(
        model,
        example_input,
        fp32_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(fp32_path))

    quantize_dynamic(
        fp32_path,
        quantized_path,
        weight_type=QuantType.QUInt8,
    )
    onnx.checker.check_model(onnx.load(quantized_path))

    print(f"FP32:      {fp32_path} ({fp32_path.stat().st_size / 1024**2:.2f} MiB)")
    print(
        f"Quantized: {quantized_path} "
        f"({quantized_path.stat().st_size / 1024**2:.2f} MiB)"
    )


if __name__ == "__main__":
    main()
