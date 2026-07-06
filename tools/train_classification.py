import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cme_aqa.dataset import HandPoseDataset, discover_files
from cme_aqa.model import build_fusion_model
from cme_aqa.training import FusionClassificationTrainer
from cme_aqa.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train CME-AQA fusion classifier.")
    parser.add_argument("--fpv_json", required=True)
    parser.add_argument("--tpv_json", required=True)
    parser.add_argument("--fpv_f", required=True)
    parser.add_argument("--tpv_f", required=True)
    parser.add_argument("--label_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int, default=61)
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_align", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_variant", choices=["l2", "l4"], default="l2")
    parser.add_argument("--in_channels", type=int, default=2048)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    dataset = HandPoseDataset(
        discover_files(args.fpv_json, ".json", args.limit),
        discover_files(args.tpv_json, ".json", args.limit),
        discover_files(args.fpv_f, ".pt", args.limit),
        discover_files(args.tpv_f, ".pt", args.limit),
        args.label_file,
    )
    model = build_fusion_model(args.model_variant, in_channels=args.in_channels, num_class=args.num_classes)
    trainer = FusionClassificationTrainer(
        model=model,
        dataset=dataset,
        out_dir=args.out_dir,
        num_classes=args.num_classes,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lambda_align=args.lambda_align,
        threshold=args.threshold,
    )
    summary = trainer.fit()
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()

