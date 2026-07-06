import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cme_aqa.dataset import HandPoseDataset, discover_files
from cme_aqa.model import build_fusion_model
from cme_aqa.training.fusion_classification import multilabel_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run CME-AQA fusion classifier inference.")
    parser.add_argument("--fpv_json", required=True)
    parser.add_argument("--tpv_json", required=True)
    parser.add_argument("--fpv_f", required=True)
    parser.add_argument("--tpv_f", required=True)
    parser.add_argument("--label_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--limit", type=int, default=61)
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model_variant", choices=["l2", "l4"], default="l2")
    parser.add_argument("--in_channels", type=int, default=2048)
    return parser.parse_args()


def collate(batch):
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "tpv_feature": torch.stack([item["tpv_feature"] for item in batch]),
        "tpv_pose": torch.stack([item["tpv_pose"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HandPoseDataset(
        discover_files(args.fpv_json, ".json", args.limit),
        discover_files(args.tpv_json, ".json", args.limit),
        discover_files(args.fpv_f, ".pt", args.limit),
        discover_files(args.tpv_f, ".pt", args.limit),
        args.label_file,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    model = build_fusion_model(args.model_variant, in_channels=args.in_channels, num_class=args.num_classes).to(device)
    try:
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    rows = []
    labels_all = []
    preds_all = []
    with torch.no_grad():
        for batch in loader:
            outputs, _ = model(batch["tpv_feature"].to(device), batch["tpv_pose"].to(device), fpv=False)
            probs = outputs.cpu().numpy()[0]
            preds = (probs > args.threshold).astype(int)
            labels = batch["label"].numpy()[0].astype(int)
            labels_all.append(labels)
            preds_all.append(preds)
            rows.append([batch["sample_id"][0]] + probs.tolist() + preds.tolist() + labels.tolist())

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_id"]
            + [f"prob_class_{i}" for i in range(args.num_classes)]
            + [f"pred_class_{i}" for i in range(args.num_classes)]
            + [f"label_class_{i}" for i in range(args.num_classes)]
        )
        writer.writerows(rows)

    metrics = multilabel_metrics(np.asarray(labels_all), np.asarray(preds_all))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

