import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset


DEFAULT_FOLD_INDICES = {
    0: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    1: [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56],
    2: [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57],
    3: [3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58],
    4: [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59],
}


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def binary_metrics(labels, preds):
    labels = labels.astype(int)
    preds = preds.astype(int)
    tp = int(((labels == 1) & (preds == 1)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    acc = float((labels == preds).mean()) if len(labels) else 0.0
    support = int((labels == 1).sum())
    return acc, precision, recall, f1, support


def multilabel_metrics(labels, preds):
    labels = labels.astype(int)
    preds = preds.astype(int)
    per_class = []
    for class_idx in range(labels.shape[1]):
        acc, precision, recall, f1, support = binary_metrics(labels[:, class_idx], preds[:, class_idx])
        per_class.append(
            {
                "class": class_idx,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    f1_values = np.array([item["f1"] for item in per_class], dtype=np.float32)
    supports = np.array([item["support"] for item in per_class], dtype=np.float32)
    weighted_f1 = float((f1_values * supports).sum() / supports.sum()) if supports.sum() > 0 else 0.0
    return {
        "accuracy_overall": float((labels == preds).mean()),
        "accuracy_exact_match": float((labels == preds).all(axis=1).mean()),
        "f1_macro": float(f1_values.mean()) if len(f1_values) else 0.0,
        "f1_weighted": weighted_f1,
        "per_class": per_class,
    }


def _folds_for_length(length):
    if length > 60:
        return DEFAULT_FOLD_INDICES
    folds = {i: [] for i in range(5)}
    for index in range(length):
        folds[index % 5].append(index)
    return folds


class FusionClassificationTrainer:
    def __init__(
        self,
        model,
        dataset,
        out_dir,
        num_classes,
        lr=1e-4,
        batch_size=2,
        num_epochs=20,
        lambda_align=1.0,
        threshold=0.5,
        device=None,
    ):
        self.model = model
        self.dataset = dataset
        self.out_dir = Path(out_dir)
        self.weights_dir = self.out_dir / "weights"
        self.metrics_csv = self.out_dir / "metrics.csv"
        self.summary_json = self.out_dir / "summary.json"
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lambda_align = lambda_align
        self.threshold = threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.criterion = nn.BCELoss()
        self.alignment_loss = nn.L1Loss()

    @staticmethod
    def _collate(batch):
        return {
            "sample_id": [item["sample_id"] for item in batch],
            "fpv_feature": torch.stack([item["fpv_feature"] for item in batch]),
            "fpv_pose": torch.stack([item["fpv_pose"] for item in batch]),
            "tpv_feature": torch.stack([item["tpv_feature"] for item in batch]),
            "tpv_pose": torch.stack([item["tpv_pose"] for item in batch]),
            "label": torch.stack([item["label"] for item in batch]),
        }

    def _write_header(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["fold", "epoch", "val_loss", "accuracy_overall", "accuracy_exact_match", "f1_macro", "f1_weighted"]
                + [f"acc_class_{i}" for i in range(self.num_classes)]
                + [f"precision_class_{i}" for i in range(self.num_classes)]
                + [f"recall_class_{i}" for i in range(self.num_classes)]
                + [f"f1_class_{i}" for i in range(self.num_classes)]
                + [f"support_class_{i}" for i in range(self.num_classes)]
            )

    def _append_row(self, fold, epoch, val_loss, metrics):
        per_class = metrics["per_class"]
        with self.metrics_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    fold,
                    epoch,
                    val_loss,
                    metrics["accuracy_overall"],
                    metrics["accuracy_exact_match"],
                    metrics["f1_macro"],
                    metrics["f1_weighted"],
                ]
                + [item["accuracy"] for item in per_class]
                + [item["precision"] for item in per_class]
                + [item["recall"] for item in per_class]
                + [item["f1"] for item in per_class]
                + [item["support"] for item in per_class]
            )

    def _evaluate(self, loader):
        self.model.eval()
        labels_all = []
        preds_all = []
        losses = []
        with torch.no_grad():
            for batch in loader:
                tpv_inputs = batch["tpv_feature"].to(self.device)
                tpv_poses = batch["tpv_pose"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs, _ = self.model(tpv_inputs, tpv_poses, fpv=False)
                losses.append(float(self.criterion(outputs, labels).item()))
                labels_all.append(labels.cpu().numpy())
                preds_all.append((outputs.cpu().numpy() > self.threshold).astype(int))
        labels_all = np.concatenate(labels_all, axis=0)
        preds_all = np.concatenate(preds_all, axis=0)
        return float(np.mean(losses)), multilabel_metrics(labels_all, preds_all), labels_all, preds_all

    def fit(self):
        self._write_header()
        self.model.to(self.device)
        initial_state = deepcopy(self.model.state_dict())
        folds = _folds_for_length(len(self.dataset))
        fold_summaries = []
        all_labels = []
        all_preds = []

        for fold in range(5):
            val_idx = [idx for idx in folds[fold] if idx < len(self.dataset)]
            train_idx = [idx for other in range(5) if other != fold for idx in folds[other] if idx < len(self.dataset)]
            if not val_idx:
                continue

            self.model.load_state_dict(initial_state)
            self.model.to(self.device)
            optimizer = Adam(self.model.parameters(), lr=self.lr)

            train_loader = DataLoader(
                Subset(self.dataset, train_idx),
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self._collate,
            )
            val_loader = DataLoader(
                Subset(self.dataset, val_idx),
                batch_size=1,
                shuffle=False,
                collate_fn=self._collate,
            )

            last_metrics = None
            last_labels = None
            last_preds = None
            for epoch in range(1, self.num_epochs + 1):
                self.model.train()
                for batch in train_loader:
                    fpv_inputs = batch["fpv_feature"].to(self.device)
                    fpv_poses = batch["fpv_pose"].to(self.device)
                    tpv_inputs = batch["tpv_feature"].to(self.device)
                    tpv_poses = batch["tpv_pose"].to(self.device)
                    labels = batch["label"].to(self.device)

                    optimizer.zero_grad()
                    fpv_outputs, fpv_feature = self.model(fpv_inputs, fpv_poses, fpv=True)
                    tpv_outputs, tpv_feature = self.model(tpv_inputs, tpv_poses, fpv=False)
                    class_loss = self.criterion(fpv_outputs, labels)
                    tpv_class_loss = self.criterion(tpv_outputs, labels)
                    align_loss = self.alignment_loss(fpv_feature, tpv_feature)
                    loss = class_loss + tpv_class_loss + self.lambda_align * align_loss
                    loss.backward()
                    optimizer.step()

                val_loss, last_metrics, last_labels, last_preds = self._evaluate(val_loader)
                self._append_row(fold + 1, epoch, val_loss, last_metrics)
                print(
                    f"fold={fold + 1} epoch={epoch} val_loss={val_loss:.4f} "
                    f"acc={last_metrics['accuracy_overall']:.4f} "
                    f"f1_macro={last_metrics['f1_macro']:.4f} "
                    f"f1_weighted={last_metrics['f1_weighted']:.4f}"
                )

            checkpoint = self.weights_dir / f"classifier_fold{fold + 1}.pth"
            torch.save(self.model.state_dict(), checkpoint)
            fold_summaries.append(last_metrics)
            all_labels.append(last_labels)
            all_preds.append(last_preds)

        labels = np.concatenate(all_labels, axis=0)
        preds = np.concatenate(all_preds, axis=0)
        summary = {
            "folds": fold_summaries,
            "overall": multilabel_metrics(labels, preds),
        }
        self.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

