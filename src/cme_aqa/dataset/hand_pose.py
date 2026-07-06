import csv
import json
import os
import re
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def natural_key(path):
    name = str(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def discover_files(folder, suffix, limit=None):
    files = sorted(Path(folder).glob(f"*{suffix}"), key=natural_key)
    if limit is not None:
        files = files[:limit]
    return [str(p) for p in files]


def _stem(path_or_name):
    return os.path.splitext(os.path.basename(path_or_name))[0]


def _to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _read_pose_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        frames = json.load(handle)
    tensors = []
    for frame in frames:
        joints = [list(joint.values()) for joint in frame]
        tensors.append(torch.tensor(joints, dtype=torch.float32))
    return tensors


def _read_feature_pt(path):
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        if data.ndim != 2:
            raise ValueError(f"Expected visual feature tensor [T, C] in {path}, got {tuple(data.shape)}")
        return [row.float() for row in data]
    return [torch.as_tensor(row, dtype=torch.float32) for row in data]


class HandPoseDataset(Dataset):
    """Dual-view pose/visual feature dataset used by the CME-AQA fusion trainer."""

    def __init__(self, fpv_json, tpv_json, fpv_features, tpv_features, label_file, padding_value=0.0):
        if not (len(fpv_json) == len(tpv_json) == len(fpv_features) == len(tpv_features)):
            raise ValueError(
                "Input lengths differ: "
                f"fpv_json={len(fpv_json)}, tpv_json={len(tpv_json)}, "
                f"fpv_features={len(fpv_features)}, tpv_features={len(tpv_features)}"
            )

        self.padding_value = padding_value
        self.fpv_pose = []
        self.tpv_pose = []
        self.fpv_features = []
        self.tpv_features = []
        self.labels_cls = []
        self.labels_reg = []
        self.sample_ids = []
        self.max_length = 0

        cls_by_stem, reg_by_stem = self._load_labels(label_file)

        for fpv_p, tpv_p, fpv_f, tpv_f in zip(fpv_json, tpv_json, fpv_features, tpv_features):
            stem = _stem(fpv_p)
            if stem not in reg_by_stem:
                raise KeyError(f"No label row for sample stem '{stem}' from {fpv_p}")

            fpv_pose_seq = _read_pose_json(fpv_p)
            tpv_pose_seq = _read_pose_json(tpv_p)
            fpv_feat_seq = _read_feature_pt(fpv_f)
            tpv_feat_seq = _read_feature_pt(tpv_f)

            self.fpv_pose.append(fpv_pose_seq)
            self.tpv_pose.append(tpv_pose_seq)
            self.fpv_features.append(fpv_feat_seq)
            self.tpv_features.append(tpv_feat_seq)
            self.labels_cls.append(cls_by_stem[stem])
            self.labels_reg.append(reg_by_stem[stem])
            self.sample_ids.append(stem)
            self.max_length = max(
                self.max_length,
                len(fpv_pose_seq),
                len(tpv_pose_seq),
                len(fpv_feat_seq),
                len(tpv_feat_seq),
            )

    @staticmethod
    def _load_labels(label_file):
        cls_by_stem = {}
        reg_by_stem = {}
        with open(label_file, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            columns = {name: i for i, name in enumerate(header)}
            for required in ("Video_Head", "Insert_time", "Withdraw_time"):
                if required not in columns:
                    raise KeyError(f"CSV missing required column: {required}")

            head_i = columns["Video_Head"]
            insert_i = columns["Insert_time"]
            frequent_i = columns.get("frequent", columns.get("Frequent"))
            if frequent_i is None:
                raise KeyError("CSV missing required column: frequent/Frequent")

            has_freq_times = "Frequent_Times" in columns
            for row in reader:
                if not row:
                    continue
                stem = _stem(row[head_i])
                cls_end = min(insert_i, frequent_i)
                cls_values = [_to_float(x) for x in row[head_i + 1 : cls_end]]
                insert_time = _to_float(row[insert_i])
                withdraw_time = _to_float(row[columns["Withdraw_time"]])

                if has_freq_times:
                    raw = (row[columns["Frequent_Times"]] or "").strip().strip('"')
                    times = [_to_float(t) for t in raw.split(",") if t.strip()]
                    if len(times) > 2:
                        duration = times[-2] - times[1]
                        frequent = ((len(times) - 2) / duration) * 60 if duration > 0 else 0.0
                    else:
                        frequent = 0.0
                else:
                    frequent = _to_float(row[frequent_i])

                cls_by_stem[stem] = torch.tensor(cls_values, dtype=torch.float32)
                reg_by_stem[stem] = torch.tensor([insert_time, withdraw_time, frequent], dtype=torch.float32)
        return cls_by_stem, reg_by_stem

    def __len__(self):
        return len(self.fpv_pose)

    def _pad_pose(self, sequence):
        if not sequence:
            raise ValueError("Pose sequence is empty and cannot be padded.")
        tensor = pad_sequence(sequence, batch_first=True, padding_value=self.padding_value)
        padded = torch.zeros(self.max_length, tensor.shape[1], tensor.shape[2])
        padded[: tensor.shape[0]] = tensor
        return padded.permute(2, 0, 1).unsqueeze(-1)

    def _pad_features(self, sequence):
        if not sequence:
            raise ValueError("Visual feature sequence is empty and cannot be padded.")
        tensor = pad_sequence(sequence, batch_first=True, padding_value=self.padding_value)
        padded = torch.zeros(self.max_length, tensor.shape[1])
        padded[: tensor.shape[0]] = tensor
        return padded.permute(1, 0)

    def __getitem__(self, index):
        return {
            "sample_id": self.sample_ids[index],
            "fpv_pose": self._pad_pose(self.fpv_pose[index]),
            "tpv_pose": self._pad_pose(self.tpv_pose[index]),
            "fpv_feature": self._pad_features(self.fpv_features[index]),
            "tpv_feature": self._pad_features(self.tpv_features[index]),
            "label": self.labels_cls[index],
            "label_reg": self.labels_reg[index],
        }
