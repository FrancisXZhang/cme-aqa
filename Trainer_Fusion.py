import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
import argparse
import importlib
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import numpy as np

from Utils.DatasetPV import HandPoseDataset
from Loss.relative_l2_distance import relative_l2_distance
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

import os
from glob import glob
from natsort import natsorted
import random
from copy import deepcopy

import logging

class HandPoseTrainer:
    def __init__(self, model, fpv_json_files, tpv_json_files, fpv_visual_features, tpv_visual_features,
    label_file, num_classes, graph_args, lr=0.001, batch_size=1, num_epochs=10):
        """
        Initialize the HandPoseTrainer class.

        Args:
            model (torch.nn.Module): The model to train.
            json_files (list of str): List of paths to JSON files containing the dataset.
            num_classes (int): Number of classes for the classification task.
            graph_args (dict): Arguments for building the graph.
            lr (float): Learning rate for the optimizer.
            batch_size (int): Batch size for training.
            num_epochs (int): Number of epochs to train the model.
        """
        self.model = model
        self.fpv_json_files = fpv_json_files
        self.tpv_json_files = tpv_json_files
        self.fpv_visual_features = fpv_visual_features
        self.tpv_visual_features = tpv_visual_features
        self.label_file = label_file
        self.num_classes = num_classes
        self.graph_args = graph_args
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Initialize dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = HandPoseDataset(fpv_json_files, tpv_json_files,
                                       fpv_visual_features, tpv_visual_features,
                                       label_file)

        self.reg_loss = nn.L1Loss()        # MAE
        self.criterion2 = nn.MSELoss()     # MSE (optional extra)
        self.alignment_loss = nn.MSELoss() # or cosine embedding loss etc.
        self.target_names = ['Insert_time','Withdraw_time','Frequent']

        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def train(self):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        spearman_corr_total = 0
        relative_l2_distance_total = 0
        test_acc = torch.zeros(1, self.num_classes).to(self.device)
        test_f1 = torch.zeros(1, self.num_classes).to(self.device)
        test_precision = torch.zeros(1, self.num_classes).to(self.device)
        test_recall = torch.zeros(1, self.num_classes).to(self.device)
        folds_metrics = []
        init_state = deepcopy(self.model.state_dict())
        device = self.device
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
            logging.info(f'Fold {fold + 1}')
            print(f'Fold {fold + 1}')
            logging.info(f'Train IDX: {train_idx}')
            logging.info(f'Val IDX: {val_idx}')
            self.model.load_state_dict(init_state)          # reset weights
            self.model.to(device)    
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)
            # ----- subsets -----
            train_subset = Subset(self.dataset, train_idx)
            val_subset   = Subset(self.dataset, val_idx)

            # ----- fit scaler on training targets (regression) -----
            y_train = []
            for i in range(len(train_subset)):
                y_train.append(train_subset[i]['label_reg'])
            y_train = np.stack([np.asarray(y, dtype=np.float32) for y in y_train], axis=0)  # [N, n_targets]
            y_scaler = StandardScaler().fit(y_train)

            # collates to keep everything else raw, but scale y on the fly
            def collate_train(batch):
                out = {}
                # features/poses
                out['fpv_feature'] = torch.stack([b['fpv_feature'] for b in batch], dim=0)
                out['fpv_pose']    = torch.stack([b['fpv_pose'] for b in batch], dim=0)
                out['tpv_feature'] = torch.stack([b['tpv_feature'] for b in batch], dim=0)
                out['tpv_pose']    = torch.stack([b['tpv_pose'] for b in batch], dim=0)
                # scale targets
                ys = np.stack([np.asarray(b['label_reg'], dtype=np.float32) for b in batch], axis=0)
                ys = y_scaler.transform(ys)
                out['label_reg'] = torch.from_numpy(ys).float()
                return out

            def collate_val(batch):
                out = {}
                out['fpv_feature'] = torch.stack([b['fpv_feature'] for b in batch], dim=0)
                out['fpv_pose']    = torch.stack([b['fpv_pose'] for b in batch], dim=0)
                out['tpv_feature'] = torch.stack([b['tpv_feature'] for b in batch], dim=0)
                out['tpv_pose']    = torch.stack([b['tpv_pose'] for b in batch], dim=0)
                ys = np.stack([np.asarray(b['label_reg'], dtype=np.float32) for b in batch], axis=0)
                ys_scaled = y_scaler.transform(ys)
                out['label_reg']      = torch.from_numpy(ys_scaled).float()
                out['label_reg_raw']  = torch.from_numpy(ys).float()   # for reporting
                return out

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True,
                                    collate_fn=collate_train)
            val_loader   = DataLoader(val_subset,   batch_size=1, shuffle=False,
                                    collate_fn=collate_val)

            # choose alignment weight (feel free to expose as arg)
            lambda_align = getattr(self, 'lambda_align', 0.1)

            last_state = None  # we’ll keep the last epoch metrics/weights only

            for epoch in range(self.num_epochs):
                self.model.train()
                total_loss = 0.0
                total_reg_fpv_loss = 0.0
                total_reg_tpv_loss = 0.0
                total_align_loss   = 0.0

                for i, data in enumerate(train_loader):
                    fpv_inputs = data['fpv_feature'].to(self.device)
                    fpv_poses  = data['fpv_pose'].to(self.device)
                    tpv_inputs = data['tpv_feature'].to(self.device)
                    tpv_poses  = data['tpv_pose'].to(self.device)
                    reg_labels = data['label_reg'].to(self.device)        # standardized targets

                    if epoch == 0 and i == 0:
                        print('fpv_inputs:', fpv_inputs.shape)
                        print('fpv_poses:', fpv_poses.shape)
                        print('tpv_inputs:', tpv_inputs.shape)
                        print('tpv_poses:', tpv_poses.shape)
                        print('reg_labels:', reg_labels.shape)

                    self.optimizer.zero_grad()

                    # forward (dual stream)
                    outputs_fpv, feat_fpv = self.model(fpv_inputs, fpv_poses, fpv=True)   # -> [B, n_targets]
                    outputs_tpv, feat_tpv = self.model(tpv_inputs, tpv_poses, fpv=False)  # -> [B, n_targets]

                    # regression losses in standardized space
                    reg_fpv_loss = self.reg_loss(outputs_fpv, reg_labels) + self.criterion2(outputs_fpv, reg_labels)
                    reg_tpv_loss = self.reg_loss(outputs_tpv, reg_labels) + self.criterion2(outputs_tpv, reg_labels)

                    # feature alignment (same batch pairing)
                    align_loss = self.alignment_loss(feat_fpv, feat_tpv)

                    loss = 0.1 * reg_fpv_loss + reg_tpv_loss + lambda_align * align_loss
                    loss.backward()
                    self.optimizer.step()

                    total_loss          += float(loss.item())
                    total_reg_fpv_loss  += float(reg_fpv_loss.item())
                    total_reg_tpv_loss  += float(reg_tpv_loss.item())
                    total_align_loss    += float(align_loss.item())

                n_batches = max(1, len(train_loader))
                print(f'Epoch [{epoch + 1}/{self.num_epochs}] '
                    f'Loss={total_loss/n_batches:.4f} | FPV={total_reg_fpv_loss/n_batches:.4f} '
                    f'| TPV={total_reg_tpv_loss/n_batches:.4f} | Align={total_align_loss/n_batches:.4f}')
                logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}] '
                            f'Loss={total_loss/n_batches:.4f} | FPV={total_reg_fpv_loss/n_batches:.4f} '
                            f'| TPV={total_reg_tpv_loss/n_batches:.4f} | Align={total_align_loss/n_batches:.4f}')

                # ---------- Validation (compute metrics on ORIGINAL scale) ----------
                self.model.eval()
                with torch.no_grad():
                    preds_raw = []
                    gts_raw   = []
                    val_losses = []

                    for data in val_loader:
                        fpv_inputs = data['fpv_feature'].to(self.device)
                        fpv_poses  = data['fpv_pose'].to(self.device)
                        tpv_inputs = data['tpv_feature'].to(self.device)
                        tpv_poses  = data['tpv_pose'].to(self.device)

                        y_scaled = data['label_reg'].to(self.device)         # standardized
                        y_raw    = data['label_reg_raw'].cpu().numpy()       # original

                        yhat_scaled, feat_f = self.model(tpv_inputs, tpv_poses, fpv=False)
                        # loss reported in standardized space
                        vloss = self.reg_loss(yhat_scaled, y_scaled).item()
                        val_losses.append(vloss)

                        # invert scaling for reporting
                        yhat_scaled_np = yhat_scaled.cpu().numpy()
                        yhat_raw_np    = y_scaler.inverse_transform(yhat_scaled_np)

                        preds_raw.append(yhat_raw_np[0])
                        gts_raw.append(y_raw[0])

                    preds_raw = np.stack(preds_raw, axis=0)  # [Nv, n_targets]
                    gts_raw   = np.stack(gts_raw,   axis=0)  # [Nv, n_targets]

                    mae  = np.mean(np.abs(preds_raw - gts_raw), axis=0)
                    mse  = np.mean((preds_raw - gts_raw) ** 2, axis=0)
                    rmse = np.sqrt(mse)

                    ss_res = np.sum((preds_raw - gts_raw)**2, axis=0)
                    ss_tot = np.sum((gts_raw - gts_raw.mean(axis=0))**2, axis=0) + 1e-12
                    r2 = 1.0 - ss_res / ss_tot

                    # Spearman per target
                    spearman = []
                    for t in range(preds_raw.shape[1]):
                        try:
                            sp = spearmanr(gts_raw[:, t], preds_raw[:, t], nan_policy='omit').correlation
                        except Exception:
                            sp = np.nan
                        spearman.append(sp)
                    spearman = np.array(spearman, dtype=np.float32)

                    val_mae_mean   = float(np.mean(mae))
                    val_loss_mean  = float(np.mean(val_losses))
                    per_target_str = " | ".join([f"{self.target_names[t]} MAE={mae[t]:.4f}"
                                                for t in range(len(self.target_names))])
                    print(f'Fold {fold+1} | Epoch {epoch+1} | ValLoss(scaled): {val_loss_mean:.4f} | '
                        f'MAE(mean)={val_mae_mean:.4f} | {per_target_str}')
                    logging.info(f'Fold {fold+1} | Epoch {epoch+1} | ValLoss(scaled): {val_loss_mean:.4f} | '
                                f'MAE(mean)={val_mae_mean:.4f} | per-target MAE: '
                                + ", ".join([f"{self.target_names[t]}={mae[t]:.4f}"
                                            for t in range(len(self.target_names))]))

                    # -------- keep LAST epoch only --------
                    last_state = {
                        'model': {k: v.cpu() for k, v in self.model.state_dict().items()},
                        'mae': mae, 'rmse': rmse, 'r2': r2, 'spearman': spearman,
                        'val_mae_mean': val_mae_mean
                    }

            # end epochs

            # ---- choose LAST ----
            if last_state is not None:
                self.model.load_state_dict(last_state['model'])
                fold_report = {
                    'val_mae_mean': float(last_state['val_mae_mean']),
                    'mae': last_state['mae'],
                    'rmse': last_state['rmse'],
                    'r2': last_state['r2'],
                    'spearman': last_state['spearman'],
                    'picked': 'last'
                }
            else:
                fold_report = {'val_mae_mean': float('inf'), 'picked': 'last'}

            folds_metrics.append(fold_report)
            print(f'=== Fold {fold+1} [LAST] MAE: {fold_report["val_mae_mean"]:.4f} ===')
            logging.info(f'=== Fold {fold+1} [LAST] MAE: {fold_report["val_mae_mean"]:.4f} ===')

            # extract model name from args.model_script or args.model_class
            model_name = os.path.basename(args.model_script).replace(".py","")
            # or better: use the class name (cleaner)
            model_name = args.model_script

            # extract dataset tag from fpv_json folder path
            fpv_tag = os.path.basename(os.path.normpath(args.fpv_json))

            # then when saving per fold:
            model_path = os.path.join('Weights', f'{model_name}_{fpv_tag}_fold{fold+1}.pth')
            torch.save(self.model.state_dict(), model_path)

        
        # ----- Aggregate across folds -----
        # Average per-target metrics if available
        has_targets = all('mae' in m for m in folds_metrics)
        if has_targets:
            mae_all = np.stack([m['mae'] for m in folds_metrics], axis=0)
            rmse_all = np.stack([m['rmse'] for m in folds_metrics], axis=0)
            r2_all = np.stack([m['r2'] for m in folds_metrics], axis=0)
            sp_all = np.stack([m['spearman'] for m in folds_metrics], axis=0)

            print('CV | MAE per-target:', np.mean(mae_all, axis=0))
            print('CV | RMSE per-target:', np.mean(rmse_all, axis=0))
            print('CV | R2 per-target:', np.mean(r2_all, axis=0))
            print('CV | Spearman per-target:', np.nanmean(sp_all, axis=0))
            print('CV | MAE mean:', float(np.mean([m["val_mae_mean"] for m in folds_metrics])))
            logging.info('CV | MAE per-target: ' + ", ".join([f"{self.target_names[t]}={np.mean(mae_all, axis=0)[t]:.4f}" for t in range(len(self.target_names))]))
            logging.info('CV | RMSE per-target: ' + ", ".join([f"{self.target_names[t]}={np.mean(rmse_all, axis=0)[t]:.4f}" for t in range(len(self.target_names))]))
            logging.info('CV | R2 per-target: ' + ", ".join([f"{self.target_names[t]}={np.mean(r2_all, axis=0)[t]:.4f}" for t in range(len(self.target_names))]))
            logging.info('CV | Spearman per-target: ' + ", ".join([f"{self.target_names[t]}={np.nanmean(sp_all, axis=0)[t]:.4f}" for t in range(len(self.target_names))]))

        print("Training complete")

def main(args):
    fpv_json_files = natsorted(glob(os.path.join(args.fpv_json, '*.json')))[:61]
    tpv_json_files = natsorted(glob(os.path.join(args.tpv_json, '*.json')))[:61]
    fpv_visual_features = natsorted(glob(os.path.join(args.fpv_f, '*.pt')))[:61]
    tpv_visual_features = natsorted(glob(os.path.join(args.tpv_f, '*.pt')))[:61]

    label_file = args.label_file
    num_classes = args.num_classes
    graph_args = {'layout': 'mediapipe', 'strategy': 'spatial'}
    logging.info(f'Training hand pose model with {len(fpv_json_files)} fpv samples')
    logging.info(f'Training hand pose model with {len(tpv_json_files)} tpv samples')

    # Dynamically import the model module and class
    model_module = importlib.import_module(args.model_script)
    model_class = getattr(model_module, args.model_class)

    # Initialize the model
    model = model_class(in_channels=2048, num_class=num_classes)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Initialize the trainer and start training
    trainer = HandPoseTrainer(model, fpv_json_files, tpv_json_files,
                            fpv_visual_features, tpv_visual_features,
                            label_file, num_classes, graph_args,
                            lr=args.lr, batch_size=args.batch_size, num_epochs=args.num_epochs)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a hand pose estimation model.")
    parser.add_argument('--fpv_json', type=str, required=True, help='Folder containing JSON files with dataset.')
    parser.add_argument('--tpv_json', type=str, required=True, help='Folder containing JSON files with dataset.')
   
    parser.add_argument('--fpv_f', type=str, required=True, help='Folder containing visual features.')
    parser.add_argument('--tpv_f', type=str, required=True, help='Folder containing visual features.')
    
    parser.add_argument('--label_file', type=str, required=True, help='CSV file containing labels.')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes for classification task.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train the model.')
    parser.add_argument('--model_script', type=str, required=True, help='Script containing the model definition.')
    parser.add_argument('--model_class', type=str, required=True, help='Name of the model class to instantiate.')
    parser.add_argument('--log_file', type=str, default='Training.log', help='Log file name.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()

    # set random seed
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    np.random.seed(seed)


    # Configure logging
    logging.basicConfig(
        filename=args.log_file,
        filemode='w',
        format='%(name)s - %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    main(args)
