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

import os
from glob import glob
from natsort import natsorted

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import numpy as np
import torch

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

        # Loss function and optimizer
        self.criterion = nn.L1Loss()
        self.criterion2 = nn.MSELoss()
        self.reg_loss = nn.L1Loss()
        self.alignment_loss = nn.MSELoss()
        

        self.target_names = ['Insert_time', 'Withdraw_time', 'Frequent'] 
        
    def train(self):


        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # aggregate across folds
        folds_metrics = []
        # 1) SAVE initial model state
        init_state = deepcopy(self.model.state_dict())
        device = self.device
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):

            # 2) RESET model + optimizer + scaler (+ scheduler) for this fold
            self.model.load_state_dict(init_state)          # reset weights
            self.model.to(device)    
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)
            
            logging.info(f'Fold {fold + 1}')
            train_subset = Subset(self.dataset, train_idx)
            val_subset   = Subset(self.dataset, val_idx)

            # ----- optional: standardize targets per fold -----
            # fit scaler on training targets
            y_train = []
            for i in range(len(train_subset)):
                y_train.append(train_subset[i]['label_reg'])
            y_train = np.stack([np.asarray(y, dtype=np.float32) for y in y_train], axis=0)  # [N, n_targets]
            y_scaler = StandardScaler()
            y_scaler.fit(y_train)

            # small wrapper to apply scaling on-the-fly
            def collate_with_scaling(batch):
                xs = torch.stack([b['tpv_pose'] for b in batch], dim=0)
                ys = np.stack([np.asarray(b['label_reg'], dtype=np.float32) for b in batch], axis=0)
                ys = y_scaler.transform(ys)  # standardize
                ys = torch.from_numpy(ys).float()
                return {'tpv_pose': xs, 'label_reg': ys}

            # val collate (also scaled, but we’ll invert for reporting)
            def collate_val(batch):
                xs = torch.stack([b['tpv_pose'] for b in batch], dim=0)
                ys = np.stack([np.asarray(b['label_reg'], dtype=np.float32) for b in batch], axis=0)
                ys_scaled = y_scaler.transform(ys)
                return {
                    'tpv_pose': xs,
                    'label_reg': torch.from_numpy(ys_scaled).float(),
                    'label_reg_raw': torch.from_numpy(ys).float()
                }

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True,
                                    collate_fn=collate_with_scaling)
            val_loader   = DataLoader(val_subset, batch_size=1, shuffle=False,
                                    collate_fn=collate_val)

            best_val_mae = float('inf')
            best_state   = None

            for epoch in range(self.num_epochs):
                self.model.train()
                total_loss = 0.0

                for i, data in enumerate(train_loader):
                    inputs     = data['tpv_pose'].to(self.device)           # [B, ...]
                    reg_labels = data['label_reg'].to(self.device)          # [B, n_targets] (standardized)

                    if epoch == 0 and i == 0:
                        print('inputs:', inputs.shape)
                        print('reg_labels:', reg_labels.shape)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)                            # [B, n_targets], no activation
                    loss    = self.reg_loss(outputs, reg_labels)
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

                train_loss = total_loss / max(1, len(train_loader))
                logging.info(f'Fold {fold+1} | Epoch [{epoch+1}/{self.num_epochs}] TrainLoss: {train_loss:.4f}')
                print(f'Fold {fold+1} | Epoch [{epoch+1}/{self.num_epochs}] TrainLoss: {train_loss:.4f}')

                # ----- Validation -----
                self.model.eval()
                with torch.no_grad():
                    preds_raw = []
                    gts_raw   = []
                    val_losses = []
                    for data in val_loader:
                        x = data['tpv_pose'].to(self.device)
                        y_scaled = data['label_reg'].to(self.device)        # standardized
                        y_raw    = data['label_reg_raw'].cpu().numpy()      # original scale for reporting

                        yhat_scaled = self.model(x)                         # standardized space
                        val_loss    = self.reg_loss(yhat_scaled, y_scaled).item()
                        val_losses.append(val_loss)

                        # invert scaling for metrics
                        yhat_scaled_np = yhat_scaled.cpu().numpy()
                        yhat_raw_np    = y_scaler.inverse_transform(yhat_scaled_np)

                        preds_raw.append(yhat_raw_np[0])
                        gts_raw.append(y_raw[0])

                    preds_raw = np.stack(preds_raw, axis=0)  # [Nv, n_targets]
                    gts_raw   = np.stack(gts_raw,   axis=0)  # [Nv, n_targets]

                    # Metrics on original scale
                    mae = np.mean(np.abs(preds_raw - gts_raw), axis=0)         # per-target
                    mse = np.mean((preds_raw - gts_raw)**2, axis=0)
                    rmse = np.sqrt(mse)
                    # R^2 (per target)
                    ss_res = np.sum((preds_raw - gts_raw)**2, axis=0)
                    ss_tot = np.sum((gts_raw - gts_raw.mean(axis=0))**2, axis=0) + 1e-12
                    r2 = 1.0 - ss_res / ss_tot

                    # Spearman (per target) – safe compute
                    spearman = []
                    for t in range(preds_raw.shape[1]):
                        try:
                            sp = spearmanr(gts_raw[:, t], preds_raw[:, t], nan_policy='omit').correlation
                        except Exception:
                            sp = np.nan
                        spearman.append(sp)
                    spearman = np.array(spearman, dtype=np.float32)

                    val_mae_mean = float(np.mean(mae))
                    val_loss_mean = float(np.mean(val_losses))

                    # ---- pretty, per-target report ----
                    per_target_str = " | ".join(
                        [f"{self.target_names[t]} MAE={mae[t]:.4f}" for t in range(len(self.target_names))]
                    )
                    print(f'Fold {fold+1} | Epoch {epoch+1} | ValLoss(scaled): {val_loss_mean:.4f} | '
                        f'MAE(mean)={val_mae_mean:.4f} | {per_target_str}')
                    logging.info(f'Fold {fold+1} | Epoch {epoch+1} | ValLoss(scaled): {val_loss_mean:.4f} | '
                                f'MAE(mean)={val_mae_mean:.4f} | per-target MAE: '
                                + ", ".join([f"{self.target_names[t]}={mae[t]:.4f}" for t in range(len(self.target_names))]))

                    # Keep best by mean MAE but store full per-target vectors too
                    if val_mae_mean < best_val_mae:
                        best_val_mae = val_mae_mean
                        best_state = {
                            'model': {k: v.cpu() for k, v in self.model.state_dict().items()},
                            'mae': mae, 'rmse': rmse, 'r2': r2, 'spearman': spearman,
                            'val_mae_mean': val_mae_mean
                        }

            if best_state is not None:
                self.model.load_state_dict(best_state['model'])
                fold_report = {
                    'val_mae_mean': float(best_state['val_mae_mean']),
                    'mae': best_state['mae'],
                    'rmse': best_state['rmse'],
                    'r2': best_state['r2'],
                    'spearman': best_state['spearman']
                }
            else:
                # fallback (shouldn’t happen)
                fold_report = {'val_mae_mean': float('inf')}

            folds_metrics.append(fold_report)
            print(f'=== Fold {fold+1} best MAE: {fold_report["val_mae_mean"]:.4f} ===')
            logging.info(f'=== Fold {fold+1} best MAE: {fold_report["val_mae_mean"]:.4f} ===')

            # Save model
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
    model = model_class(in_channels=3, num_class=num_classes, graph_args=graph_args, edge_importance_weighting=True)
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
