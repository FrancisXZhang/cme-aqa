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
        self.aligmnet_loss =  nn.L1Loss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def train(self):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        spearman_corr_total = 0
        relative_l2_distance_total = 0
        test_acc = torch.zeros(1, self.num_classes).to(self.device)
        test_f1 = torch.zeros(1, self.num_classes).to(self.device)
        test_precision = torch.zeros(1, self.num_classes).to(self.device)
        test_recall = torch.zeros(1, self.num_classes).to(self.device)

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
            logging.info(f'Fold {fold + 1}')
            print(f'Fold {fold + 1}')
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

            for epoch in range(self.num_epochs):
                self.model.train()
                total_loss = 0
                total_class_loss = 0
                total_tpv_class_loss = 0
                total_aligmnet_loss = 0
                acc_error = torch.zeros(1, self.num_classes).to(self.device)

                for i, data in enumerate(train_loader):
                    fpv_inputs = data['fpv_feature'].to(self.device)
                    fpv_poses = data['fpv_pose'].to(self.device)
                    tpv_inputs = data['tpv_feature'].to(self.device)
                    tpv_poses = data['tpv_pose'].to(self.device)

                    labels = data['label'].to(self.device)
                    
                    if epoch == 0 and i == 0:
                        print('fpv_inputs:', fpv_inputs.shape)
                        print('fpv_poses:', fpv_poses.shape)
                        print('tpv_inputs:', tpv_inputs.shape)
                        print('tpv_poses:', tpv_poses.shape)
                        print('labels:', labels.shape)

                    # Reset gradients
                    self.optimizer.zero_grad()
                    # Forward pass
                    outputs, feature = self.model(fpv_inputs, fpv_poses, fpv = True)

                    tpv_outputs, tpv_feature = self.model(tpv_inputs, tpv_poses, fpv = False)
                    
                    print('outputs:', outputs)  
                    print('labels:', labels)
                    class_loss = self.criterion(outputs, labels) + self.criterion2(outputs, labels)
                    tpv_class_loss = self.criterion(tpv_outputs, labels) + self.criterion2(tpv_outputs, labels)
                    aligmnet_loss = self.aligmnet_loss(feature, tpv_feature)
                    loss = class_loss + aligmnet_loss + tpv_class_loss
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()

                    # Calculate accuracy error
                    output_index = outputs > 0.5

                    # accuarcy
                    acc_error += (output_index == labels).sum(dim=0)

                    total_class_loss += class_loss.item()
                    total_tpv_class_loss += tpv_class_loss.item()
                    total_aligmnet_loss += aligmnet_loss.item()
                    total_loss += loss.item()

                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {total_loss / len(train_loader)}')
                logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {total_loss / len(train_loader)}')
                logging.info(f'Class Loss: {total_class_loss / len(train_loader)}')
                logging.info(f'Aligmnet Loss: {total_aligmnet_loss / len(train_loader)}')
                acc_error = acc_error / len(train_loader) / self.batch_size
                for i in range(self.num_classes):
                    logging.info(f'Class {i} Accuracy: {acc_error[0][i]}')
                    print(f'Class {i} Accuracy: {acc_error[0][i]}')

            # Validation phase
            self.model.eval()
            val_loss = 0
            all_labels = []
            all_outputs = []
            all_outputs_index = []

            acc_error = torch.zeros(1, self.num_classes).to(self.device)

            spearman_corr_total = 0
            relative_l2_distance_total = 0

            with torch.no_grad():
                for data in val_loader:
                    
                    fpv_inputs = data['fpv_feature'].to(self.device)
                    fpv_poses = data['fpv_pose'].to(self.device)
                    tpv_inputs = data['tpv_feature'].to(self.device)
                    tpv_poses = data['tpv_pose'].to(self.device)

                    labels = data['label'].to(self.device)
                    
                    outputs, feature = self.model(fpv_inputs, fpv_poses, fpv = True)
                    tpv_outputs, tpv_feature = self.model(tpv_inputs, tpv_poses, fpv = False)

                    loss = self.criterion(outputs, labels) + self.criterion2(outputs, labels)
                    val_loss += loss.item()

                    output_index = outputs > 0.5

                    all_labels.append(labels.cpu().numpy())
                    all_outputs.append(outputs.cpu().numpy())
                    all_outputs_index.append(output_index.cpu().numpy())


                    # accuarcy
                    acc_error += (output_index == labels).sum(dim=0)
                    
            acc_error = acc_error / len(val_loader)
            test_acc += acc_error                
            
            all_labels = np.concatenate(all_labels)
            all_outputs = np.concatenate(all_outputs)
            all_outputs_index = np.concatenate(all_outputs_index)
            logging.info(f'All Labels: {all_labels.shape}')
            logging.info(f'All Outputs: {all_outputs.shape}')
            logging.info(f'All Outputs Index: {all_outputs_index.shape}')

            for i in range(self.num_classes):
                logging.info(f'Class {i} Accuracy: {acc_error[0][i]}')

                tmp_f1 = f1_score(all_labels[:,i], all_outputs_index[:,i], average='macro')
                logging.info(f'Class {i} F1 Score, {tmp_f1}')
                test_f1[0][i] += tmp_f1

                tmp_precision = precision_score(all_labels[:,i], all_outputs_index[:,i], average='macro')
                logging.info(f'Class {i} Precision, {tmp_precision}')
                test_precision[0][i] += tmp_precision

                tmp_recall = recall_score(all_labels[:,i], all_outputs_index[:,i], average='macro')
                logging.info(f'Class {i} Recall, {tmp_recall}')
                test_recall[0][i] += tmp_recall

            # Calculate Spearman correlation after summing the labels and outputs
            spearman_corr, _ = spearmanr(all_labels.sum(axis=1).flatten(), all_outputs.sum(axis=1).flatten())
            spearman_corr_total += spearman_corr
            logging.info(f'Spearman Correlation: {spearman_corr}')

            # Calculate relative L2 distance
            relative_l2 = relative_l2_distance(all_labels.sum(axis=1).flatten(), all_outputs.sum(axis=1).flatten()
            , ymax = self.num_classes, ymin = 0)
            relative_l2_distance_total += relative_l2
            logging.info(f'Relative L2 Distance: {relative_l2}')

  
        for i in range(self.num_classes):
            test_acc[0][i] = test_acc[0][i] / 5
            test_f1[0][i] = test_f1[0][i] / 5
            test_precision[0][i] = test_precision[0][i] / 5
            test_recall[0][i] = test_recall[0][i] / 5

            logging.info(f'Average Class {i} Accuracy: {test_acc[0][i]}')
            logging.info(f'Average Class {i} F1 Score, {test_f1[0][i]}')
            logging.info(f'Average Class {i} Precision, {test_precision[0][i]}')
            logging.info(f'Average Class {i} Recall, {test_recall[0][i]}')

        logging.info(f'Average Spearman Correlation: {spearman_corr_total / 5}')
        logging.info(f'Average Relative L2 Distance: {relative_l2_distance_total / 5}')

        print("Training complete")
        # Save model
        model_path = 'hand_pose_model.pth'
        torch.save(self.model.state_dict(), model_path)


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
    parser.add_argument('--num_classes', type=int, default=9, help='Number of classes for classification task.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model.')
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
