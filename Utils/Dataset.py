import torch
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class HandPoseDataset(Dataset):
    def __init__(self, fpv_json, tpv_json, label_file, padding_value=0, mode='mix'):
        self.fpv_data = []
        self.tpv_data = []
        self.labels = []
        self.max_length = 0

        for file in fpv_json:
            print(file)
            with open(file, 'r') as f:
                frames = json.load(f)
                tensors = [torch.tensor([list(joint.values()) for joint in frame], dtype=torch.float32) for frame in frames]
                self.fpv_data.append(tensors)
                self.max_length = max(self.max_length, len(tensors))

            # find the corresponding label from csv file
            file_name = file.split('/')[-1].split('.')[0]

            with open(label_file, 'r') as f:
                for line in f.readlines():
                    if file_name in line:
                        label = line.split(',')[2:]
                        label = [float(l) for l in label]
                        label = torch.tensor(label)
                        self.labels.append(label)
        
        for file in tpv_json:
            print(file)
            with open(file, 'r') as f:
                frames = json.load(f)
                tensors = [torch.tensor([list(joint.values()) for joint in frame], dtype=torch.float32) for frame in frames]
                self.tpv_data.append(tensors)
                self.max_length = max(self.max_length, len(tensors))

        

        self.padding_value = padding_value

    def __len__(self):
        return len(self.fpv_data)

    def __getitem__(self, idx):
        fpv_sequence = self.fpv_data[idx]
        tpv_sequence = self.tpv_data[idx]

        print('fpv_sequence:', len(fpv_sequence))
        print('tpv_sequence:', len(tpv_sequence))
        if len(fpv_sequence) == 0:
            fpv_sequence = [torch.zeros_like(tpv_sequence[0]) for _ in tpv_sequence]
        if len(tpv_sequence) == 0:
            tpv_sequence = [torch.zeros_like(fpv_sequence[0]) for _ in fpv_sequence]

        print('fpv_sequence_shape:', fpv_sequence[0].shape)
        print('tpv_sequence_shape:', tpv_sequence[0].shape)
        
        label = self.labels[idx]

        fpv_sequence_tensor = pad_sequence(fpv_sequence, batch_first=True, padding_value=self.padding_value)
        tpv_sequence_tensor = pad_sequence(tpv_sequence, batch_first=True, padding_value=self.padding_value)
        
        # Pad to the maximum sequence length
        fpv_padded_tensor = torch.zeros(self.max_length, fpv_sequence_tensor.shape[1], fpv_sequence_tensor.shape[2])
        fpv_padded_tensor[:fpv_sequence_tensor.shape[0], :, :] = fpv_sequence_tensor
        fpv_padded_tensor = fpv_padded_tensor.permute(2, 0, 1) # from (T, V, C) to (C, T, V)
        fpv_padded_tensor = fpv_padded_tensor.unsqueeze(-1) # from (C, T, V) to (C, T, V, M)

        tpv_padded_tensor = torch.zeros(self.max_length, tpv_sequence_tensor.shape[1], tpv_sequence_tensor.shape[2])
        tpv_padded_tensor[:tpv_sequence_tensor.shape[0], :, :] = tpv_sequence_tensor
        tpv_padded_tensor = tpv_padded_tensor.permute(2, 0, 1)
        tpv_padded_tensor = tpv_padded_tensor.unsqueeze(-1)

        return {'fpv_data': fpv_padded_tensor, 'tpv_data': tpv_padded_tensor, 'label': label}


if __name__ == '__main__':
    dataset = HandPoseDataset(['output_data/1_0_0.json', 'output_data/1_0_1.json'], 'Accu.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        print(batch[0].shape)
        print(batch[1].shape)
        break
