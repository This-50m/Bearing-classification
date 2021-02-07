import os
import scipy.io as scio
import librosa
from librosa import display as LD
from matplotlib import pyplot as plt
import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset

DATA_DISTRIBUTION = {
    105: {'fault_diameter': 0.1778, 'fault_type': 'inner_ring', 'data_id': '105'},
    106: {'fault_diameter': 0.1778, 'fault_type': 'inner_ring', 'data_id': '106'},
    107: {'fault_diameter': 0.1778, 'fault_type': 'inner_ring', 'data_id': '107'},
    108: {'fault_diameter': 0.1778, 'fault_type': 'inner_ring', 'data_id': '108'},
    118: {'fault_diameter': 0.1778, 'fault_type': 'rolling_element', 'data_id': '118'},
    119: {'fault_diameter': 0.1778, 'fault_type': 'rolling_element', 'data_id': '119'},
    120: {'fault_diameter': 0.1778, 'fault_type': 'rolling_element', 'data_id': '120'},
    121: {'fault_diameter': 0.1778, 'fault_type': 'rolling_element', 'data_id': '121'},
    130: {'fault_diameter': 0.1778, 'fault_type': 'outer_ring', 'data_id': '130', 'position': 'center'},
    131: {'fault_diameter': 0.1778, 'fault_type': 'outer_ring', 'data_id': '131', 'position': 'center'},
    132: {'fault_diameter': 0.1778, 'fault_type': 'outer_ring', 'data_id': '132', 'position': 'center'},
    133: {'fault_diameter': 0.1778, 'fault_type': 'outer_ring', 'data_id': '133', 'position': 'center'},
    144: {'fault_diameter': 0.1778, 'fault_type': 'outer_ring', 'data_id': '144', 'position': 'orthogonal'},
    145: {'fault_diameter': 0.1778, 'fault_type': 'outer_ring', 'data_id': '145', 'position': 'orthogonal'},
    146: {'fault_diameter': 0.1778, 'fault_type': 'outer_ring', 'data_id': '146', 'position': 'orthogonal'},
    147: {'fault_diameter': 0.1778, 'fault_type': 'outer_ring', 'data_id': '147', 'position': 'orthogonal'},
    156: {'fault_diameter': 0.1778, 'fault_type': 'outer_ring', 'data_id': '156', 'position': 'relative'},
    158: {'fault_diameter': 0.1778, 'fault_type': 'outer_ring', 'data_id': '158', 'position': 'relative'},
    159: {'fault_diameter': 0.1778, 'fault_type': 'outer_ring', 'data_id': '159', 'position': 'relative'},
    160: {'fault_diameter': 0.1778, 'fault_type': 'outer_ring', 'data_id': '160', 'position': 'relative'},

    169: {'fault_diameter': 0.3556, 'fault_type': 'inner_ring', 'data_id': '169'},
    170: {'fault_diameter': 0.3556, 'fault_type': 'inner_ring', 'data_id': '170'},
    171: {'fault_diameter': 0.3556, 'fault_type': 'inner_ring', 'data_id': '171'},
    172: {'fault_diameter': 0.3556, 'fault_type': 'inner_ring', 'data_id': '172'},
    185: {'fault_diameter': 0.3556, 'fault_type': 'rolling_element', 'data_id': '185'},
    186: {'fault_diameter': 0.3556, 'fault_type': 'rolling_element', 'data_id': '186'},
    187: {'fault_diameter': 0.3556, 'fault_type': 'rolling_element', 'data_id': '187'},
    188: {'fault_diameter': 0.3556, 'fault_type': 'rolling_element', 'data_id': '188'},
    197: {'fault_diameter': 0.3556, 'fault_type': 'outer_ring', 'data_id': '197', 'position': 'center'},
    198: {'fault_diameter': 0.3556, 'fault_type': 'outer_ring', 'data_id': '198', 'position': 'center'},
    199: {'fault_diameter': 0.3556, 'fault_type': 'outer_ring', 'data_id': '199', 'position': 'center'},
    200: {'fault_diameter': 0.3556, 'fault_type': 'outer_ring', 'data_id': '200', 'position': 'center'},

    209: {'fault_diameter': 0.5334, 'fault_type': 'inner_ring', 'data_id': '209'},
    210: {'fault_diameter': 0.5334, 'fault_type': 'inner_ring', 'data_id': '210'},
    211: {'fault_diameter': 0.5334, 'fault_type': 'inner_ring', 'data_id': '211'},
    212: {'fault_diameter': 0.5334, 'fault_type': 'inner_ring', 'data_id': '212'},
    222: {'fault_diameter': 0.5334, 'fault_type': 'rolling_element', 'data_id': '222'},
    223: {'fault_diameter': 0.5334, 'fault_type': 'rolling_element', 'data_id': '223'},
    224: {'fault_diameter': 0.5334, 'fault_type': 'rolling_element', 'data_id': '224'},
    225: {'fault_diameter': 0.5334, 'fault_type': 'rolling_element', 'data_id': '225'},
    234: {'fault_diameter': 0.5334, 'fault_type': 'outer_ring', 'data_id': '234', 'position': 'center'},
    235: {'fault_diameter': 0.5334, 'fault_type': 'outer_ring', 'data_id': '235', 'position': 'center'},
    236: {'fault_diameter': 0.5334, 'fault_type': 'outer_ring', 'data_id': '236', 'position': 'center'},
    237: {'fault_diameter': 0.5334, 'fault_type': 'outer_ring', 'data_id': '237', 'position': 'center'},
    246: {'fault_diameter': 0.5334, 'fault_type': 'outer_ring', 'data_id': '246', 'position': 'orthogonal'},
    247: {'fault_diameter': 0.5334, 'fault_type': 'outer_ring', 'data_id': '247', 'position': 'orthogonal'},
    248: {'fault_diameter': 0.5334, 'fault_type': 'outer_ring', 'data_id': '248', 'position': 'orthogonal'},
    249: {'fault_diameter': 0.5334, 'fault_type': 'outer_ring', 'data_id': '249', 'position': 'orthogonal'},
    258: {'fault_diameter': 0.5334, 'fault_type': 'outer_ring', 'data_id': '258', 'position': 'relative'},
    259: {'fault_diameter': 0.5334, 'fault_type': 'outer_ring', 'data_id': '259', 'position': 'relative'},
    260: {'fault_diameter': 0.5334, 'fault_type': 'outer_ring', 'data_id': '260', 'position': 'relative'},
    261: {'fault_diameter': 0.5334, 'fault_type': 'outer_ring', 'data_id': '261', 'position': 'relative'},

    3001: {'fault_diameter': 0.7112, 'fault_type': 'inner_ring', 'data_id': '056'},
    3002: {'fault_diameter': 0.7112, 'fault_type': 'inner_ring', 'data_id': '057'},
    3003: {'fault_diameter': 0.7112, 'fault_type': 'inner_ring', 'data_id': '058'},
    3004: {'fault_diameter': 0.7112, 'fault_type': 'inner_ring', 'data_id': '059'},
    3005: {'fault_diameter': 0.7112, 'fault_type': 'rolling_element', 'data_id': '048'},
    3006: {'fault_diameter': 0.7112, 'fault_type': 'rolling_element', 'data_id': '049'},
    3007: {'fault_diameter': 0.7112, 'fault_type': 'rolling_element', 'data_id': '050'},
    3008: {'fault_diameter': 0.7112, 'fault_type': 'rolling_element', 'data_id': '051'},
}

DATA_TYPE_TO_ID = {
    'inner_ring': 0,
    'outer_ring': 1,
    'rolling_element': 2
}
DATA_TYPE_TO_ID_L = {
    'inner_ring_0.1778': 0,
    'inner_ring_0.3556': 1,
    'inner_ring_0.5334': 2,
    'inner_ring_0.7112': 3,
    'rolling_element_0.1778': 4,
    'rolling_element_0.3556': 5,
    'rolling_element_0.5334': 6,
    'rolling_element_0.7112': 7,
    'outer_ring_0.1778': 8,
    'outer_ring_0.3556': 9,
    'outer_ring_0.5334': 10,
}
DATA_TYPE_TO_ID_M = {
    'inner_ring_0.1778': 0,
    'inner_ring_0.3556': 1,
    'inner_ring_0.5334': 2,
    'inner_ring_0.7112': 3,
    'rolling_element_0.1778': 4,
    'rolling_element_0.3556': 5,
    'rolling_element_0.5334': 6,
    'rolling_element_0.7112': 7,
    'outer_ring_0.1778_center': 8,
    'outer_ring_0.1778_orthogonal': 9,
    'outer_ring_0.1778_relative': 10,
    'outer_ring_0.3556_center': 11,
    'outer_ring_0.5334_center': 12,
    'outer_ring_0.5334_orthogonal': 13,
    'outer_ring_0.5334_relative': 14,
}


class DriveDataSet(Dataset):
    """
    ----- For 12k Drive End Bearing Fault Data -----
    num_classes == 3
        The inner ring, outer ring or rolling element are classified
    num_classes == 11
        The inner ring, outer ring or rolling element and their fault diameter are classified
    num_classes == 15
        The damage position of outer ring, inner ring or rolling element and their fault diameter are classified

    is_split == True
        recommend n_mel=64  train data is one second
    is_split == False
        recommend n_mel=256 train data is ten second
    """

    def __init__(self, mode='train', num_classes=3, n_mel=256, is_split=False):
        self.mode = mode
        self.num_classes = num_classes
        self.sr = 12000
        self.n_mel = n_mel
        self.is_split = is_split

        self.data_dir = '12k Drive End Bearing Fault Data'
        self.data_distribution = DATA_DISTRIBUTION
        self.sample_ids = list(self.data_distribution.keys())

        if self.num_classes == 3:
            self.data_type_to_id = DATA_TYPE_TO_ID
        elif self.num_classes == 11:
            self.data_type_to_id = DATA_TYPE_TO_ID_L
        elif self.num_classes == 15:
            self.data_type_to_id = DATA_TYPE_TO_ID_M
        else:
            raise ValueError('num_classes error! you can reselect it in [3, 11, 15]')

        if not is_split:
            np.random.shuffle(self.sample_ids)
            if self.mode == 'train':
                self.sample_ids = self.sample_ids[:-int(len(self.sample_ids) * 0.2)]
            else:
                self.sample_ids = self.sample_ids[-int(len(self.sample_ids) * 0.2):]

            self.signals, self.labels = [], []
            self.get_data()
        else:
            self.signals, self.labels = [], []
            self.get_data()
            c = list(zip(self.signals, self.labels))
            np.random.shuffle(c)
            self.signals, self.labels = zip(*c)
            del c
            num = int(len(self.signals) * 0.2)
            if self.mode == 'train':
                self.signals = self.signals[:-num]
                self.labels = self.labels[:-num]
            else:
                self.signals = self.signals[-num:]
                self.labels = self.labels[-num:]

    def get_data(self):
        for sample_id in tqdm(self.sample_ids):
            # get sample info
            sample_info = self.data_distribution[sample_id]
            sample_path = os.path.join('dataset', self.data_dir, f'{sample_id}.mat')
            sample_data_id = sample_info['data_id']

            # load data what we need and reshape to one-dimensional
            wave_data = scio.loadmat(sample_path)[f'X{sample_data_id}_DE_time'].reshape(-1)
            # make sure wave data shape = 12000*10
            wave_data = wave_data[int(len(wave_data) // 2 - self.sr * 5):int(len(wave_data) // 2 + self.sr * 5)]
            if not self.is_split:
                self.append_data(wave_data, sample_info)
            else:
                for i in range(10):
                    self.append_data(wave_data[i * self.sr:(i + 1) * self.sr], sample_info)

    def append_data(self, wave_data, sample_info):
        # mel spectrogram
        mel_spec = librosa.feature.melspectrogram(wave_data, sr=self.sr, n_mels=self.n_mel)
        # log mel spectrogram
        log_mel_spec = librosa.amplitude_to_db(mel_spec)
        # normalization
        log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 0.00001)
        # (w,h) => (1,w,h)
        log_mel_spec = np.expand_dims(log_mel_spec, axis=0)
        # to tensor
        sample = torch.tensor(log_mel_spec, dtype=torch.float32)
        # (1,w,h) => (3,w,h)
        sample = torch.cat([sample, sample, sample], dim=0)

        # get sample label
        sample_fault_type = sample_info['fault_type']
        sample_fault_diameter = sample_info['fault_diameter']
        if self.num_classes == 3:
            index = f"{sample_fault_type}"
        elif self.num_classes == 11:
            index = f"{sample_fault_type}_{sample_fault_diameter}"
        elif self.num_classes == 15:
            if sample_fault_type == 'outer_ring':
                index = f"{sample_fault_type}_{sample_fault_diameter}_{sample_info['position']}"
            else:
                index = f'{sample_fault_type}_{sample_fault_diameter}'
        label = self.data_type_to_id[index]
        label = torch.tensor([label], dtype=torch.long)
        self.signals.append(sample)
        self.labels.append(label)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, item):
        return self.signals[item], self.labels[item]

