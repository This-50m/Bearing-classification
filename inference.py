import torch
import time
import repvgg
import librosa
import argparse
import numpy as np
from scipy import io as scio
from data import DATA_DISTRIBUTION, DATA_TYPE_TO_ID, DATA_TYPE_TO_ID_L, DATA_TYPE_TO_ID_M


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_path", "-sample", type=str, default=None,
                        help='The sample address relative to the current home directory')
    parser.add_argument("--num_classes", '-num_classes', type=int, default=15, help="classification nums")
    parser.add_argument('--weights', "-p", type=str, default='logs/repvgg_deploy.pth',
                        help=' pretrained model weights pt file')
    parser.add_argument("--n_mels", '-n_mels', type=int, default=64, help="Number of filter bank matrices")
    parser.add_argument("--is_split", '-split', type=bool, default=True)
    parser.add_argument("--deploy", '-deploy', type=bool, default=True)
    args = parser.parse_args()
    return args


def inference(weight_path, sample_path, n_mel, num_classes, is_split=True, deploy=True):
    assert num_classes in [3, 11, 15]
    print('loading model')
    model = repvgg.create_RepVGG(deploy=deploy, num_classes=num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    print('loading data')
    sr = 12000
    data_id = DATA_DISTRIBUTION[int(sample_path.split('.')[0].split('/')[-1])]['data_id']
    wave_data = scio.loadmat(sample_path)[f'X{data_id}_DE_time'].reshape(-1)
    wave_data = wave_data[int(len(wave_data) // 2 - sr * 5):int(len(wave_data) // 2 + sr * 5)]

    def load_data(wave_data):
        mel_spec = librosa.feature.melspectrogram(wave_data, sr=sr, n_mels=n_mel)
        log_mel_spec = librosa.amplitude_to_db(mel_spec)
        log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 0.00001)
        log_mel_spec = np.expand_dims(log_mel_spec, axis=0)
        sample = torch.tensor(log_mel_spec, dtype=torch.float32)
        sample = torch.cat([sample, sample, sample], dim=0)
        sample = torch.unsqueeze(sample, dim=0)
        return sample
    if not is_split:
        data = load_data(wave_data)
    else:
        data = torch.cat([load_data(wave_data[i*sr:(i+1)*sr]) for i in range(10)], dim=0)

    print('starting prediction')
    if torch.cuda.is_available():
        data = data.cuda()

    output = model(data)
    output = output.cpu().detach().numpy()
    # print(np.max(output, axis=0))
    if not is_split:
        index = int(np.argmax(output))
    else:
        index = int(np.argmax(np.max(output, axis=0)))
    if num_classes == 3:
        data_type_to_id = DATA_TYPE_TO_ID
    elif num_classes == 11:
        data_type_to_id = DATA_TYPE_TO_ID_L
    elif num_classes == 15:
        data_type_to_id = DATA_TYPE_TO_ID_M
    result = list(data_type_to_id.keys())[index]
    print(result)


if __name__ == '__main__':
    parsers = get_args()
    sample_path = parsers.sample_path
    weight_path = parsers.weights
    num_classes = parsers.num_classes
    n_mels = parsers.n_mels
    inference(weight_path, sample_path, n_mel=n_mels, num_classes=num_classes)
