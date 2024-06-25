import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import text_processing
import video_processing
import audio_processing

class MultimodalLSTM(nn.Module):
    def __init__(self, text_modal, audio_modal, video_modal, hidden_size, output_size):
        super(MultimodalLSTM, self).__init__()
        self.text_layer = nn.LSTM(input_size=text_modal, hidden_size=hidden_size, batch_first=True)
        self.audio_layer = nn.LSTM(input_size=audio_modal, hidden_size=hidden_size, batch_first=True)
        self.video_layer = nn.LSTM(input_size=video_modal, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 3, output_size)

    def forward(self, x1, x2, x3):
        _, (h1, _) = self.text_layer(x1)
        _, (h2, _) = self.audio_layer(x2)
        _, (h3, _) = self.video_layer(x3)
        combined = torch.cat((h1[-1], h2[-1], h3[-1]), dim=0)
        output = torch.sigmoid(self.fc(combined))
        return float(output)
    
def make_predictions(input_data1, input_data2, input_data3):
    model = MultimodalLSTM(300, 74, 388, 64, 1)
    model.load_state_dict(torch.load('./models/multimodal_lstm_model_10_epochs.pth'))
    model.eval()

    output = model(input_data1, input_data2, input_data3)

    return output

def process_pds(transcript, covarep, clnf_au, clnf_feat, clnf_feat3d, clnf_gaze, clnf_pose):
    text_tensor = text_processing.return_tensor(transcript)
    audio_tensor = audio_processing.return_tensor(covarep)
    video_tensor = video_processing.return_tensor(clnf_au, clnf_feat, clnf_feat3d, clnf_gaze, clnf_pose)

#    return(str(text_tensor.shape) + str(audio_tensor.shape) + str(video_tensor.shape))
    return make_predictions(text_tensor, audio_tensor, video_tensor)

'''transcript = pd.read_csv('daic_woz/dev_data/490/490_TRANSCRIPT.csv', delimiter = '\t', encoding = 'utf-8', engine = 'python')

covarep = pd.read_csv('daic_woz/dev_data/490/490_COVAREP.csv', header = None)

clnf_au = pd.read_csv('daic_woz/dev_data/490/490_CLNF_AUs.txt', delimiter = ',', engine = 'python')
clnf_feat = pd.read_csv('daic_woz/dev_data/490/490_CLNF_features.txt', delimiter = ',', engine = 'python')
clnf_feat3d = pd.read_csv('daic_woz/dev_data/490/490_CLNF_features3D.txt', delimiter = ',', engine = 'python')
clnf_gaze = pd.read_csv('daic_woz/dev_data/490/490_CLNF_gaze.txt', delimiter = ',', engine = 'python')
clnf_pose = pd.read_csv('daic_woz/dev_data/490/490_CLNF_pose.txt', delimiter = ',', engine = 'python')'''

'''text_tensor = text_processing.return_tensor(transcript)
audio_tensor = audio_processing.return_tensor(covarep)
video_tensor = video_processing.return_tensor(clnf_au, clnf_feat, clnf_feat3d, clnf_gaze, clnf_pose)

output = make_predictions(text_tensor, audio_tensor, video_tensor)
print(output)'''