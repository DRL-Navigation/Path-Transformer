import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple

def mlp(input_mlp: List[Tuple[int, int, str]]) -> nn.Sequential:
    if not input_mlp:
        return nn.Sequential()
    mlp_list = []
    for input_dim, out_put_dim, af in input_mlp:
        mlp_list.append(nn.Linear(input_dim, out_put_dim, bias=True))
        if af == "relu":
            mlp_list.append(nn.ReLU())
        if af == 'sigmoid':
            mlp_list.append(nn.Sigmoid())
    return nn.Sequential(*mlp_list)

class StateEmbed(nn.Module):
    def __init__(self, dim=512):
        super(StateEmbed, self).__init__()
        # self.conv1 = torch.nn.Conv2d(3, 64, 7, stride=1, padding=(1,1))
        # self.conv2 = torch.nn.Conv2d(64, 128, 5, stride=1, padding=(1,1))
        # self.conv3 = torch.nn.Conv2d(128, 256, 3, stride=1, padding=(1,1))
        # self.fc_2d = mlp([(6400, dim, "relu")])
        # self.conv1d1 =  torch.nn.Conv1d(1, 32, 5, 2, "valid")
        # self.conv1d2 =  torch.nn.Conv1d(32, 32, 3, 2, "valid")
        # self.fc_1d = mlp([(7616, dim, "relu")])
        self.fc_1d = mlp([(90, dim, "relu")])
        self.fc = mlp([(3, dim, "relu")])

    # def _encode_image(self, image):
    #     image_x = F.max_pool2d(F.relu(self.conv1(image), inplace=True), 2, stride=2)
    #     image_x = F.max_pool2d(F.relu(self.conv2(image_x), inplace=True), 2, stride=2)
    #     image_x = F.max_pool2d(F.relu(self.conv3(image_x), inplace=True), 2, stride=2)
    #     image_x = self.fc_2d(image_x.reshape(image_x.shape[0], -1))
    #     return image_x

    def _encode_laser(self, laser):
        # x = self.conv1d1(x)
        # x = self.conv1d2(x)
        # x = self.fc_1d(x.view(x.shape[0], -1))
        laser_list = laser.split((laser.shape[1]//4,)*4, dim=1)
        lasers = [self.fc_1d(laser_split) for laser_split in laser_list]
        return lasers

    def forward(self, states):
        #Size: (Batch, Length, Dim)
        #states: [Sensor, Vector, Ped]
        batch_size, seq_length = states[0].shape[0], states[0].shape[1]
        laser = states[0].reshape((batch_size*seq_length, -1))
        #ped = states[2].reshape((batch_size*seq_length,)+states[2].shape[2:])
        vector = states[1].reshape((batch_size*seq_length,)+states[1].shape[2:])[:,0:3]
        laser = self._encode_laser(laser)
        laser = [l.reshape(batch_size, seq_length, -1) for l in laser]
        #ped = self._encode_image(ped).reshape(batch_size, seq_length, -1)
        vector = self.fc(vector).reshape(batch_size, seq_length, -1)
        laser.append(vector)
        return laser

class RewardEmbed(nn.Module):
    def __init__(self, dim=512):
        super(RewardEmbed, self).__init__()
        self.fc = mlp([(1, dim, "None")])

    def forward(self, reward):
        return self.fc(reward)

class PathEmbed(nn.Module):
    def __init__(self, dim=512):
        super(PathEmbed, self).__init__()
        self.embed = nn.Embedding(361, dim)

    def _encode_path(self, path):
        batch_size, seq_length = path.shape[0], path.shape[1]
        return self.embed(path).reshape(batch_size, seq_length, -1)

    def forward(self, path):
        return [self._encode_path(path_i) for path_i in path]

class PathPredict(nn.Module):
    def __init__(self, dim=512):
        super(PathPredict, self).__init__()
        self.fc = nn.Linear(dim, 361, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, token):
        return self.softmax(self.fc(token))

class StateEmbed_ReturnPred(nn.Module):
    def __init__(self, dim=512):
        super(StateEmbed_ReturnPred, self).__init__()
        # self.conv1 = torch.nn.Conv2d(3, 64, 7, stride=1, padding=(1,1))
        # self.conv2 = torch.nn.Conv2d(64, 128, 5, stride=1, padding=(1,1))
        # self.conv3 = torch.nn.Conv2d(128, 256, 3, stride=1, padding=(1,1))
        # self.fc_2d = mlp([(6400, 256, "relu")])
        self.conv1d1 =  torch.nn.Conv1d(1, 32, 5, 2, "valid")
        self.conv1d2 =  torch.nn.Conv1d(32, 32, 3, 2, "valid")
        self.fc_1d = mlp([(2816, 512, "relu")])
        # self.fc0 = mlp([(512+256, 512, "relu")])
        # self.fc1 = mlp([(512 + 3, dim, "relu")])
        self.fc = mlp([(512+3, dim, "relu")])

    # def _encode_image(self, image):
    #     image_x = F.max_pool2d(F.relu(self.conv1(image), inplace=True), 2, stride=2)
    #     image_x = F.max_pool2d(F.relu(self.conv2(image_x), inplace=True), 2, stride=2)
    #     image_x = F.max_pool2d(F.relu(self.conv3(image_x), inplace=True), 2, stride=2)
    #     image_x = self.fc_2d(image_x.reshape(image_x.shape[0], -1))
    #     return image_x

    def _encode_laser(self,x ):
        x = self.conv1d1(x)
        x = self.conv1d2(x)
        x = self.fc_1d(x.view(x.shape[0], -1))
        return x

    def forward(self, states):
        #Size: (Batch, Dim)
        #states: [Sensor, Vector, Ped]
        laser = states[0]
        # ped = states[2]
        vector = states[1][:,0:3]
        laser = self._encode_laser(laser)
        # ped = self._encode_image(ped)
        # x = self.fc0(torch.cat([laser, ped], dim=1))
        x = torch.cat([laser, vector], dim=1)
        x = self.fc(x)
        return x