import timm
from torch import nn
import torch.nn.functional as F


class CassvaImgTrunk(nn.Module):
    def __init__(self, model_arch, n_class, output_dim, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        # n_features = self.model.classifier.in_features
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, output_dim)
        # self.model.classifier = nn.Linear(n_features, CFG['metric_hidden'])
        '''
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
            nn.Linear(n_features, n_class, bias=True)
        )
        '''

    def forward(self, x):
        x = self.model(x)
        return x


class CassvaImgEmbedder(nn.Module):
    def __init__(self, num_features):
        super(CassvaImgEmbedder, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.Linear(num_features, num_features // 2)
        self.relu1 = nn.ReLU()

        self.batch_norm2 = nn.BatchNorm1d(num_features // 2)
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(num_features // 2, num_features // 4)

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.relu1(self.dense1(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x


class CassvaImgClassifier(nn.Module):
    def __init__(self, num_features, n_class, hidden_size):
        super(CassvaImgClassifier, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.Linear(num_features, hidden_size)

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(hidden_size, int(hidden_size / 2))

        self.batch_norm3 = nn.BatchNorm1d(int(hidden_size / 2))
        self.dropout3 = nn.Dropout(0.3)
        self.dense3 = nn.Linear(int(hidden_size / 2), n_class)

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x
