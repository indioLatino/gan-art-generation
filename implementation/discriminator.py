import numpy as np
import torch
import torch.nn as nn

n_epochs=1000
batch_size=100
lr=0.0002
b1=0.5
b2=0.999
n_cpu=8
latent_dim=100
img_size=128
channels=3
sample_interval=400


img_shape = (channels, img_size, img_size)
cuda = True if torch.cuda.is_available() else False

class Discriminador(nn.Module):
    def __init__(self):
        super(Discriminador, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, imagen):
        imagen_aplanada = imagen.view(imagen.size(0), -1)
        resultado = self.model(imagen_aplanada)
        return resultado
