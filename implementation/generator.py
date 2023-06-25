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

class Generador(nn.Module):
    def __init__(self):
        super(Generador, self).__init__()

        # función para generar redes neuronales con la siguiente estructura:
        # capa lineal -> capa de normalización (opcional) -> capa RELU
        # a partir de los parámetros:
        # dim_in: Dimensionalidad del vector de entrada
        # dim_out: Dimensionalidad del vector de salida
        # normalizar: flag para normalizar o no el resultado de la capa lineal interior
        def genera_capa(dim_in, dim_out, normalizar=True):
            capas = []  # lista de capas que tendrá la sub-red
            capas.append(nn.Linear(dim_in, dim_out))
            if normalizar:
                capas.append(nn.BatchNorm1d(dim_out, 0.8))
            capas.append(nn.LeakyReLU(0.2, inplace=True))
            return capas

        self.model = nn.Sequential(
            *genera_capa(latent_dim, 128, normalizar=False),
            *genera_capa(128, 256),
            *genera_capa(256, 512),
            *genera_capa(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, vector):
        img = self.model(vector)
        img = img.view(img.size(0), *img_shape)
        return img
