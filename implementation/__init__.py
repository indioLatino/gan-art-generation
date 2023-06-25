import os
from skimage import io, transform
#import numpy as np
#from google.colab import drive
#import argparse
import numpy as np
#import math
import torch

#import torchvision.transforms as transforms
#from torchvision.utils import save_image

#from torch.utils.data import DataLoader
#from torchvision import datasets
from torch.autograd import Variable

##############################################################
######### HIPER PARÁMETROS DEL MODELO ########################
##############################################################

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



#import torch
import torchvision.transforms.functional as TF

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from implementation.discriminator import Discriminador
from implementation.generator import Generador





################## ACCESO A GOOGLE DRIVE ########################
#drive.mount('/gdrive')
#%cd /gdrive
#os.chdir('/gdrive')
ruta_dataset = 'C:\\Users\\GaloRA\\Máster\\TFM\\archive\\Images'
ruta_guardado_generados = 'C:\\Users\\GaloRA\\Máster\\TFM\\retratos_generados'


################### FUNCIÓN DE LOSS ###############################
gan_loss = torch.nn.BCELoss()
# Declaración del generador y el discriminador
generador = Generador()
discriminador = Discriminador()
cuda = True if torch.cuda.is_available() else False
if cuda:
    print('cudaaaaa')
    generador.cuda()
    discriminador.cuda()
    gan_loss.cuda()

# Optimizadores
optimizador_generador = torch.optim.Adam(generador.parameters(), lr=lr, betas=(b1, b2))
optimizador_discriminador = torch.optim.Adam(discriminador.parameters(), lr=lr, betas=(b1, b2))

# Uso de GPU cuando sea posible
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def imprimir_imagenes(images):
    output = torch.clamp(images, 0, 1)
    output = (output * 255).to(torch.uint8)
    output = output.permute(0, 2, 3, 1)
    print(torch.max(output))
    io.imshow_collection(output)


    #####################################################################
    ########## FUNCIÓN DE ENTRENAMIENTO DE LA RED GAN ###################
    #####################################################################
def entrena_gan(imagenes_reales, optimizador_generador, optimizador_discriminador, epoca):
    # Etiquetas de la red GAN
    reales = Variable(Tensor(imagenes_reales.shape[0], 1).fill_(1.0), requires_grad=False)
    falsas = Variable(Tensor(imagenes_reales.shape[0], 1).fill_(0.0), requires_grad=False)
    # Configure input
    imagenes_reales = Variable(Tensor(imagenes_reales), requires_grad=False)

    #####################################################################
    ############# ENTRENAMIENTO DEL GENERADOR ###########################
    #####################################################################

    optimizador_generador.zero_grad()
    # Vector aleatorio de entrada del generador
    vector_entrada_generador = Variable(Tensor(np.random.normal(0, 1, (imagenes_reales.shape[0], latent_dim))))
    # Generación de imágenes artificiales
    imagenes_generadas = generador(vector_entrada_generador)
    # Función de loss del generador
    gen_loss = gan_loss(discriminador(imagenes_generadas), reales)
    gen_loss.backward()
    optimizador_generador.step()

    #####################################################################
    ############# ENTRENAMIENTO DEL DISCRIMINADOR ###########################
    #####################################################################

    optimizador_discriminador.zero_grad()

    real_loss = gan_loss(discriminador(imagenes_reales), reales)
    falsas_loss = gan_loss(discriminador(imagenes_generadas.detach()), falsas)

    # Función de loss del discriminador
    dis_loss = (real_loss + falsas_loss) / 2

    dis_loss.backward()
    optimizador_discriminador.step()

    print(
        "Epoca %d/%d , Dis loss: %f , Gen loss: %f"
        % (epoca, n_epochs, dis_loss.item(), gen_loss.item())
    )
    imagen_a_guardar = TF.to_pil_image(imagenes_generadas[0])  # Convertir el tensor en una imagen PIL
    ruta_imagen = os.path.join(ruta_guardado_generados, f'imagen_{epoca}.png')  # Ruta de destino para guardar la imagen
    imagen_a_guardar.save(ruta_imagen)





def entrenar_gan(ruta_imagenes, batch_size):
    # Redimensionado y normalización de la imagen
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Cargar imágenes desde Google Drive
    dataset = ImageFolder(ruta_imagenes, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Entrenamiento
    for epoch in range(n_epochs):
        for images, labels in dataloader:
            # Realizar pasos de avance, cálculo de pérdida y retropropagación
            entrena_gan(Variable(images, requires_grad=False), optimizador_generador, optimizador_discriminador, epoch)

    # Guardar el modelo entrenado
    torch.save(generador.state_dict(), "generador_entrenado.pth")
entrenar_gan(ruta_dataset, batch_size)