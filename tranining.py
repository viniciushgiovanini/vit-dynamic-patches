import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import sys
import time
import shutil
import pandas as pd
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.accelerators import find_usable_cuda_devices
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from classes.modelo_custom import ModeloCustom
from classes.modelo import Modelo
from classes.modelo_binario import ModeloBin


# Identificar GPUs disponíveis
devices = find_usable_cuda_devices()
print("Dispositivos CUDA disponíveis:", devices)

print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION', )
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
print('Active CUDA Device: GPU', torch.cuda.current_device())
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())


#########################
#      HYPERPARAMS
#########################
start_time = time.time()
batch_size = 64
num_epochs = 100
learning_rate = 0.0001
# total_steps = 100
img_size = (224, 224)
patch_size = (16,16)

# Dataset path
train_data_path = './data/base_treinamento/train/'
test_data_path = './data/base_treinamento/test/'

# Transformando a imagem test
transform = v2.Compose([
    v2.Resize(img_size),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cria o path para os logs do lightning
if os.path.exists("./lightning_logs/"):
  shutil.rmtree("./lightning_logs/")
if os.path.exists("./models/"):
  shutil.rmtree("./models/")
  os.mkdir("./models/") 
if not os.path.exists("./models/"):
  os.mkdir("./models/") 
  

##########################
# Carregando Dados
##########################
train_dataset = ImageFolder(root=train_data_path, transform=transform)
test_dataset = ImageFolder(root=test_data_path, transform=transform)

#########################
# Lendo Classes
#########################
class_to_idx = train_dataset.class_to_idx
print("Class to Index Mapping:")
print(class_to_idx)
num_classes = len(train_dataset.classes)
print(f"Numero de classes {num_classes}")

# Seleciona as steps automaticamente
total_steps = len(train_dataset) // batch_size

# Seleciona o Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n\n Device ---> {device} and Current Device --> {torch.cuda.current_device()}\n\n")

# Divisão do dataset em Batch, colocando shuffle, acelera o carregando dos dados com num_workers
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=11)
val_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=11)

# Instancia o Modelo criado



num_patch = ((img_size[0]/patch_size[0]) * (img_size[0]/patch_size[0]))
print(f"Numero de patches: {num_patch}\nTamanho da Imagem: {img_size}\nPatch_Size: {patch_size}\n")

# model = ModeloCustom(num_classes, learning_rate, num_patch, img_size[0], patch_size)
model = Modelo(num_classes, learning_rate)
# model = ModeloBin(num_classes, learning_rate)

###########################
# Cria Logger para Metricas
###########################
csv_logger = CSVLogger(
    save_dir='./lightning_logs/',
    name='csv_file'
)

###########################
# Checkpoint Model
###########################
checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',
    dirpath='models/checkpoint/',
    filename='{epoch}-{val_loss:.2f}-{val_accuracy:.2f}',
    save_top_k=1,
    mode='max'
)

#############################
# Criar o Trainer e faz o FIT
#############################
trainer = pl.Trainer(max_epochs=num_epochs,  limit_train_batches= total_steps,limit_val_batches=total_steps, log_every_n_steps=1, logger=[csv_logger, TensorBoardLogger("./lightning_logs/")], accelerator="gpu", devices="auto", callbacks=[checkpoint_callback, ModelSummary(max_depth=10)])
trainer.fit(model, train_loader, val_loader)

# salva modelo Treinado
torch.save(model.state_dict(), './models/modelo_vit_gpu.pth')

#############################################################################
#               Realiza criação do gráfico de loss e acuracia
#############################################################################
df = pd.read_csv('./lightning_logs/csv_file/version_0/metrics.csv')

print("\n\n  %s minutos" % ((time.time() - start_time) / 60 ))

epochs = []
train_accuracy_means = []
train_loss_means = []
val_accuracy_uniques = []
val_loss_uniques = []

for epoch in df['epoch'].unique():
    dados_epoca = df[df['epoch'] == epoch]
    
    train_accuracy_mean = dados_epoca['train_accuracy'].mean()
    
    train_loss_mean = dados_epoca['train_loss'].mean()
    
    val_accuracy_unique = dados_epoca['val_accuracy'].mean()
    
    val_loss_unique = dados_epoca['val_loss'].mean()
    
    epochs.append(epoch)
    train_accuracy_means.append(train_accuracy_mean)
    val_accuracy_uniques.append(val_accuracy_unique)
    train_loss_means.append(train_loss_mean)
    val_loss_uniques.append(val_loss_unique)
    
    
resultados =pd.DataFrame({'epoch': epochs,
                                'train_accuracy': train_accuracy_means,
                                'val_accuracy': val_accuracy_uniques,
                                'train_loss': train_loss_means,
                                'val_loss': val_loss_uniques,                                    
                                },
                              )

# Save graphs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(resultados['epoch'], resultados['train_accuracy'], label='Train Accuracy')
plt.plot(resultados['epoch'], resultados['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(resultados['epoch'], resultados['train_loss'], label='Train Loss')
plt.plot(resultados['epoch'], resultados['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()
plt.savefig("./graph/loss_and_accuracy_pytorch.jpg")


# ############################################################################
#          Calcula e Compara a acuracia do Modelo e da Callback
# ############################################################################

# def calcular_acuracia_multiclasse(model, dataloader):
#     model.to(device) 
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     return correct / total


def calcular_acuracia(model, dataloader):
    model.to(device) 
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).int()
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()
    
    return correct / total

# Calcular a acurácia no conjunto de teste
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=11)
accuracy = calcular_acuracia(model, test_loader)
print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")


best_model_path = checkpoint_callback.best_model_path
print(f"Best model path: {best_model_path}")
model.load_state_dict(torch.load(best_model_path)['state_dict'])

model.to(device)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=11)
accuracy = calcular_acuracia(model, test_loader)
print(f"Acurácia no conjunto de teste (Melhor ponto do modelo): {accuracy * 100:.2f}%")