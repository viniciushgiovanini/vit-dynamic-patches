import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
import torchvision.models as models
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torchvision

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
from torchvision.datasets import ImageFolder


from lightning.pytorch.accelerators import find_usable_cuda_devices

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

start_time = time.time()
batch_size = 32
num_epochs = 60
learning_rate = 0.0001
# learning_rate = 0.00001

train_data_path = './data/base_treinamento/train/'
test_data_path = './data/base_treinamento/test/'

dataset = ImageFolder(root=train_data_path)

class_to_idx = dataset.class_to_idx


print("Class to Index Mapping:")
print(class_to_idx)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if os.path.exists("./lightning_logs/"):
  shutil.rmtree("./lightning_logs/")

train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)

num_classes = len(train_dataset.classes)

print(f"Numero de classes {num_classes}")

# total_steps = len(train_dataset) // batch_size
total_steps = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n\n Device ---> {device} and Current Device --> {torch.cuda.current_device()}\n\n")

class Modelo(pl.LightningModule):
    def __init__(self):
        super(Modelo, self).__init__()
        # Carregar um modelo pré-treinado
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=num_classes, ignore_mismatched_sizes=True)
        print(self.model)
        self.model.to(device)
        print("----------------------------------------------------------------")
        for param in self.model.parameters():
            param.requires_grad = False
            
        for param in self.model.classifier.parameters():
            param.requires_grad = True
            
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()
        print(self.model)
        print("----------------------------------------------------------------")
        # for name, module in self.model.named_modules():
        #   print(f"{name}: {module}")

    def forward(self, x):
        logits = self.model(x).logits
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        logits = self(images)
        loss = self.criterion(logits, labels)
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == labels).float().mean()
                
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        logits = self(images)
        loss = self.criterion(logits, labels)
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=11)
val_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=11)

modelo = Modelo()

csv_logger = CSVLogger(
    save_dir='./lightning_logs/',
    name='csv_file'
)

trainer = pl.Trainer(max_epochs=num_epochs,  limit_train_batches= total_steps,limit_val_batches=total_steps, log_every_n_steps=1, logger=[csv_logger, TensorBoardLogger("./lightning_logs/")], accelerator="gpu", devices="auto")

trainer.fit(modelo, train_loader, val_loader)

torch.save(modelo.state_dict(), './models/modelo_vit_gpu.pth')

df = pd.read_csv('./lightning_logs/csv_file/version_0/metrics.csv')

print("\n\n  %s minutos" % ((time.time() - start_time) / 60 ))

epochs = []
train_accuracy_means = []
train_loss_means = []
val_accuracy_uniques = []
val_loss_uniques = []

for epoch in df['epoch'].unique():
    dados_epoca = df[df['epoch'] == epoch]
    
    # filtered_df = dados_epoca.tail(2)
    
    # train_accuracy_mean = filtered_df.iloc[0]["train_accuracy"]
    # train_loss_mean = filtered_df.iloc[0]["train_loss"]
    # val_accuracy_unique = filtered_df.iloc[1]["val_accuracy"]
    # val_loss_unique = filtered_df.iloc[1]["val_loss"]
    
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


def calcular_acuracia(model, dataloader):
    model.to(device)  # Mover o modelo para o dispositivo correto
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)  # Mover dados para o dispositivo correto
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Calcular a acurácia no conjunto de teste
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=11)
accuracy = calcular_acuracia(modelo, test_loader)
print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")