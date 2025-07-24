import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification
import os
from Class.modelo import Modelo


num_classes = 6
learning_rate = 0.0001


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n\n Device ---> {device} and Current Device --> {torch.cuda.current_device()}\n\n")


def calcular_acuracia(model, dataloader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
  

def carregar_e_avaliar_modelo(checkpoint_path, data_path, batch_size=32):
  transform = T.Compose([
      T.Resize((224, 224)),
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  test_dataset = ImageFolder(root=data_path, transform=transform)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=11)

  num_classes = len(test_dataset.classes)
  model = Modelo(learning_rate=learning_rate, num_class=num_classes)

  checkpoint = torch.load(checkpoint_path)
  if 'state_dict' in checkpoint:
      state_dict = checkpoint['state_dict']
  else:
      state_dict = checkpoint

  model.load_state_dict(state_dict)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  accuracy = calcular_acuracia(model, test_loader, device)
  return accuracy



cwd = os.getcwd()
checkpoint_path = 'resultados/google-224-model/google-vit-16-224-100-50-Balanceado/checkpoint/best-checkpoint.ckpt'
data_path = 'data/base_treinamento/test/'

accuracy = carregar_e_avaliar_modelo(checkpoint_path, data_path)
print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")