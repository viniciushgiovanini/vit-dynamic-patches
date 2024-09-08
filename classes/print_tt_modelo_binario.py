import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTModel
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModeloBin(pl.LightningModule):
    def __init__(self, num_class, learning_rate):
        super(ModeloBin, self).__init__()
        
        # Salvar os hyperparametros
        self.save_hyperparameters()
        
        
        self.num_class = num_class
        self.learning_rate = learning_rate
      
        # Carregar um modelo pré-treinado
        # base_model = ViTModel.from_pretrained('google/vit-large-patch16-224')
        # base_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        base_model = ViTModel.from_pretrained('WinKawaks/vit-tiny-patch16-224')
        # base_model = ViTModel.from_pretrained('WinKawaks/vit-small-patch16-224')
        # base_model = ViTModel.from_pretrained('amunchet/rorshark-vit-base')
        # base_model = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')
        # self.model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224', num_labels=self.num_class, ignore_mismatched_sizes=True)
        
        self.model = ViTForImageClassification(config=base_model.config)
        self.model.vit = base_model
        print(self.model)
        self.model.to(device)
        print("----------------------------------------------------------------")
        
        # Congela todos os parametros
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Descongela somente o de classificação 
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        # Descongela o MLP
        cont  = 0
        for name, param in self.model.named_parameters():
          if 'encoder.layer' in name and ('intermediate.dense' in name or 'output.dense' in name):
          # if 'encoder.layer' in name and ('output.dense' in name):
            if cont < 12:
              param.requires_grad = True
              cont += 1
        print("Quantidade de Camadas do MLP: " + str(cont))
              
        # self.model.classifier = torch.nn.Linear(base_model.config.hidden_size, self.num_class)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 12),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(12, 12),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(12, 12),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(12, 1)
        )
        # Criterio de Perda é o Bianry Cross Entropy
        self.criterion = nn.BCEWithLogitsLoss()
        print("----------------------------------------------------------------")
        print(self.model)

    # Passagem para frente (Backpropagation) retorna os valores finais do modelo não normalizados
    # Retorna os logits para passar na funcao softmax
    def forward(self, x):
        logits = self.model(x).logits
        return logits

    # Passo a passo do treinamento
    # Batch -> Lote de img (32 img por batch)
    def training_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        logits = self(images)
        loss = self.criterion(logits, labels)
        predicted = torch.sigmoid(logits) > 0.5
        accuracy = (predicted == labels).float().mean()
        
        # Salvar imagens que o modelo acertou durante o treinamento
        correct_images = images[(predicted == labels).squeeze()]
        if batch_idx == 0: 
            self.log_images(correct_images, "train_correct_images")

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy, prog_bar=True)
        
        return loss

    # Faz a mesma coisa do training_step só que na etapa de validação
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        logits = self(images)
        loss = self.criterion(logits, labels)
        predicted = torch.sigmoid(logits) > 0.5
        accuracy = (predicted == labels).float().mean()
        
       
        incorrect_images = images[(predicted != labels).squeeze()]
        if batch_idx == 0:  
            self.log_images(incorrect_images, "val_incorrect_images")

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)

    # Configura o otimizador que é o adam com Learning Rate que passa no (Traning_multiclass)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.001)
        return optimizer
      
    def gerar_string_aleatoria(self, tamanho):
        import string
        import random
        caracteres = string.ascii_letters + string.digits  
        return ''.join(random.choice(caracteres) for _ in range(tamanho))
      
    def log_images(self, images, title):
      num_images = len(images)
      if num_images == 0:
          print(f"No images to display for {title}.")
          return
      
      num_rows = int(np.ceil(num_images / 4))
      fig, ax = plt.subplots(num_rows, 4, figsize=(15, 5 * num_rows))
      
      if num_images == 1:
          ax = np.array([[ax]])
      
      for i in range(num_images):
          current_ax = ax[i // 4, i % 4]
          
          img = np.transpose(images[i].cpu().numpy(), (1, 2, 0))
          
          img = (img - img.min()) / (img.max() - img.min())
          img = np.clip(img, 0, 1)  
          img = (img * 255).astype(np.uint8) 
          
          current_ax.imshow(img)
          current_ax.axis('off')
      
      for j in range(num_images, num_rows * 4):
          ax[j // 4, j % 4].axis('off')
      
      plt.suptitle(title, fontsize=76) 
      plt.savefig(f"figs/log_img/{title}_{self.gerar_string_aleatoria(4)}.png")
      plt.close()