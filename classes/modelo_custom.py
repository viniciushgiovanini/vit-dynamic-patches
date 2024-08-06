import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig, ViTForImageClassification
import pytorch_lightning as pl
from classes.patch_visualizer import PatchVisualizer

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RandomPatchEmbedding(nn.Module):
    def __init__(self, input_size, patch_size, embed_dim, num_patches):
        super(RandomPatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
        self.projection = nn.Linear(patch_size[0] * patch_size[1] * input_size[0], embed_dim)
        self.visualizer = PatchVisualizer(patch_size)
    
    def generate_random_patch_centers(self, image_height, image_width, patch_size, num_patches):
      patch_height, patch_width = patch_size
      
      centers = []
      for _ in range(num_patches):
          h = random.uniform(patch_height / 2, image_height - patch_height / 2)
          w = random.uniform(patch_width / 2, image_width - patch_width / 2)
          centers.append((h, w))

      return centers
    
    def generate_patch_centers(self, image_height, image_width, patch_size):
      patch_height, patch_width = patch_size
      
      num_patches_h = image_height // patch_height
      num_patches_w = image_width // patch_width
      
      centers_h = [(i * patch_height + patch_height // 2) for i in range(num_patches_h)]
      centers_w = [(j * patch_width + patch_width // 2) for j in range(num_patches_w)]
      
      centers = [(h, w) for h in centers_h for w in centers_w]
      
      return centers
    
    def forward(self, x, **kwargs):
        """
        Args:
            x (torch.Tensor): Imagem de entrada com forma (batch_size, channels, height, width)
        
        Returns:
            torch.Tensor: Embeddings dos patches com forma (batch_size, num_patches, embed_dim)
        """
        batch_size, channels, height, width = x.size()
        
        all_patches = []
        all_h_indices = []
        all_w_indices = []

        for b in range(batch_size):
          
            centers = self.generate_random_patch_centers(height, width, self.patch_size, self.num_patches)
            # centers = self.generate_patch_centers(height, width, self.patch_size)
            
            h_indices = [int(h) for h, _ in centers]
            w_indices = [int(w) for _, w in centers]
            
            patches = []

            for (h_idx, w_idx) in zip(h_indices, w_indices):
                if (0 <= h_idx < height and 0 <= w_idx < width and
                    h_idx + self.patch_size[0] <= height and
                    w_idx + self.patch_size[1] <= width):
                    
                    patch = x[b, :, h_idx:h_idx + self.patch_size[0], w_idx:w_idx + self.patch_size[1]]
                    patches.append(patch.to(device))
            
            if len(patches) < self.num_patches:
                missing_patches = self.num_patches - len(patches)
                patches += [torch.zeros(channels, self.patch_size[0], self.patch_size[1], device=device)] * missing_patches

            patches = torch.stack(patches)  
            patches = patches.flatten(start_dim=1)
            patches = self.projection(patches)    
            
            all_patches.append(patches)
            all_h_indices.append(h_indices)
            all_w_indices.append(w_indices)
        
        all_patches = torch.stack(all_patches).to(device) 
        
        self.visualizer.visualize_patches(x[0].cpu(), all_h_indices[0], all_w_indices[0])
        
        return all_patches
           
      
class ModeloCustom(pl.LightningModule):
    def __init__(self, num_class, learning_rate):
        super(ModeloCustom, self).__init__()
        
        
        self.num_class = num_class
        self.learning_rate = learning_rate
      
        # Carregar um modelo pré-treinado
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=self.num_class, ignore_mismatched_sizes=True)
        # self.model = ViTForImageClassification.from_pretrained('amunchet/rorshark-vit-base', num_labels=self.num_class, ignore_mismatched_sizes=True)
        # self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch32-224-in21k', num_labels=self.num_class, ignore_mismatched_sizes=True)
        # Precisa testar o de baixo
        # self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384', num_labels=self.num_class, ignore_mismatched_sizes=True)
        
        self.model.vit.embeddings.patch_embeddings = RandomPatchEmbedding(
            input_size=(3, 224, 224),  # Ajustar o tamanho da imagem de entrada
            patch_size=(16, 16),       # Tamanho do patch
            embed_dim=self.model.config.hidden_size,
            num_patches=196
        )
        # self.model.vit.embeddings.patch_embeddings = ViTPatchEmbeddingsCustom(
        #     input_size=(3, 224, 224),  # Ajustar o tamanho da imagem de entrada
        #     patch_size=(16, 16),       # Tamanho do patch
        #     embed_dim=self.model.config.hidden_size,
        # )
        
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
            nn.Linear(128, self.num_class)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer