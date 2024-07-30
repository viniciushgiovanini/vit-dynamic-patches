import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig, ViTForImageClassification
import pytorch_lightning as pl
from classes.patch_visualizer import PatchVisualizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class ViTPatchEmbeddingsCustom(nn.Module):
#     def __init__(self, input_size, patch_size, embed_dim):
#         super(ViTPatchEmbeddingsCustom, self).__init__()
#         self.patch_size = patch_size
#         self.embed_dim = embed_dim

#         # Projeção dos patches
#         self.projection = nn.Conv2d(
#             in_channels=input_size[0],  # Número de canais da imagem
#             out_channels=embed_dim,      # Dimensão do embedding
#             kernel_size=patch_size,      # Tamanho do patch
#             stride=patch_size            # Deslocamento da janela
#         )
    
#     def forward(self, x, **kwargs):
#       # Passar pela camada de projeção
#       x = self.projection(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
      
#       # Obter a forma de saída
#       batch_size, embed_dim, num_patches_h, num_patches_w = x.size()
      
#       # Reformatar para (batch_size, num_patches, embed_dim)
#       x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)
      
#       return x


class RandomPatchEmbedding(nn.Module):
    def __init__(self, input_size, patch_size, embed_dim, num_patches):
        super(RandomPatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
        
        # Camada de projeção dos patches
        self.projection = nn.Linear(patch_size[0] * patch_size[1] * input_size[0], embed_dim)
    
    def forward(self, x, **kwargs):
        """
        Args:
            x (torch.Tensor): Imagem de entrada com forma (batch_size, channels, height, width)
        
        Returns:
            torch.Tensor: Embeddings dos patches com forma (batch_size, num_patches, embed_dim)
        """
        batch_size, channels, height, width = x.size()
        
        # Calcular o número total de patches possíveis
        num_patches_h = height // self.patch_size[0]
        num_patches_w = width // self.patch_size[1]
        
        # Gerar posições aleatórias para os patches
        random_h_indices = torch.randint(0, num_patches_h, (batch_size, self.num_patches), device=x.device)
        random_w_indices = torch.randint(0, num_patches_w, (batch_size, self.num_patches), device=x.device)
        
        patches = []
        for i in range(self.num_patches):
            h_idx = random_h_indices[:, i]
            w_idx = random_w_indices[:, i]
            
            # Convertendo índices para inteiros para indexação
            h_start = h_idx.unsqueeze(1) * self.patch_size[0]
            w_start = w_idx.unsqueeze(1) * self.patch_size[1]
            
            # Extração de patches com sobreposição
            patch = torch.stack([
                x[b, :, h_start[b].item():h_start[b].item() + self.patch_size[0],
                w_start[b].item():w_start[b].item() + self.patch_size[1]]
                for b in range(batch_size)
            ])
            
            patch = patch.flatten(start_dim=1)  # Flatten para (batch_size, num_patches, patch_size * patch_size * channels)
            patch = self.projection(patch)      # Projeção para (batch_size, num_patches, embed_dim)
            patches.append(patch)
        
        patches = torch.stack(patches, dim=1)  # (batch_size, num_patches, embed_dim)
        
        return patches


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