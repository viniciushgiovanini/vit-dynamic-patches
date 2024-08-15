import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTModel
import pytorch_lightning as pl
from classes.patch_visualizer import PatchVisualizer
from classes.dynamic_patches import DynamicPatches


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomPatchEmbedding(nn.Module):
    def __init__(self, input_size, patch_size, embed_dim, num_patches, is_visualizer):
        super(CustomPatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.is_visualizer = is_visualizer
        
        # Projeta os patches para uma dimensão `embed_dim`
        self.projection = nn.Linear(patch_size[0] * patch_size[1] * input_size[0], embed_dim)
        # Lida com a visualizacao de patches
        self.visualizer = PatchVisualizer(patch_size)
    
    
    def forward(self, x, **kwargs):
      
        # X -> Tensor de entrada (batch_size, channels, height, width)

        batch_size, channels, height, width = x.size()
        
        # armazena patches extraidos
        all_patches = []
        
        # armazena os índices dos centros dos patches
        all_h_indices = []
        all_w_indices = []

        # Loop sobre cada img do batch
        for b in range(batch_size):

            # Seleciona os centros de acordo com o metodo escolhido
            # centers = DynamicPatches().generate_random_patch_centers(height, width, self.patch_size, self.num_patches)
            centers = DynamicPatches().generate_patch_centers(height, width, self.patch_size)
            
            # converter as cordernadas do centers em indices inteiros  
            h_indices = [int(h) for h, _ in centers]
            w_indices = [int(w) for _, w in centers]
            
            patches = []

            # Para cada par ded indices h,w
            for (h_idx, w_idx) in zip(h_indices, w_indices):
              
                # Calcular as coordenadas de início do patch
                start_h = h_idx - self.patch_size[0] // 2
                start_w = w_idx - self.patch_size[1] // 2
                
                
                # Calcular as coordenadas de fim do patch
                end_h = start_h + self.patch_size[0]
                end_w = start_w + self.patch_size[1]
              
                # Verificar se o patch está dentro dos limites da imagem
                if (0 <= start_h and start_h + self.patch_size[0] <= height and
                    0 <= start_w and start_w + self.patch_size[1] <= width):
                    
                    # Extrair o patch
                    patch = x[b, :, start_h:end_h, start_w:end_w]
                    patches.append(patch.to(device))
                else:
                    print(f"Patch fora dos limites: start_h={start_h}, end_h={end_h}, start_w={start_w}, end_w={end_w}")
            
            # se o numero patches for menor que o ncessário, prenche com tesnores vazios
            
            print(len(patches))
            print(self.num_patches)
            
            
            if len(patches) < self.num_patches:
                print("AAAAAAAAAAAAAAAAAAAAA\n\n\n\n\n\n\n\n\n")
                missing_patches = self.num_patches - len(patches)
                patches += [torch.zeros(channels, self.patch_size[0], self.patch_size[1], device=device)] * missing_patches

            # Concatena os patches em um unico tensor
            patches = torch.stack(patches)  

            # faz o flatten
            patches = patches.flatten(start_dim=1)
            
            # projeta os patches para um espaço de maior dimensão
            patches = self.projection(patches)    
            
            all_patches.append(patches)
            all_h_indices.append(h_indices)
            all_w_indices.append(w_indices)
        
        # self.visualizer.save_patches_to_file(all_patches=all_patches, output_dir='/figs/batch_0/', batch_idx=0)
        
        # combina todos os patches de todas as imagens no batch em um uico tensor tridimensional (batch_size, num_patches, embed_dim)
        all_patches = torch.stack(all_patches).to(device) 
        
        if self.is_visualizer:
          for x, h, w in zip(x, all_h_indices, all_w_indices):
            self.visualizer.visualize_patches(x.cpu(), h, w)
        
        return all_patches
           
      
class ModeloCustom(pl.LightningModule):
    def __init__(self, num_class, learning_rate):
        super(ModeloCustom, self).__init__()
        
        
        self.num_class = num_class
        self.learning_rate = learning_rate
      
        # Carregar um modelo pré-treinado
        base_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        # base_model = ViTModel.from_pretrained('amunchet/rorshark-vit-base')
        # base_model = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')
        # self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384', num_labels=self.num_class, ignore_mismatched_sizes=True)
        
        self.model = ViTForImageClassification(config=base_model.config)
        self.model.vit = base_model
        
        self.model.vit.embeddings.patch_embeddings = CustomPatchEmbedding(
            input_size=(3, 224, 224),  # Ajustar o tamanho da imagem de entrada
            patch_size=(16, 16),       # Tamanho do patch
            embed_dim=self.model.config.hidden_size,
            num_patches=196,
            is_visualizer=False,
        )
              
        print(self.model)
        
        self.model.to(device)
        print("----------------------------------------------------------------")
        
        # Congelar o modelo Todo
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Congelar a camada de embeddings original
        for param in self.model.vit.embeddings.parameters():
            param.requires_grad = False

        # Descongelar o classificador e o pooler
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        for param in self.model.vit.pooler.parameters():
            param.requires_grad = True

        # Descongelar algumas camadas finais do encoder
        num_layers_to_unfreeze = 6 
        for i in range(-num_layers_to_unfreeze, 0):
            for param in self.model.vit.encoder.layer[i].parameters():
                param.requires_grad = True
        
        self.model.classifier = torch.nn.Linear(base_model.config.hidden_size, self.num_class)
          
        # self.model.classifier = nn.Sequential(
        #     nn.Linear(self.model.config.hidden_size, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(128, self.num_class)
        # )

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