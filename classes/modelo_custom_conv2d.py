import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTModel
import pytorch_lightning as pl
from classes.patch_visualizer import PatchVisualizer
from classes.dynamic_patches import DynamicPatches
import matplotlib.pyplot as plt
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_names_dict = {}

class CustomPatchEmbedding(nn.Module):
    def __init__(self, input_size, patch_size, embed_dim, num_patches, is_visualizer):
        super(CustomPatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.is_visualizer = is_visualizer
        
        
        # Projeta os patches para uma dimensão `embed_dim`
        # self.projection = nn.Linear(patch_size[0] * patch_size[1] * input_size[0], embed_dim)
        
        
        self.projection = nn.Conv2d(
            in_channels=input_size[0],   # Número de canais da imagem de entrada
            out_channels=embed_dim,       # Dimensão de saída do embedding
            kernel_size=patch_size[0],       # Tamanho do patch
            stride=patch_size[0]             # Avanço igual ao tamanho do patch
        )
        
        
        
        # Lida com a visualizacao de patches
        # self.visualizer = PatchVisualizer(patch_size)
        
        # Arquivo dos centros randomicos melhorados
        # self.dict_center = self.load_dict('./data/centros_pre_salvos/randomico_melhorado_identificador_por_imgname.pkl')
        
        # Arquivo dos centros segmentados
        self.dict_center = self.load_dict('./data/centros_pre_salvos/segmentacao_dicionario.pkl')
        
        
    
    def load_dict(self, path):
      with open(path, 'rb') as f:
        lista_centro_dict  = pickle.load(f)
        
      return lista_centro_dict
    
    def forward(self, x, **kwargs):
        
        # X -> Tensor de entrada (batch_size, channels, height, width)

        batch_size, channels, height, width = x.size()
        
        # armazena patches extraidos
        all_patches = []
        
        # armazena os índices dos centros dos patches
        all_h_indices = []
        all_w_indices = []

        
        ################################################
        #                   Print de Log               #
        ################################################
        # print("Iniciou um loop de batch\n")
        # print(f"Printando de dentro do CustomPatchEmbedding: {image_names_dict}")
        
        each_image = {}
        
        # Loop sobre cada img do batch
        for b in range(batch_size):
            # Seleciona os centros de acordo com o metodo escolhido
            
            try:
              centers = self.dict_center[image_names_dict[b]]
            except:
              print(f"Erro ao encontrar centro --> {image_names_dict[b]}")
                    
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
                    # print(f'Shape antes da conv: {patch.shape}')
                    output_patches = self.projection(patch)
                    # print(f'Shape depois da conv: {output_patches.shape}')

                    # fazendo o flatten
                    patches.append(output_patches.view(-1))
                    
                    # patches.append(output_patches)
                else:
                    print(f"Patch fora dos limites: start_h={start_h}, end_h={end_h}, start_w={start_w}, end_w={end_w}")
            # a = torch.stack(patches)
            # each_image[image_name[b]] = torch.stack(patches)
            
            each_image[image_names_dict[b]] = torch.stack(patches)
            
            # print(f"Shape do tensor da imagem {image_name[b]} é {each_image[image_name[b]].shape}")
            # print(f"Shape de um patch: {patches[0].shape}")
            # print(f"Shape de todos os patches empilhados: {each_image[image_key].shape}")
            # str("")
        
        
        
        all_images = torch.stack(list(each_image.values()))  # Shape: [batch_size, num_patches, output_channels]
        # print(f"Forma final das imagens: {all_images.shape}")
        return all_images
           
      
class ModeloCustomConv2d(pl.LightningModule):
    def __init__(self, num_class, learning_rate, num_patch, input_size, patch_size, batch_size):
        super(ModeloCustomConv2d, self).__init__()
        
        self.save_hyperparameters()
        
        self.num_class = num_class
        self.learning_rate = learning_rate
        self.layer_dropout = nn.Dropout(0.4)
        self.batch_size = batch_size
      
        # Carregar um modelo pré-treinado
        base_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        # base_model = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')
        
        # Acessar os pesos da camada Conv2d da projeção do modelo base
        pretrained_conv_weights = base_model.embeddings.patch_embeddings.projection.weight.data.clone()
        pretrained_conv_bias = base_model.embeddings.patch_embeddings.projection.bias.data.clone()
        
        self.model = ViTForImageClassification(config=base_model.config)
        self.model.vit = base_model
        
        self.model.vit.embeddings.patch_embeddings = CustomPatchEmbedding(
            input_size=(3, input_size, input_size),  
            patch_size=patch_size,       
            embed_dim=self.model.config.hidden_size,
            num_patches=num_patch,
            is_visualizer=True,
        )
              
        
        
        # Copiar os pesos e bias para a sua camada personalizada
        self.model.vit.embeddings.patch_embeddings.projection.weight.data.copy_(pretrained_conv_weights)
        self.model.vit.embeddings.patch_embeddings.projection.bias.data.copy_(pretrained_conv_bias)

        print(self.model)
        self.model.to(device)
        print("----------------------------------------------------------------")
        # print(f'Weight data: {self.model.vit.embeddings.patch_embeddings.projection.weight.data} \n\n\n\n\n\n\n')  



        

        # Congela todos os parametros
        for param in self.model.parameters():
            param.requires_grad = False

        # Descongela somente o de classificação
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # Descongelar as camadas específicas
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in [
                "vit.embeddings.patch_embeddings.projection",
                "vit.encoder.layer.11.intermediate",
                "vit.encoder.layer.11.output",
                "vit.encoder.layer.11.layernorm",
                "vit.layernorm",
                "vit.pooler"
            ]):
                param.requires_grad = True
        
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.ReLU(),
            self.layer_dropout,
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.ReLU(),
            self.layer_dropout,
            nn.Linear(self.model.config.hidden_size, self.num_class)
        )

        # Conferir as camadas que foram descongeladas
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Layer {name} is trainable")
        # Criterio de Perda é o CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss()
        print("----------------------------------------------------------------")
        print(self.model)

    # Passagem para frente (Backpropagation) retorna os valores finais do modelo não normalizados
    # Retorna os logits para passar na funcao softmax
    def forward(self, x, validation_mode = False, img_names_validation=None):

        
      if validation_mode:
        global image_names_dict
      
        image_names_dict.clear()
        
        for i, name in enumerate(img_names_validation):
          image_names_dict[i] = name
        
      logits = self.model(x).logits
      return logits

    # Passo a passo do treinamento
    # Batch -> Lote de img (32 img por batch)
    def training_step(self, batch):

        # Passa as img para os dispositivos GPU/CPU
        images, labels, img_names = batch
        images, labels = images.to(device), labels.to(device)

        global image_names_dict
        
        image_names_dict.clear()
        
        for i, name in enumerate(img_names):
          image_names_dict[i] = name
        
        # Obte os logits passando as imagens através do foward
        logits = self(images) 

        # Calcula a perda
        loss = self.criterion(logits, labels)

        # Retorna a previsão do modelo
        _, predicted = torch.max(logits, 1)

        # Realiza o calcula da acuracia
        accuracy = (predicted == labels).float().mean()

        # Realiza o registro das maetricas com CSVLogger
        self.log('train_loss', loss, prog_bar=True,  batch_size=self.batch_size)
        self.log('train_accuracy', accuracy, prog_bar=True,  batch_size=self.batch_size)

        # Retorna o valor do loss
        return loss

    # Faz a mesma coisa do training_step só que na etapa de validação
    def validation_step(self, batch):
        images, labels, img_names = batch
        images, labels = images.to(device), labels.to(device)
        
        global image_names_dict
        
        image_names_dict.clear()
        
        for i, name in enumerate(img_names):
          image_names_dict[i] = name
        
        logits = self(images) 
        loss = self.criterion(logits, labels)
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

    # Configura o otimizador que é o adam com Learning Rate que passa no (Traning_multiclass)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer