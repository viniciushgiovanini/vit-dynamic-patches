import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTModel
import pytorch_lightning as pl

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
        # cont  = 0
        # for name, param in self.model.named_parameters():
        #   if 'encoder.layer' in name and ('intermediate.dense' in name or 'output.dense' in name):
        #   # if 'encoder.layer' in name and ('output.dense' in name):
        #     if cont < 12:
        #       param.requires_grad = True
        #       cont += 1
        # print("Quantidade de Camadas do MLP: " + str(cont))
              
        # self.model.classifier = torch.nn.Linear(base_model.config.hidden_size, self.num_class)
        
        
        # TREINAR ESSE ASSIM COM 100 EPOCAS
        
        
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.7),  # Aumentar o dropout
            nn.Linear(16, 1)
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
    def training_step(self, batch):

        # Passa as img para os dispositivos GPU/CPU
        images, labels = batch
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        # Obte os logits passando as imagens através do foward 
        logits = self(images)
        
        # Calcula a perda
        loss = self.criterion(logits, labels)
        
        # Retorna a previsão do modelo
        predicted = torch.sigmoid(logits) > 0.5
        
        # Realiza o calcula da acuracia
        accuracy = (predicted == labels).float().mean()
        
        # Realiza o registro das maetricas com CSVLogger
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy, prog_bar=True)
        
        # Retorna o valor do loss
        return loss

    # Faz a mesma coisa do training_step só que na etapa de validação
    def validation_step(self, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        logits = self(images)
        loss = self.criterion(logits, labels)
        predicted = torch.sigmoid(logits) > 0.5
        accuracy = (predicted == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)

    # Configura o otimizador que é o adam com Learning Rate que passa no (Traning_multiclass)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.001)
        return optimizer