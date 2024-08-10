import torch
import torch.nn as nn
from transformers import ViTForImageClassification
import pytorch_lightning as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Modelo(pl.LightningModule):
    def __init__(self, num_class, learning_rate):
        super(Modelo, self).__init__()
        
        
        self.num_class = num_class
        self.learning_rate = learning_rate
      
        # Carregar um modelo pré-treinado
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=self.num_class, ignore_mismatched_sizes=True)
        # self.model = ViTForImageClassification.from_pretrained('amunchet/rorshark-vit-base', num_labels=self.num_class, ignore_mismatched_sizes=True)
        # self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch32-224-in21k', num_labels=self.num_class, ignore_mismatched_sizes=True)
        # self.model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224', num_labels=self.num_class, ignore_mismatched_sizes=True)
        print(self.model)
        self.model.to(device)
        print("----------------------------------------------------------------")
        for param in self.model.parameters():
            param.requires_grad = False
            
        for param in self.model.classifier.parameters():
            param.requires_grad = True
            
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