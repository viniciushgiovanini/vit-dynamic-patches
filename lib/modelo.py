import math
from operator import itemgetter

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Modelo(pl.LightningModule):
    def __init__(self, model_data: dict, argumentos):
        super(Modelo, self).__init__()

        (num_class, learning_rate) = itemgetter(
            "num_class",
            "learning_rate",
        )(model_data)

        self.save_hyperparameters()

        self.num_class = num_class
        self.learning_rate = learning_rate
        # self.layer_dropout = nn.Dropout(0.4)

        # base_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        base_model = ViTModel.from_pretrained("WinKawaks/vit-small-patch16-224")
        # base_model = ViTModel.from_pretrained('google/vit-large-patch16-224')
        # base_model = ViTModel.from_pretrained('WinKawaks/vit-tiny-patch16-224')
        # base_model = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')

        if argumentos.model == "small16":
            base_model = ViTModel.from_pretrained(
                "WinKawaks/vit-small-patch16-224"
            )
        elif argumentos.model == "base16":
            base_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        elif argumentos.model == "tiny16":
            base_model = ViTModel.from_pretrained(
                "WinKawaks/vit-tiny-patch16-224"
            )
        elif argumentos.model == "base32":
            base_model = ViTModel.from_pretrained(
                "google/vit-base-patch32-224-in21k"
            )

        self.model = ViTForImageClassification(config=base_model.config)
        self.model.vit = base_model
        print(self.model)
        self.model.to(device)
        print(
            "----------------------------------------------------------------"
        )

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True

        # for name, param in self.model.named_parameters():
        #     if any(
        #         layer_name in name
        #         for layer_name in [
        #             "vit.embeddings",
        #             "vit.encoder.layer.8.",
        #             "vit.encoder.layer.9.",
        #             "vit.encoder.layer.10.",
        #             "vit.encoder.layer.11.",
        #             "vit.head",
        #         ]
        #     ):
        #         param.requires_grad = True

        # self.model.vit.encoder.layer[1].output.dropout = self.layer_dropout
        # self.model.vit.encoder.layer[2].output.dropout = self.layer_dropout
        # self.model.vit.encoder.layer[10].attention.output.dropout = self.layer_dropout
        # self.model.vit.encoder.layer[11].attention.attention.dropout = self.layer_dropout

        # self.model.classifier = torch.nn.Linear(base_model.config.hidden_size, self.num_class)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.num_class),
        )
        # self.model.classifier = nn.Sequential(
        #     nn.Linear(self.model.config.hidden_size, 16),
        #     nn.ReLU(),
        #     self.layer_dropout,
        #     nn.Linear(16, self.num_class)
        # )

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Layer {name} is trainable")

        self.criterion = nn.CrossEntropyLoss()
        print(
            "----------------------------------------------------------------"
        )
        print(self.model)

    # Passagem para frente (Backpropagation) retorna os valores finais do modelo não normalizados
    # Retorna os logits para passar na funcao softmax
    def forward(self, x):
        logits = self.model(x, interpolate_pos_encoding=False).logits
        return logits

    # Batch -> Lote de img (32 img por batch)
    def training_step(self, batch):

        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        logits = self(images)

        loss = self.criterion(logits, labels)

        _, predicted = torch.max(logits, 1)

        accuracy = (predicted == labels).float().mean()

        lr = self.optimizers().param_groups[0]["lr"]

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        self.log("lr", lr, prog_bar=True)

        return loss

    def validation_step(self, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        logits = self(images)
        loss = self.criterion(logits, labels)
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            # weight_decay=1e-3,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=0.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
