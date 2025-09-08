from operator import itemgetter

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTModel

from lib.VITEmbeddings import CustomViTEmbeddings
from lib.ViTPatchEmbeddings import CustomVITPatchEmbeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModeloCustom(pl.LightningModule):
    def __init__(self, model_data: dict, argumentos):
        super(ModeloCustom, self).__init__()

        num_class, learning_rate, batch_size = itemgetter(
            "num_class", "learning_rate", "batch_size"
        )(model_data)

        self.save_hyperparameters()

        self.num_class = num_class
        self.learning_rate = learning_rate
        self.layer_dropout = nn.Dropout(0.4)
        self.batch_size = batch_size

        # Carregar um modelo pré-treinado
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

        pretrained_conv_weights = (
            base_model.embeddings.patch_embeddings.projection.weight.data.clone()
        )
        pretrained_conv_bias = (
            base_model.embeddings.patch_embeddings.projection.bias.data.clone()
        )

        self.model = ViTForImageClassification(config=base_model.config)
        self.model.vit = base_model

        model_data["embed_dim"] = self.model.config.hidden_size

        input_size = model_data.get("input_size")
        model_data["input_size"] = (3, input_size[0], input_size[1])

        embeddings = {}

        embeddings["CustomViTEmbeddings"] = CustomVITPatchEmbeddings(model_data)
        embeddings["DefaultEmbeddings"] = base_model.embeddings
        ViTEmbeddings = CustomViTEmbeddings(
            config=base_model.config, embeddings_dict=embeddings
        )
        self.model.vit.embeddings = ViTEmbeddings

        if argumentos.projecao == "conv":
            self.model.vit.embeddings.patch_embeddings.projection.weight.data.copy_(
                pretrained_conv_weights
            )
            self.model.vit.embeddings.patch_embeddings.projection.bias.data.copy_(
                pretrained_conv_bias
            )

        print(self.model)
        self.model.to(device)
        print(
            "----------------------------------------------------------------"
        )

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        for name, param in self.model.named_parameters():
            if any(
                layer_name in name
                for layer_name in [
                    "vit.embeddings.patch_embeddings.projection",
                    "vit.encoder.layer.1.",
                    "vit.encoder.layer.2.",
                    "vit.encoder.layer.9.",
                    "vit.encoder.layer.10.",
                    "vit.encoder.layer.11.",
                    "vit.layernorm",
                    "vit.pooler",
                ]
            ):
                param.requires_grad = True

        # Adicionando Regularização
        self.model.vit.encoder.layer[1].output.dropout = self.layer_dropout
        self.model.vit.encoder.layer[2].output.dropout = self.layer_dropout
        # self.model.vit.encoder.layer[10].attention.output.dropout = self.layer_dropout
        # self.model.vit.encoder.layer[11].attention.attention.dropout = self.layer_dropout

        # self.model.classifier = torch.nn.Linear(base_model.config.hidden_size, self.num_class)
        self.model.classifier = nn.Sequential(
            nn.Linear(
                self.model.config.hidden_size, self.model.config.hidden_size
            ),
            nn.ReLU(),
            self.layer_dropout,
            nn.Linear(
                self.model.config.hidden_size, self.model.config.hidden_size
            ),
            nn.ReLU(),
            self.layer_dropout,
            nn.Linear(self.model.config.hidden_size, self.num_class),
        )

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Layer {name} is trainable")

        self.criterion = nn.CrossEntropyLoss()
        print(
            "----------------------------------------------------------------"
        )
        print(self.model)

    def forward(
        self,
        x,
        image_name=None,
        centers_image=None,
    ):
        self.model.vit.embeddings.patch_embeddings.current_centers_image = (
            image_name,
            centers_image,
        )

        logits = self.model(x).logits

        self.model.vit.embeddings.patch_embeddings.current_centers_image = None

        return logits

    def training_step(self, batch):

        images, labels, image_name, centers_img = batch
        images, labels = images.to(device), labels.to(device)

        logits = self(images, image_name, centers_img)

        loss = self.criterion(logits, labels)

        _, predicted = torch.max(logits, 1)

        accuracy = (predicted == labels).float().mean()

        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size)
        self.log(
            "train_accuracy",
            accuracy,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        return loss

    def validation_step(self, batch):
        images, labels, image_name, centers_img = batch
        images, labels = images.to(device), labels.to(device)

        logits = self(images, image_name, centers_img)

        loss = self.criterion(logits, labels)
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)
        self.log(
            "val_accuracy",
            accuracy,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
