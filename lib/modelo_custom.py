from operator import itemgetter

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTModel

from lib.dynamic_patches import DynamicPatches
from lib.VITEmbeddings import CustomViTEmbeddings
from lib.ViTPatchEmbeddings import CustomVITPatchEmbeddings
from lib.VITSelfAttention import ViTSelfAttentionCustom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModeloCustom(pl.LightningModule):
    def __init__(self, model_data: dict, argumentos):
        super(ModeloCustom, self).__init__()

        (
            num_class,
            learning_rate,
            batch_size,
            patch_size,
            input_size,
            num_patches,
        ) = itemgetter(
            "num_class",
            "learning_rate",
            "batch_size",
            "patch_size",
            "input_size",
            "num_patches",
        )(
            model_data
        )

        self.save_hyperparameters()

        self.num_class = num_class
        self.learning_rate = learning_rate
        self.layer_dropout = nn.Dropout(0.4)
        self.batch_size = batch_size
        self.argumentos = argumentos
        self.patch_size = patch_size
        self.input_size = input_size
        self.num_patches = num_patches
        self.patch_generator = DynamicPatches()

        # Carregar um modelo pre-treinado
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
        model_data["input_size"] = (3, self.input_size[0], self.input_size[1])

        embeddings = {}

        embeddings["CustomPatchEmbeddings"] = CustomVITPatchEmbeddings(
            model_data
        )
        embeddings["DefaultEmbeddings"] = base_model.embeddings
        ViTEmbeddings = CustomViTEmbeddings(
            config=base_model.config, embeddings_dict=embeddings
        )
        self.model.vit.embeddings = ViTEmbeddings
        self.model.vit.embeddings.patch_embeddings = embeddings[
            "CustomPatchEmbeddings"
        ]

        self_attention = [
            (i, layer.attention.attention)
            for i, layer in enumerate(base_model.encoder.layer)
        ]

        self_attention_custom = [
            ViTSelfAttentionCustom(
                config=base_model.config, defaultSelfAttention=x
            )
            for x in self_attention
        ]

        for (i, _), custom_attn in zip(self_attention, self_attention_custom):
            base_model.encoder.layer[i].attention.attention = custom_attn

        if argumentos.projecao == "conv":

            self.model.vit.embeddings.patch_embeddings.projection.weight.data.copy_(
                pretrained_conv_weights
            )
            self.model.vit.embeddings.patch_embeddings.projection.bias.data.copy_(
                pretrained_conv_bias
            )

        # if argumentos.pde == "espiral_position":

        #     print("\n###################################\n")
        #     print("\n REORDENANDO POSITION EMBEDDINS \n")
        #     print("\n###################################\n")

        #     self.model.vit.embeddings.reorder_position_embbeddings(
        #         self.model.vit.embeddings,
        #         load_dict(
        #             "./data/centros_pre_salvos/espiral_sem_sobrepoisicao_INDEX.pkl"
        #         ),
        #     )

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

        # Adicionando Regularização
        # self.model.vit.encoder.layer[1].output.dropout = self.layer_dropout
        # self.model.vit.encoder.layer[2].output.dropout = self.layer_dropout
        # self.model.vit.encoder.layer[10].attention.output.dropout = self.layer_dropout
        # self.model.vit.encoder.layer[11].attention.attention.dropout = self.layer_dropout

        # self.model.classifier = torch.nn.Linear(base_model.config.hidden_size, self.num_class)
        self.model.classifier = nn.Sequential(
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

        for i in range(0, 11):
            self.model.vit.encoder.layer[
                i
            ].attention.attention.current_centers = (image_name, centers_image)

        self.model.vit.embeddings.patch_embeddings.current_centers_image = (
            image_name,
            centers_image,
        )

        self.model.vit.embeddings.current_centers_image_PE = (
            image_name,
            centers_image,
        )

        logits = self.model(
            x,
            interpolate_pos_encoding=False,
        ).logits

        self.model.vit.embeddings.patch_embeddings.current_centers_image = None
        self.model.vit.embeddings.current_centers_image_PE = None

        return logits

    def training_step(self, batch):

        images, labels, image_name, centers_img = batch
        images, labels = images.to(device), labels.to(device)

        logits = self(images, image_name, centers_img)

        loss = self.criterion(logits, labels)

        _, predicted = torch.max(logits, 1)

        accuracy = (predicted == labels).float().mean()

        lr = self.optimizers().param_groups[0]["lr"]

        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size)
        self.log(
            "train_accuracy",
            accuracy,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log("lr", lr, prog_bar=True)

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
