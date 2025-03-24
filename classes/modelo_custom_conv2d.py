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
    def __init__(self, input_size, patch_size, embed_dim, num_patches, is_visualizer, argumentos):
        super(CustomPatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.is_visualizer = is_visualizer

        # self.projection = nn.Linear(patch_size[0] * patch_size[1] * input_size[0], embed_dim)

        self.projection = nn.Conv2d(
            in_channels=input_size[0],
            out_channels=embed_dim,
            kernel_size=patch_size[0],
            stride=patch_size[0]
        )

        self.visualizer = PatchVisualizer(patch_size)

        self.abordagem_selecionada = ""

        print('#####################################')
        if argumentos.pde == "ra":
            self.dict_center = self.load_dict(
                './data/centros_pre_salvos/randomico_melhorado_identificador_por_imgname.pkl')
            print("Abordagem selecionada: Randomica Aprimorado")
        elif argumentos.pde == "ss":
            self.dict_center = self.load_dict(
                './data/centros_pre_salvos/segmentacao_dicionario.pkl')
            print("Abordagem selecionada: Seleção por Segmentação")
        elif argumentos.pde == "grid":
            print("Abordagem selecionada: Grid")
        elif argumentos.pde == "sr":
            print("Abordagem selecionada: Seleção Randomica")
        elif argumentos.pde == "zigzag":
            self.centers_zigzag_espiral = self.load_dict(
                "./data/centros_pre_salvos/zigzag_centers.pkl")
            print("Abordagem selecionada: Seleção por ZigZag")
        elif argumentos.pde == "espiral":
            print("Abordagem selecionada: Seleção por Espiral")
            self.centers_zigzag_espiral = self.load_dict(
                "./data/centros_pre_salvos/espiral_centers.pkl")
        self.abordagem_selecionada = argumentos.pde
        print('#####################################')

        # Arquivo dos centros randomicos melhorados
        # self.dict_center = self.load_dict('./data/centros_pre_salvos/randomico_melhorado_identificador_por_imgname.pkl')

        # Arquivo dos centros segmentados
        # self.dict_center = self.load_dict('./data/centros_pre_salvos/segmentacao_dicionario.pkl')

    def load_dict(self, path):
        with open(path, 'rb') as f:
            lista_centro_dict = pickle.load(f)

        return lista_centro_dict

    def forward(self, x, **kwargs):

        batch_size, channels, height, width = x.size()

        each_image = {}

        for b in range(batch_size):

            ############################################################
            #             Caso use aboradagem SR                       #
            ############################################################
            if self.abordagem_selecionada == "grid":
                centers = DynamicPatches().generate_patch_centers(height, width, self.patch_size)

            if self.abordagem_selecionada == "sr":
                centers = DynamicPatches().generate_random_patch_centers(
                    height, width, self.patch_size, self.num_patches)

            ############################################################
            #             Caso use aboradagem SS e RA                  #
            ############################################################
            if self.abordagem_selecionada != "grid" and self.abordagem_selecionada != "sr" and self.abordagem_selecionada != "zigzag" and self.abordagem_selecionada != "espiral":
                try:
                    centers = self.dict_center[image_names_dict[b]]
                except:
                    print(
                        f"Erro ao encontrar centro --> {image_names_dict[b]}")

            ############################################################
            #             Caso use aboradagem Zigzag e Espiral         #
            ############################################################
            if self.abordagem_selecionada == "espiral" or self.abordagem_selecionada == "zigzag":
                try:
                    centers = self.centers_zigzag_espiral
                except:
                    print(
                        'Erro ao ler a lista de centros do metodo espiral ou zigzag !!!')

            h_indices = [int(h) for h, _ in centers]
            w_indices = [int(w) for _, w in centers]

            patches = []

            for (h_idx, w_idx) in zip(h_indices, w_indices):

                start_h = h_idx - self.patch_size[0] // 2
                start_w = w_idx - self.patch_size[1] // 2

                end_h = start_h + self.patch_size[0]
                end_w = start_w + self.patch_size[1]

                if (0 <= start_h and start_h + self.patch_size[0] <= height and
                        0 <= start_w and start_w + self.patch_size[1] <= width):

                    # Extrair o patch
                    patch = x[b, :, start_h:end_h, start_w:end_w]
                    output_patches = self.projection(patch)

                    patches.append(output_patches.view(-1))

                else:
                    print(
                        f"Patch fora dos limites: start_h={start_h}, end_h={end_h}, start_w={start_w}, end_w={end_w}")

            each_image[image_names_dict[b]] = torch.stack(patches)

        # Shape: [batch_size, num_patches, output_channels]
        all_images = torch.stack(list(each_image.values()))
        return all_images


class ModeloCustomConv2d(pl.LightningModule):
    def __init__(self, num_class, learning_rate, num_patch, input_size, patch_size, batch_size, argumentos):
        super(ModeloCustomConv2d, self).__init__()

        self.save_hyperparameters()

        self.num_class = num_class
        self.learning_rate = learning_rate
        self.layer_dropout = nn.Dropout(0.4)
        self.batch_size = batch_size

        # Carregar um modelo pré-treinado
        # base_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        base_model = ViTModel.from_pretrained(
            'WinKawaks/vit-small-patch16-224')
        # base_model = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')

        if argumentos.model == "small16":
            base_model = ViTModel.from_pretrained(
                'WinKawaks/vit-small-patch16-224')
        elif argumentos.model == "base16":
            base_model = ViTModel.from_pretrained(
                'google/vit-base-patch16-224')
        elif argumentos.model == "tiny16":
            base_model = ViTModel.from_pretrained(
                'WinKawaks/vit-tiny-patch16-224')
        elif argumentos.model == "base32":
            base_model = ViTModel.from_pretrained(
                'google/vit-base-patch32-224-in21k')

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
            argumentos=argumentos
        )

        self.model.vit.embeddings.patch_embeddings.projection.weight.data.copy_(
            pretrained_conv_weights)
        self.model.vit.embeddings.patch_embeddings.projection.bias.data.copy_(
            pretrained_conv_bias)

        print(self.model)
        self.model.to(device)
        print("----------------------------------------------------------------")

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # Descongelar as camadas especificas
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in [
                "vit.embeddings.patch_embeddings.projection",
                "vit.encoder.layer.1.",
                "vit.encoder.layer.2.",
                "vit.encoder.layer.9.",
                "vit.encoder.layer.10.",
                "vit.encoder.layer.11.",
                "vit.layernorm",
                "vit.pooler"
            ]):
                param.requires_grad = True

        self.model.vit.encoder.layer[1].output.dropout = self.layer_dropout
        self.model.vit.encoder.layer[2].output.dropout = self.layer_dropout
        # self.model.vit.encoder.layer[10].attention.output.dropout = self.layer_dropout
        # self.model.vit.encoder.layer[11].attention.attention.dropout = self.layer_dropout

        # self.model.classifier = torch.nn.Linear(base_model.config.hidden_size, self.num_class)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size,
                      self.model.config.hidden_size),
            nn.ReLU(),
            self.layer_dropout,
            nn.Linear(self.model.config.hidden_size,
                      self.model.config.hidden_size),
            nn.ReLU(),
            self.layer_dropout,
            nn.Linear(self.model.config.hidden_size, self.num_class)
        )

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Layer {name} is trainable")

        self.criterion = nn.CrossEntropyLoss()
        print("----------------------------------------------------------------")
        print(self.model)

    def forward(self, x, validation_mode=False, img_names_validation=None):

        if validation_mode:
            global image_names_dict

            image_names_dict.clear()

            for i, name in enumerate(img_names_validation):
                image_names_dict[i] = name

        logits = self.model(x).logits
        return logits

    def training_step(self, batch):

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

        self.log('train_loss', loss, prog_bar=True,
                 batch_size=self.batch_size)
        self.log('train_accuracy', accuracy, prog_bar=True,
                 batch_size=self.batch_size)

        return loss

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
        self.log('val_accuracy', accuracy, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
