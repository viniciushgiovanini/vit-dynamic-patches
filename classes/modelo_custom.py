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

        # Projeta os patches para uma dimensão `embed_dim`
        self.projection = nn.Linear(
            patch_size[0] * patch_size[1] * input_size[0], embed_dim)
        # Lida com a visualizacao de patches
        # self.visualizer = PatchVisualizer(patch_size)

        #########################################################################
        # Descomente a linha que será carregado o centros de determinados métodos
        #########################################################################
        self.abordagem_selecionada = ""

        print('#####################################')
        if argumentos.pde == "ra":
            self.dict_center = self.load_dict(
                './data/centros_pre_salvos/randomico_melhorado_identificador_por_imgname.pkl')
            self.abordagem_selecionada = argumentos.pde
            print("Abordagem selecionada: Randomica Aprimorado")
        elif argumentos.pde == "ss":
            self.dict_center = self.load_dict(
                './data/centros_pre_salvos/segmentacao_dicionario.pkl')
            self.abordagem_selecionada = argumentos.pde
            print("Abordagem selecionada: Seleção por Segmentação")
        elif argumentos.pde == "grid":
            self.abordagem_selecionada = argumentos.pde
            print("Abordagem selecionada: Grid")
        elif argumentos.pde == "sr":
            self.abordagem_selecionada = argumentos.pde
            print("Abordagem selecionada: Seleção Randomica")
        print('#####################################')

        # Arquivo dos centros randomicos melhorados
        # self.dict_center = self.load_dict('./data/centros_pre_salvos/randomico_melhorado_identificador_por_imgname.pkl')

        # Arquivo dos centros segmentados
        # self.dict_center = self.load_dict('./data/centros_pre_salvos/segmentacao_dicionario.pkl')

        # Arquivo dos centros segmentados 32x32 de patch_size
        # self.dict_center = self.load_dict('./data/centros_pre_salvos/segmentacao_dicionario_32_patch_size.pkl')

    def load_dict(self, path):
        with open(path, 'rb') as f:
            lista_centro_dict = pickle.load(f)

        return lista_centro_dict

    def forward(self, x, **kwargs):

        # X -> Tensor de entrada (batch_size, channels, height, width)

        batch_size, channels, height, width = x.size()

        # armazena patches extraidos
        all_patches = []

        # armazena os indices dos centros dos patches
        all_h_indices = []
        all_w_indices = []

        ################################################
        #                   Print de Log               #
        ################################################
        # print("Iniciou um loop de batch\n")
        # print(f"Printando de dentro do CustomPatchEmbedding: {image_names_dict}")

        # Loop sobre cada img do batch
        for b in range(batch_size):

            # Seleciona os centros de acordo com o metodo escolhido

            ############################################################
            #             Caso use aboradagem SR                       #
            ############################################################
            if self.abordagem_selecionada == "grid":
                centers = DynamicPatches().generate_patch_centers(height, width, self.patch_size)

            if self.abordagem_selecionada == "sr":
                centers = DynamicPatches().generate_random_patch_centers(
                    height, width, self.patch_size, self.num_patches)
            # centers = DynamicPatches().random_patchs_melhorados(self.patch_size, self.num_patches, x[b])
            # centers = DynamicPatches().grabcutextractcenters(imagem_tensor=x[b], tamanho_img=(height, width), stride=self.patch_size[0])
            # centers = DynamicPatches().random_patchs_melhorados(self.patch_size, self.num_patches, x[b])

            ############################################################
            #             Caso use aboradagem SS e RA                  #
            ############################################################

            if self.abordagem_selecionada != "grid" and self.abordagem_selecionada != "sr":
                try:
                    centers = self.dict_center[image_names_dict[b]]
                except:
                    print(
                        f"Erro ao encontrar centro --> {image_names_dict[b]}")

            h_indices = [int(h) for h, _ in centers]
            w_indices = [int(w) for _, w in centers]

            patches = []

            # Para cada par ded indices h,w
            for (h_idx, w_idx) in zip(h_indices, w_indices):

                start_h = h_idx - self.patch_size[0] // 2
                start_w = w_idx - self.patch_size[1] // 2

                end_h = start_h + self.patch_size[0]
                end_w = start_w + self.patch_size[1]

                # Verificar se o patch esta dentro dos limites
                if (0 <= start_h and start_h + self.patch_size[0] <= height and
                        0 <= start_w and start_w + self.patch_size[1] <= width):

                    # Extrair o patch
                    patch = x[b, :, start_h:end_h, start_w:end_w]
                    patches.append(patch.to(device))
                else:
                    print(
                        f"Patch fora dos limites: start_h={start_h}, end_h={end_h}, start_w={start_w}, end_w={end_w}")

            # se o numero patches for menor que o ncessario, prenche com tesnores vazios
            if len(patches) < self.num_patches:
                print("ERRO: Gerando patch preto\n\n\n\n\n\n\n\n\n")
                missing_patches = self.num_patches - len(patches)
                patches += [torch.zeros(channels, self.patch_size[0],
                                        self.patch_size[1], device=device)] * missing_patches

            ##################################
            # Visualização do Patch Tensor
            ##################################
            # if self.is_visualizer:
                # self.visualizer.visualize_patches_with_tensor(patches)
                # self.visualizer.visualize_patch_centers(x[b], centers, self.patch_size, image_names_dict[b])

            # Concatena os patches em um unico tensor
            patches = torch.stack(patches)

            # faz o flatten
            patches = patches.flatten(start_dim=1)

            # passa na projecao
            patches = self.projection(patches)

            all_patches.append(patches)
            all_h_indices.append(h_indices)
            all_w_indices.append(w_indices)

        # combina todos os patches de todas as imagens no batch em um uico tensor tridimensional (batch_size, num_patches, embed_dim)
        all_patches = torch.stack(all_patches).to(device)

        return all_patches


class ModeloCustom(pl.LightningModule):
    def __init__(self, num_class, learning_rate, num_patch, input_size, patch_size, batch_size, argumentos):
        super(ModeloCustom, self).__init__()

        self.save_hyperparameters()

        self.num_class = num_class
        self.learning_rate = learning_rate
        self.layer_dropout = nn.Dropout(0.4)
        self.batch_size = batch_size

        # Carregar um modelo pré-treinado
        base_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        # base_model = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')
        # base_model = ViTModel.from_pretrained('WinKawaks/vit-small-patch16-224')

        if argumentos.model == "small16":
            base_model = ViTModel.from_pretrained(
                'WinKawaks/vit-small-patch16-224')
        elif argumentos.model == "base16":
            base_model = ViTModel.from_pretrained(
                'google/vit-large-patch16-224')
        elif argumentos.model == "tiny16":
            base_model = ViTModel.from_pretrained(
                'WinKawaks/vit-tiny-patch16-224')
        elif argumentos.model == "base32":
            base_model = ViTModel.from_pretrained(
                'google/vit-base-patch32-224-in21k')

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

        print(self.model)
        self.model.to(device)
        print("----------------------------------------------------------------")

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

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

    # Retorna os logits para passar na funcao softmax

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
