import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTModel
import pytorch_lightning as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Modelo(pl.LightningModule):
    def __init__(self, num_class, learning_rate, argumentos):
        super(Modelo, self).__init__()

        # Salvar os hyperparametros
        self.save_hyperparameters()

        self.num_class = num_class
        self.learning_rate = learning_rate
        self.layer_dropout = nn.Dropout(0.4)

        # Carregar um modelo pré-treinado
        # base_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        base_model = ViTModel.from_pretrained(
            'WinKawaks/vit-small-patch16-224')
        # base_model = ViTModel.from_pretrained('google/vit-large-patch16-224')
        # base_model = ViTModel.from_pretrained('WinKawaks/vit-tiny-patch16-224')
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

        # Descongelar as camadas específicas
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

        # Adicionando Regularização
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
        # self.model.classifier = nn.Sequential(
        #     nn.Linear(self.model.config.hidden_size, 16),
        #     nn.ReLU(),
        #     self.layer_dropout,
        #     nn.Linear(16, self.num_class)
        # )

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
    def forward(self, x):
        logits = self.model(x).logits
        return logits

    # Passo a passo do treinamento
    # Batch -> Lote de img (32 img por batch)
    def training_step(self, batch):

        # Passa as img para os dispositivos GPU/CPU
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # Obte os logits passando as imagens através do foward
        logits = self(images)

        # Calcula a perda
        loss = self.criterion(logits, labels)

        # Retorna a previsão do modelo
        _, predicted = torch.max(logits, 1)

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
        images, labels = images.to(device), labels.to(device)
        logits = self(images)
        loss = self.criterion(logits, labels)
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)

    # Configura o otimizador que é o adam com Learning Rate que passa no (Traning_multiclass)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
