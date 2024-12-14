import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pickle


class CustomConv2D(nn.Module):
    def __init__(self, input_channels, output_channels, patch_size):
        super(CustomConv2D, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels,
                              kernel_size=patch_size[0], stride=patch_size[0],)
        self.patch_size = patch_size

    def load_dict(self, path):
        with open(path, 'rb') as f:
            lista_centro_dict = pickle.load(f)
            return lista_centro_dict

    def forward(self, x, image_name):
        # x: Tensor de entrada (batch_size, channels, height, width)
        # centers: lista de coordenadas (h, w) para os centros dos patches
        print(x.size())
        batch_size, channels, height, width = x.size()

        lista_centro_dict = self.load_dict(
            "data/centros_pre_salvos/segmentacao_dicionario.pkl")

        each_image = {}

        for b in range(batch_size):
            # Seleciona os centros de acordo com o metodo escolhido

            image_key = image_name[b]

            centers = lista_centro_dict[image_key]

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

                    # torch.Size([3, 16, 16])
                    print(f'Shape antes da conv: {patch.shape}')

                    # torch.Size([768, 1, 1])
                    output_patches = self.conv(patch)
                    print(f'Shape depois da conv: {output_patches.shape}')

                    # fazendo o flatten
                    patches.append(output_patches.view(-1))

                    # patches.append(output_patches)
                else:
                    print(
                        f"Patch fora dos limites: start_h={start_h}, end_h={end_h}, start_w={start_w}, end_w={end_w}")
            # a = torch.stack(patches)
            # each_image[image_name[b]] = torch.stack(patches)

            each_image[image_key] = torch.stack(patches)
            # torch.Size([196, 768])
            print(
                f"Shape do tensor da imagem {image_name[b]} é {each_image[image_name[b]].shape}")
            # torch.Size([768])
            print(f"Shape de um patch: {patches[0].shape}")
            # torch.Size([196, 768])
            print(
                f"Shape de todos os patches empilhados: {each_image[image_key].shape}")
            str("")

        # torch.Size([32, 196, 768])
        # Shape: [batch_size, num_patches, output_channels]
        all_images = torch.stack(list(each_image.values()))
        print(f"Forma final das imagens: {all_images.shape}")
        return all_images


def load_image(image_path):
    # Abre a imagem usando PIL
    image = Image.open(image_path).convert(
        "RGB")  # Converte para RGB, se necessário

    # Define as transformações que você deseja aplicar
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensiona a imagem para 224x224
        transforms.ToTensor(),           # Converte a imagem em um tensor
    ])

    # Aplica as transformações
    image_tensor = transform(image)

    return image_tensor


class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        path = self.imgs[index][0]

        image_name = os.path.basename(path)

        return img, target, image_name


# Exemplo de uso
if __name__ == '__main__':

    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToTensor(),
    ])

    dataset = CustomImageFolder(
        root="data/base_treinamento/train", transform=transform)
    dataloaderr = DataLoader(dataset, batch_size=32,
                             shuffle=True, num_workers=0)

    custom_conv = CustomConv2D(
        input_channels=3, output_channels=768, patch_size=(16, 16))

    for batch_idx, batch in enumerate(dataloaderr):

        tensor_iamges, labels, img_name = batch

        output_patches = custom_conv(tensor_iamges, img_name)
        if output_patches is not None:
            print(output_patches.shape)
