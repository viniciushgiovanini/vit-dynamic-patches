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


image_names_dict = {}


class CustomPatchEmbedding(nn.Module):
    def __init__(self, input_size, patch_size, embed_dim, num_patches, is_visualizer):
        super(CustomPatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.is_visualizer = is_visualizer
        
        # Projeta os patches para uma dimensão `embed_dim`
        self.projection = nn.Linear(patch_size[0] * patch_size[1] * input_size[0], embed_dim)
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
        
        
        
        # Loop sobre cada img do batch
        for b in range(batch_size):
            # Seleciona os centros de acordo com o metodo escolhido
            # centers = DynamicPatches().generate_patch_centers(height, width, self.patch_size)
            # centers = DynamicPatches().generate_random_patch_centers(height, width, self.patch_size, self.num_patches)
            # centers = DynamicPatches().random_patchs_melhorados(self.patch_size, self.num_patches, x[b])
            # centers = DynamicPatches().grabcutextractcenters(imagem_tensor=x[b], tamanho_img=(height, width), stride=self.patch_size[0])
            # centers = DynamicPatches().random_patchs_melhorados(self.patch_size, self.num_patches, x[b])
            
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
                    patches.append(patch)
                    print(f"Shape de cada patch extraido: {patch.shape}")
                else:
                    print(f"Patch fora dos limites: start_h={start_h}, end_h={end_h}, start_w={start_w}, end_w={end_w}")
            
            
            
            # se o numero patches for menor que o ncessário, prenche com tesnores vazios
            if len(patches) < self.num_patches:
                print("ERRO: Gerando patch preto\n\n\n\n\n\n\n\n\n")
                missing_patches = self.num_patches - len(patches)
                patches += [torch.zeros(channels, self.patch_size[0], self.patch_size[1])] * missing_patches

            
            ##################################
            # Visualização do Patch Tensor
            ##################################
            # if self.is_visualizer:
              # self.visualizer.visualize_patches_with_tensor(patches)
              # self.visualizer.visualize_patch_centers(x[b], centers, self.patch_size, image_names_dict[b])              
            
            # Concatena os patches em um unico tensor
            
            # torch.Size([196, 3, 16, 16])
            patches = torch.stack(patches)
            print(f"Shape apos a empilhamento: {patches.shape}")
              

            # torch.Size([196, 768])
            # faz o flatten
            patches = patches.flatten(start_dim=1)
            print(f"Shape apos a flatten: {patches.shape}")
            print(f"Shape apos a flatten: {patches.shape}")
            
            # projeta os patches para um espaço de maior dimensão
            patches = self.projection(patches)    
            print(f"Shape apos a projeção linear: {patches.shape}")
            all_patches.append(patches)
            all_h_indices.append(h_indices)
            all_w_indices.append(w_indices)
        
        
        # combina todos os patches de todas as imagens no batch em um uico tensor tridimensional (batch_size, num_patches, embed_dim)
        
        # torch.Size([32, 196, 768])
        all_patches = torch.stack(all_patches)
        print(f"Shape final: {all_patches.shape}")
                        
        return all_patches


















def load_image(image_path):
    # Abre a imagem usando PIL
    image = Image.open(image_path).convert("RGB")  # Converte para RGB, se necessário

    # Define as transformações que você deseja aplicar
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensiona a imagem para 224x224
        transforms.ToTensor(),           # Converte a imagem em um tensor
    ])

    # Aplica as transformações
    image_tensor = transform(image)

    return image_tensor





from torchvision.datasets import ImageFolder
import os


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


    dataset = CustomImageFolder(root="data/base_treinamento/train", transform=transform)
    dataloaderr = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    
    custom_conv = CustomPatchEmbedding(
            input_size=(3, 224, 224),  
            patch_size=(16,16),       
            embed_dim=768,
            num_patches=196,
            is_visualizer=True,
        )
    
    for batch_idx, batch in enumerate(dataloaderr):
        
        tensor_iamges, labels, img_name = batch
        
        image_names_dict = img_name
        
        output_patches = custom_conv(tensor_iamges)
        if output_patches is not None:
            print(output_patches.shape) 
            
            