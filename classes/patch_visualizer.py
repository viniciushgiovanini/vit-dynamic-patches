import matplotlib.pyplot as plt
import string
import random
import matplotlib.patches as patches
import torch

class PatchVisualizer:
  
  def __init__(self,patch_size):
    self.patch_size = patch_size
  
  def gerar_string_aleatoria(self,tamanho):
    caracteres = string.ascii_letters + string.digits  
    return ''.join(random.choice(caracteres) for _ in range(tamanho))
  
  
  def visualize_patches_with_tensor(self, patch_array):

    num_images = len(patch_array)
    
    print(num_images)
    print("-----------------\n\n\n\n\n\n\n\n\n\n\n\n")
    
    cols = 10
    rows = (num_images + cols - 1) // cols 

    # Criar a figura e os subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # Se axes for uma matriz bidimensional, achatar para iteração fácil
    if rows > 1:
        axes = axes.flatten()

    # Iterar sobre os tensores e plotar
    for i, tensor in enumerate(patch_array):
        # Converter o tensor para NumPy
        tensor_np = tensor.cpu().numpy()
        
        # Transpor para [H, W, C] se necessário
        tensor_np = tensor_np.transpose(1, 2, 0)
        
        # Normalizar a imagem para [0, 1] se necessário
        tensor_np = tensor_np - tensor_np.min()
        tensor_np = tensor_np / tensor_np.max()
        
        # Plotar a imagem
        axes[i].imshow(tensor_np, cmap='gray')
        axes[i].axis('off')  # Remove os eixos

    # Se houver subplots sem imagem, desative-os
    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    # Ajustar o layout
    plt.tight_layout()

    # Salvar a imagem com todas as imagens
    plt.savefig("figs/patches_extraidos.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    # Fechar a figura para liberar memória
    plt.close(fig)
      
  def visualize_patches_with_px(self, image, h_indices, w_indices):
        fig, ax = plt.subplots(1)
        
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
            
            if image.min() < 0 or image.max() > 1:
                image = (image - image.min()) / (image.max() - image.min())  
            
        ax.imshow(image)
        
        patch_height, patch_width = self.patch_size
        
        for (h, w) in zip(h_indices, w_indices):
            rect = patches.Rectangle(
                (w - patch_width // 2, h - patch_height // 2),
                patch_width, patch_height,
                linewidth=1, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
        
        file_name = f"figs/{self.gerar_string_aleatoria(10)}.jpg"
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()