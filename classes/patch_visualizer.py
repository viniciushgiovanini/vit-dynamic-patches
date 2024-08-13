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
  

  def visualize_patches(self, image, h_indices, w_indices):
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