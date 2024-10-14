import random
import numpy as np

class DynamicPatches:
  
  
  #############################
  # Patches default 
  ############################# 
  def generate_patch_centers(self, image_height, image_width, patch_size):
        stride = patch_size[0]  
        
        num_patches_h = image_height // stride
        num_patches_w = image_width // stride
        
        centers_h = []
        centers_w = []
        
        for i in range(num_patches_h):
          centers_h.append((i * stride + stride // 2))
        
        for j in range(num_patches_w):
          centers_w.append((j * stride + stride // 2))
        
        centers = []

        for h in centers_h:
          for w in centers_w:
            centers.append((h,w))
        
        return centers
  
  
  ####################################
  # Random Pixels default
  ####################################
  def generate_random_patch_centers(self, image_height, image_width, patch_size, num_patches):
      patch_height, patch_width = patch_size
      
      centers = []
      for _ in range(num_patches):
          h = random.uniform(patch_height / 2, image_height - patch_height / 2)
          w = random.uniform(patch_width / 2, image_width - patch_width / 2)
          centers.append((h, w))

      return centers
    
  ####################################
  # Random Pixels Melhorado
  ####################################
  def pixels_adj(self, matriz, x, y, n_voltas):
    coords_voltas = []
    n_linhas, n_colunas = matriz.shape
    
    for volta in range(1, n_voltas + 1):
        for i in range(-volta, volta + 1):
            if 0 <= y + i < n_linhas:
                if 0 <= x - volta < n_colunas:
                    coords_voltas.append((y + i, x - volta))
                if 0 <= x + volta < n_colunas:
                    coords_voltas.append((y + i, x + volta))
        
        for j in range(-volta + 1, volta):
            if 0 <= x + j < n_colunas:
                if 0 <= y - volta < n_linhas:
                    coords_voltas.append((y - volta, x + j))
                if 0 <= y + volta < n_linhas:
                    coords_voltas.append((y + volta, x + j))
    coords_voltas.append((y,x))
    return coords_voltas

def verificar_adj(self,matriz, x, y, lista_centros):
    x, y = int(round(x)), int(round(y))
    if len(lista_centros) == 0:
        return False
    else:
        for each in lista_centros:
            each_x, each_y = int(round(each[1])), int(round(each[0]))
            ret = self.pixels_adj(matriz=matriz, x=each_x, y=each_y, n_voltas=8)
            if (y, x) in ret:  
                return True
    return False

def random_patchs_melhorados(self, image_height, image_width, patch_size, num_patches, img_PIL):
    patch_height, patch_width = patch_size
    
    img_gray = img_PIL.convert('L')
    img_mtx = np.array(img_gray)
    
    centers = []
    
    for _ in range(num_patches):
        h = random.uniform(patch_height / 2, image_height - patch_height / 2)
        w = random.uniform(patch_width / 2, image_width - patch_width / 2)
        
        
        check = verificar_adj(img_mtx, w, h, centers)
        
        while check:
            h = random.uniform(patch_height / 2, image_height - patch_height / 2)
            w = random.uniform(patch_width / 2, image_width - patch_width / 2)
            check = verificar_adj(img_mtx, w, h, centers)
        
        centers.append((h, w))
    
    return centers