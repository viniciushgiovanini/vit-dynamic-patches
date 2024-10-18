import random
import numpy as np
import cv2

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
          
          
          check = self.verificar_adj(img_mtx, w, h, centers)
          
          while check:
              h = random.uniform(patch_height / 2, image_height - patch_height / 2)
              w = random.uniform(patch_width / 2, image_width - patch_width / 2)
              check = self.verificar_adj(img_mtx, w, h, centers)
          
          centers.append((h, w))
      
      return centers
    
  ####################################
  # Patches segmentados
  ####################################
    
  def grabcutextractcenters(self, path_img, tamanho_img=(224, 224), stride=16):
  
  
    qtd_patches = int((tamanho_img[0]/stride) * (tamanho_img[0]/stride))
    
    imagem = cv2.imread(path_img)
    
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    
    
    _, mask, _ = self.remover_fundo_com_grabcut_recortado(imagem=imagem)
    
    if self.is_image_black_percentage(mask):
      
      altura, largura, _ = imagem.shape
      
      centers_merge = self.generate_patch_centers(altura, largura, stride)
      
      return centers_merge
    else:
      mask = cv2.resize(mask, tamanho_img)
      
      centers_randomicos = []
      centers_stride = []
      centers_merge = []
      altura, largura = mask.shape

      for i in range(altura):  
          for j in range(largura):
              pixel = mask[i, j]
              
              if len(pixel.shape) == 0:
                  if pixel == 255:  
                        centers_randomicos.append((i, j))
              else:  
                  if np.array_equal(pixel, [255, 255, 255]): 
                        centers_randomicos.append((i, j))
      
      
      for i in range(0, altura, stride):
            for j in range(0, largura, stride):
                pixel = mask[i, j]
                
                if len(pixel.shape) == 0: 
                    if pixel == 255:  
                        centers_stride.append((i, j))
                else:  
                    if np.array_equal(pixel, [255, 255, 255]):
                        centers_stride.append((i, j))
                        
      
      quantidade_patches_stride = len(centers_stride)
      
      if quantidade_patches_stride < qtd_patches:
        diferentes_lista1 = set(centers_randomicos) - set(centers_stride)
        diferentes_lista2 = set(centers_stride) - set(centers_randomicos)

        resultado = list(diferentes_lista1) + list(diferentes_lista2)
        
        random.shuffle(resultado)
        qtd_faltante_patches = qtd_patches - quantidade_patches_stride
        
        centers_merge = centers_stride.copy()
        centers_merge.extend(resultado[0:qtd_faltante_patches])
      elif quantidade_patches_stride == qtd_patches:
        centers_merge = centers_stride.copy()
        
      return centers_merge
    
  def remover_fundo_com_grabcut_recortado(self, imagem):
      mascara = np.zeros(imagem.shape[:2], np.uint8)
      backgroundModel = np.zeros((1, 65), np.float64)
      foregroundModel = np.zeros((1, 65), np.float64)
      altura, largura = imagem.shape[:2]
      
      x1 = 0
      y1 = 0
      x2 = largura - 1
      y2 = altura - 1
          
      rectangle = (x1, y1, x2 - x1, y2 - y1)
      
      cv2.grabCut(imagem, mascara, rectangle,  
              backgroundModel, foregroundModel,
              3, cv2.GC_INIT_WITH_RECT)
      
      mascara_objeto = np.where((mascara == 2) | (mascara == 0), 0, 1).astype('uint8')
      
      imagem_sem_fundo = imagem * mascara_objeto[:, :, np.newaxis]
      
      img_recortada = imagem_sem_fundo[y1:y2, x1:x2]
      
      imagem_gray = cv2.cvtColor(img_recortada, cv2.COLOR_BGR2GRAY)
      
      _, mascara = cv2.threshold(imagem_gray, 10, 255, cv2.THRESH_BINARY)
      
      img_original_recortada = imagem[y1:y2, x1:x2]
      
      return img_recortada, mascara, img_original_recortada
    
  def is_image_black_percentage(self, image, threshold=0.9):
      total_pixels = image.size
      
      if image.ndim == 2:
          black_pixels = np.sum(image == 0)
      elif image.ndim == 3: 
          black_pixels = np.sum(np.all(image == 0, axis=-1))

      black_percentage = black_pixels / total_pixels
      return black_percentage >= threshold