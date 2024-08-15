import random

class DynamicPatches:
  
  def generate_random_patch_centers(self, image_height, image_width, patch_size, num_patches):
      patch_height, patch_width = patch_size
      
      centers = []
      for _ in range(num_patches):
          h = random.uniform(patch_height / 2, image_height - patch_height / 2)
          w = random.uniform(patch_width / 2, image_width - patch_width / 2)
          centers.append((h, w))

      return centers
    
    
    
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