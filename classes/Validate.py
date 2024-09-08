from classes.modelo_custom import ModeloCustom
from classes.modelo import Modelo
from classes.modelo_binario import ModeloBin
from classes.modelo import Modelo
from classes.modelo_custom import ModeloCustom
import torch
import os
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm



class Validate:
  def __init__(self, num_class, learning_rate, model_name):
    self.num_class = num_class
    self.learning_rate = learning_rate
    self.model_name = model_name

    if model_name == "binario":
      self.model = ModeloBin(self.num_class, self.learning_rate)
    elif model_name == "multiclass":
      self.model = Modelo(self.num_class, self.learning_rate)
    elif model_name == "custom":
      self.model = ModeloCustom(self.num_class, self.learning_rate)
      
  
    
  
  def load_default_model(self, path_model, map_location="cpu"):
    self.model.load_state_dict(torch.load(path_model, map_location=map_location))
    self.model.eval()  
    
  def load_checkpoint_model(self, path_model):
    
    if self.model_name == "binario":
      self.model =  ModeloBin.load_from_checkpoint(path_model)
    elif self.model_name == "multiclass":
      self.model =  Modelo.load_from_checkpoint(path_model)
    elif self.model_name == "custom":
      self.model =  ModeloCustom.load_from_checkpoint(path_model)
  
  def _get_key_from_value(self, dicte, target_value):
    for key, value in dicte.items():
        if value == target_value:
            return key
    return None
  
  def validate_show(self,path_paste, dicionario_labels, qtd_exibicao=5):
    all_files_and_dirs = os.listdir(path_paste)

    # Filtra apenas os arquivos
    files = [f for f in all_files_and_dirs if os.path.isfile(os.path.join(path_paste, f))]

    for i, file in enumerate(files):
      
      
      image_path = f"{path_paste}/{file}"
      image = Image.open(image_path).convert('RGB')

      # Transforme a imagem
      transform = T.Compose([
          T.Resize((224, 224)),
          T.ToTensor(),
          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])

      image_tensor = transform(image).unsqueeze(0) 

      self.model.eval()

      if self.model_name == "multiclass":
        with torch.no_grad():
          output = self.model(image_tensor)
          
          probabilities = torch.softmax(output, dim=1)
          
          prediction = torch.argmax(probabilities, dim=1).item()
     
      elif(self.model_name == "binario"):
        with torch.no_grad():
          output = self.model(image_tensor)

          probabilities = torch.sigmoid(output)

          prediction = (probabilities > 0.5).int().item() 
      
      plt.imshow(image)
      plt.axis('off')
      
      retorno = self._get_key_from_value(dicte=dicionario_labels, target_value=prediction)
      
      plt.title(f'Predicted class: {retorno}')
      plt.show()
      
      if i == qtd_exibicao:
        break
      
  def _validate_qtd(self, files, resultado, dicionario_labels, ):

    for i, file in tqdm(enumerate(files), desc=f"Processando imagens...", unit=" Imagens"):
          
      image = Image.open(file).convert('RGB')

      # Transforme a imagem
      transform = T.Compose([
          T.Resize((224, 224)),
          T.ToTensor(),
          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])

      image_tensor = transform(image).unsqueeze(0)

      # Certifique-se de que o modelo está em modo de avaliação
      self.model.eval()

      if self.model_name  == "multiclass":
        with torch.no_grad():
          output = self.model(image_tensor)
          
          probabilities = torch.softmax(output, dim=1)
          
          prediction = torch.argmax(probabilities, dim=1).item()
     
      elif(self.model_name  == "binario"):
        with torch.no_grad():
          output = self.model(image_tensor)

          probabilities = torch.sigmoid(output)

          prediction = (probabilities > 0.5).int().item() 
      
      retorno = self._get_key_from_value(dicte=dicionario_labels, target_value=prediction)
      
      resultado[retorno] +=1
        
    return resultado
  
  def _listar_images(self, path):
    all_image = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    all_images_path_complete = []    
    for each_img in all_image:
        all_images_path_complete.append(path+ "/" + each_img)
    
    return all_images_path_complete  
     
  
  def plot_confusion_matrix(self, arrays, labels_name):
    conf_matrix = np.array(arrays)

    # Plotar a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels_name, yticklabels=labels_name)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
  def generate_confusion_matrix(self, main_path, labels):
    all_folders = os.listdir(main_path)
    
    resultados = {folder: {predicts: 0 for predicts in all_folders} for folder in all_folders}
    
    # Para cada pasta (por exemplo: gato)
    for each_folder in all_folders:
        # Gera as combinações das outras pastas (exemplo: [cachorro, passarinho])
        outras_pastas = [folder for folder in all_folders]

        lista_imagens_each_class = self._listar_images(os.path.join(main_path, each_folder))
        
            # Roda a validação para each_folder com resposta sendo outra_pasta
        resultados[each_folder] = self._validate_qtd(
              files=lista_imagens_each_class,
              resultado=resultados[each_folder],              
              dicionario_labels=labels,
              
          )
        

    full_list = []
    
    for each in resultados.values():
      
      merge_list = []
      
      for each_predict in each.values():
        merge_list.append(each_predict)
        
      full_list.append(merge_list)
    
    self.plot_confusion_matrix(arrays=full_list, labels_name=all_folders)