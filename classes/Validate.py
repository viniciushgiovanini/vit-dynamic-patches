from classes.modelo_custom import ModeloCustom
from classes.modelo import Modelo
from classes.modelo_binario import ModeloBin
from classes.modelo_custom import ModeloCustom
import torch
import os
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
import cv2
import shap


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

    files = [f for f in all_files_and_dirs if os.path.isfile(os.path.join(path_paste, f))]

    for i, file in enumerate(files):
      
      
      image_path = f"{path_paste}/{file}"
      image = Image.open(image_path).convert('RGB')

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

      transform = T.Compose([
          T.Resize((224, 224)),
          T.ToTensor(),
          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])

      image_tensor = transform(image).unsqueeze(0)

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
     
  
  def plot_confusion_matrix(self, arrays, labels_name, type_plot):
    
    if type_plot == "sns":
      conf_matrix = np.array(arrays)

      plt.figure(figsize=(18, 10))
      sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels_name, yticklabels=labels_name)
      plt.xlabel('Predicted')
      plt.ylabel('True')
      plt.title('Confusion Matrix')
      plt.show()
    if(type_plot == "scikit"):
      conf_matrix = np.array(arrays)
      plt.figure(figsize=(8, 6))
      disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels_name)
      disp.plot()
      plt.title('Confusion Matrix')
      plt.show()
    
  def generate_confusion_matrix(self, main_path, labels, type_plot):
    all_folders = os.listdir(main_path)
    
    resultados = {folder: {predicts: 0 for predicts in all_folders} for folder in all_folders}
    
    for each_folder in all_folders:
      
        lista_imagens_each_class = self._listar_images(os.path.join(main_path, each_folder))
        
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
    
    self.plot_confusion_matrix(arrays=full_list, labels_name=all_folders, type_plot=type_plot)
    
    
  def run_grad_cam(self, image_dir):
    image_paths = [os.path.join(image_dir, img_name) for img_name in os.listdir(image_dir) if img_name.endswith('.png')]
    num_images = len(image_paths)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))

    target_layers = [self.model.model.vit.embeddings.patch_embeddings.projection]
    gradcam = GradCAM(model=self.model, target_layers=target_layers)
    
    for i, image_path in enumerate(image_paths):
        rgb_img = cv2.imread(image_path)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        targets = None
        grayscale_cam = gradcam(input_tensor=input_tensor, targets=targets)

        cam_image = show_cam_on_image(rgb_img, grayscale_cam[0, :], use_rgb=True)
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title('Imagem Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(cam_image)
        axes[i, 1].set_title('Grad-CAM')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()
    
  
  def shapley_model_predict(self, images):
    with torch.no_grad():
        images = torch.tensor(images).float()
        images = images.permute(0, 3, 1, 2) 
        return self.model(images).cpu().numpy()
    
  def run_shapley(self, image_dir,class_names, img_tam=(224,224)):
    image_paths = [os.path.join(image_dir, img_name) for img_name in os.listdir(image_dir) if img_name.endswith('.png')]
    num_images = len(image_paths)
    
    class_names_list = list(class_names.keys())
    
    for each_image_path in image_paths:
        # Abrir e preparar a imagem
        image = Image.open(each_image_path).convert('RGB')
        image = image.resize(img_tam)
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)

        # Fazer a previsão
        predictions = self.shapley_model_predict(image_array)
        predicted_class_index = np.argmax(predictions)  # Índice da classe prevista
        predicted_class_name = class_names_list[predicted_class_index]  # Nome da classe prevista

        print(f'Previsão: {predicted_class_name}')
        # Criar o masker
        masker = shap.maskers.Image("inpaint_telea", image_array[0].shape)

        # Criar o explainer
        explainer = shap.Explainer(self.shapley_model_predict, masker, output_names=class_names_list)

        # Calcular os valores SHAP
        shap_values = explainer(image_array)

        # Plotar os resultados
        plt.figure(figsize=(20, num_images * 5))
        shap.image_plot(shap_values, image_array)

        plt.show()