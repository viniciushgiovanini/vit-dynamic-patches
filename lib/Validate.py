import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import torch
import torch.serialization
import torchvision.transforms as T
from PIL import Image
from pytorch_grad_cam import EigenCAM, GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from lib.CustomImageFolder import CustomImageFolder
from lib.modelo import Modelo
from lib.modelo_custom import ModeloCustom
from lib.utils import strategy_centers_patch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

g = torch.Generator(device="cpu")
g.manual_seed(seed)

torch.serialization.add_safe_globals([argparse.Namespace])


def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class Validate:
    def __init__(
        self,
        num_class,
        learning_rate,
        num_patch=196,
        input_size=(224, 224),
        patch_size=(16, 16),
        batch_size=32,
        argumentos=None,
        device="cuda",
    ):
        self.num_class = num_class
        self.learning_rate = learning_rate
        self.model_name = argumentos.model_type
        self.num_patch = num_patch
        self.input_size = input_size
        self.patch_size = patch_size
        self.batch_size = batch_size

        self.centers = (
            strategy_centers_patch(argumentos.pde)
            if "pde" in argumentos
            else None
        )
        self.device = device

        self.model_data = {
            "num_class": self.num_class,
            "learning_rate": self.learning_rate,
        }

        if self.model_name == "default":
            self.model = Modelo(self.model_data, argumentos)
        elif self.model_name == "custom":
            self.model_data["num_patches"] = self.num_patch
            self.model_data["input_size"] = self.input_size
            self.model_data["patch_size"] = self.patch_size
            self.model_data["batch_size"] = self.batch_size
            self.model_data["is_visualizer"] = False
            self.model_data["abordagem_selecionada"] = argumentos.pde
            self.model_data["projection_type"] = argumentos.projecao

            self.model = ModeloCustom(
                model_data=self.model_data,
                argumentos=argumentos,
            )

    def load_model_architecture(self, path_model):
        self.model = torch.load(path_model, map_location=self.device)
        self.model.eval()

    def load_default_model(self, path_model):
        self.model.load_state_dict(
            torch.load(path_model, map_location=self.device)
        )
        self.model.eval()

    def load_checkpoint_model(self, path_model):
        if self.model_name == "default":
            self.model = Modelo.load_from_checkpoint(path_model)
        elif self.model_name == "custom":
            self.model = ModeloCustom.load_from_checkpoint(path_model)

    def _get_key_from_value(self, dicte, target_value):
        for key, value in dicte.items():
            if value == target_value:
                return key
        return None

    def _get_centers_from_file(self, file_name):
        center_image = (
            self.centers.get(file_name)
            if isinstance(self.centers, dict)
            else self.centers
        )

        if center_image is None:
            raise Exception(
                f"Não encontrou o centro da respectiva imagem: {file_name}"
            )

        return torch.tensor([center_image])

    def validate_show(self, path_paste, dicionario_labels, qtd_exibicao=5):
        all_files_and_dirs = os.listdir(path_paste)

        files = [
            f
            for f in all_files_and_dirs
            if os.path.isfile(os.path.join(path_paste, f))
        ]

        for i, file in enumerate(files):

            image_path = f"{path_paste}/{file}"
            image = Image.open(image_path).convert("RGB")

            transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                ]
            )

            image_tensor = transform(image).unsqueeze(0).to(self.device)

            self.model.eval()

            if self.model_name == "default":
                with torch.no_grad():
                    output = self.model(image_tensor)

                    probabilities = torch.softmax(output, dim=1)

                    prediction = torch.argmax(probabilities, dim=1).item()

            elif self.model_name == "binario":
                with torch.no_grad():
                    output = self.model(image_tensor)

                    probabilities = torch.sigmoid(output)

                    prediction = (probabilities > 0.5).int().item()
            elif self.model_name == "custom":
                with torch.no_grad():
                    output = self.model(
                        image_tensor,
                        image_name=file,
                        centers_image=self._get_centers_from_file(file),
                    ).to(self.device)
                    probabilities = torch.softmax(output, dim=1)
                    prediction = torch.argmax(probabilities, dim=1).item()

            plt.imshow(image)
            plt.axis("off")

            retorno = self._get_key_from_value(
                dicte=dicionario_labels, target_value=prediction
            )

            plt.title(f"Predicted class: {retorno}")
            plt.show()

            if i == qtd_exibicao:
                break

    def _validate_qtd(
        self,
        files,
        resultado,
        dicionario_labels,
    ):

        for i, file in tqdm(
            enumerate(files), desc=f"Processando imagens...", unit=" Imagens"
        ):

            image = Image.open(file).convert("RGB")

            transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            image_tensor = transform(image).unsqueeze(0).to(self.device)

            self.model.eval()

            if self.model_name == "default":
                with torch.no_grad():
                    output = self.model(image_tensor)

                    probabilities = torch.softmax(output, dim=1)

                    prediction = torch.argmax(probabilities, dim=1).item()

            elif self.model_name == "binario":
                with torch.no_grad():
                    output = self.model(image_tensor)

                    probabilities = torch.sigmoid(output)

                    prediction = (probabilities > 0.5).int().item()
            elif self.model_name == "custom":

                image_name = file.split("/")[-1]

                with torch.no_grad():
                    output = self.model(
                        x=image_tensor,
                        image_name=image_name,
                        centers_image=self._get_centers_from_file(image_name),
                    )

                    probabilities = torch.softmax(output, dim=1)

                    prediction = torch.argmax(probabilities, dim=1).item()

            retorno = self._get_key_from_value(
                dicte=dicionario_labels, target_value=prediction
            )

            resultado[retorno] += 1

        return resultado

    def _listar_images(self, path):
        all_image = [
            f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
        ]

        all_images_path_complete = []
        for each_img in all_image:
            all_images_path_complete.append(path + "/" + each_img)

        return all_images_path_complete

    def plot_confusion_matrix(self, arrays, labels_name, type_plot):

        labels_name.remove("Negative for intraepithelial lesion")

        if self.model_name == "binario":
            labels_name.insert(0, "NFIL")
        else:
            labels_name.insert(4, "NFIL")

        if type_plot == "sns":
            conf_matrix = np.array(arrays)
            row_sums = conf_matrix.sum(axis=1, keepdims=True)
            percent_matrix = conf_matrix / row_sums * 100
            plt.figure(figsize=(8, 6))
            sns.set(font_scale=1.5)
            sns.heatmap(
                percent_matrix,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                cbar=False,
                xticklabels=labels_name,
                yticklabels=labels_name,
            )
            # label_font = {'size':'18'}
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            # title_font = {'size':'21'}
            plt.title("Confusion Matrix")
            plt.show()
        if type_plot == "scikit":
            conf_matrix = np.array(arrays)
            row_sums = conf_matrix.sum(axis=1, keepdims=True)
            percent_matrix = conf_matrix / row_sums * 100
            plt.figure(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=percent_matrix, display_labels=labels_name
            )
            disp.plot()
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.show()

    def generate_confusion_matrix(self, main_path, labels, type_plot):
        all_folders = os.listdir(main_path)

        resultados = {
            folder: {predicts: 0 for predicts in all_folders}
            for folder in all_folders
        }

        for each_folder in all_folders:

            lista_imagens_each_class = self._listar_images(
                os.path.join(main_path, each_folder)
            )

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

        self.plot_confusion_matrix(
            arrays=full_list, labels_name=all_folders, type_plot=type_plot
        )

    def __reshape_transform(self, tensor, height=14, width=14):
        result = tensor[:, 1:, :]
        result = result.reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.permute(0, 3, 1, 2)
        return result

    def run_grad_cam(self, image_dir, qtd=100):
        image_paths = [
            os.path.join(image_dir, img_name)
            for img_name in os.listdir(image_dir)
            if img_name.endswith(".png")
        ]

        if qtd == None:
            num_images = len(image_paths)
        else:

            num_images = qtd

            random.shuffle(image_paths)

            image_paths = image_paths[:qtd]

        fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))

        target_layers = [
            self.model.model.vit.encoder.layer[11].attention.output
        ]
        gradcam = EigenCAM(
            model=self.model,
            target_layers=target_layers,
            reshape_transform=self.__reshape_transform,
        )

        for i, image_path in enumerate(image_paths):
            rgb_img = cv2.imread(image_path)[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img, (224, 224))
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(
                rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )

            targets = None
            grayscale_cam = gradcam(input_tensor=input_tensor, targets=targets)

            cam_image = show_cam_on_image(
                rgb_img, grayscale_cam[0, :], use_rgb=True
            )
            axes[i, 0].imshow(rgb_img)
            axes[i, 0].set_title("Imagem Original")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(cam_image)
            axes[i, 1].set_title("Grad-CAM")
            axes[i, 1].axis("off")

            if i == qtd:
                break

        plt.tight_layout()
        plt.show()

    def shapley_model_predict(self, images):
        with torch.no_grad():
            images = torch.tensor(images).float()
            images = images.permute(0, 3, 1, 2)
            return self.model(images).cpu().numpy()

    def run_shapley(self, image_dir, class_names, img_tam=(224, 224)):
        image_paths = [
            os.path.join(image_dir, img_name)
            for img_name in os.listdir(image_dir)
            if img_name.endswith(".png")
        ]
        num_images = len(image_paths)

        class_names_list = list(class_names.keys())

        for each_image_path in image_paths:
            image = Image.open(each_image_path).convert("RGB")
            image = image.resize(img_tam)
            image_array = np.array(image)
            image_array = np.expand_dims(image_array, axis=0)

            transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                ]
            )

            image_tensor = transform(image).unsqueeze(0)

            if self.model_name == "default":
                with torch.no_grad():
                    output = self.model(image_tensor)

                    probabilities = torch.softmax(output, dim=1)

                    prediction = torch.argmax(probabilities, dim=1).item()

            elif self.model_name == "binario":
                with torch.no_grad():
                    output = self.model(image_tensor)

                    probabilities = torch.sigmoid(output)

                    prediction = (probabilities > 0.5).int().item()

            retorno = self._get_key_from_value(
                dicte=class_names, target_value=prediction
            )

            print(f"Previsão: {retorno}")
            masker = shap.maskers.Image("inpaint_telea", image_array[0].shape)

            explainer = shap.Explainer(
                self.shapley_model_predict,
                masker,
                output_names=class_names_list,
            )

            shap_values = explainer(image_array)

            plt.figure(figsize=(20, num_images * 5))
            shap.image_plot(shap_values, image_array)

            plt.show()

    def calculate_accuracy(self, data_loader):
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(
                data_loader, desc="Calculando acurácia", unit="batch"
            ):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def get_dataloader_from_directory(
        self, root_path, batch_size=32, image_size=(224, 224)
    ):
        transform = T.Compose(
            [
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        dataset = ImageFolder(root=root_path, transform=transform)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            generator=g,
            worker_init_fn=seed_worker,
        )

        return data_loader

    def calculate_accuracy_custom(self, data_loader):
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, img_names, centers in tqdm(
                data_loader, desc="Calculando acurácia", unit="batch"
            ):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(
                    x=images,
                    image_name=img_names,
                    centers_image=centers,
                )

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def get_dataloader_from_directory_CustoImageFolder(
        self, root_path, batch_size=32, image_size=(224, 224)
    ):
        transform = T.Compose(
            [
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        dataset = CustomImageFolder(
            root=root_path,
            centers_dict=self.centers,
            transform=transform,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            generator=g,
            worker_init_fn=seed_worker,
        )

        return data_loader
