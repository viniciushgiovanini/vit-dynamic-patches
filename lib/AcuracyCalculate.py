from typing import Callable

import torch


class AcuracyCalculate:
    def __init__(self, device):
        self.device = device

    def acc_calculate(self, model, dataloader, model_type):
        forward_handler = self.__select_calculator_handler(model_type)

        if forward_handler is None:
            raise Exception(
                "Não encontrou nenhuma estratégia para calcular acc final"
            )

        return self.__acc_calculate(model, dataloader, forward_handler)

    def __select_calculator_handler(self, model_type) -> Callable:
        if model_type == "custom":
            return self.__custom_forward
        elif model_type == "default":
            return self.__default_forward
        else:
            return None

    def __acc_calculate(self, model, dataloader, forward_handler: Callable):
        model.to(self.device)
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in dataloader:
                images, labels, *extra = batch
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = forward_handler(model, images, extra)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total if total > 0 else 0.0

    def __custom_forward(self, model, images, extra):

        image_names, centers_images = extra
        model.model.vit.embeddings.patch_embeddings.current_centers_image = (
            image_names,
            centers_images,
        )
        outputs = model(
            images,
            image_name=image_names,
            centers_image=centers_images,
        )
        model.model.vit.embeddings.patch_embeddings.current_centers_image = None
        return outputs

    def __default_forward(self, model, images, extra):
        return model(images)
