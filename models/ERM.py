import torch
import torch.nn as nn
import torchvision.models as models

class ERM(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True):
        super().__init__()

        # Map backbone names to their constructors
        factories = {
            'resnet18': lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
            'resnet50': lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
            'vgg16'   : lambda: models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1),
            'alexnet' : lambda: models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1),
        }

        try:
            self.backbone = factories[backbone]()
        except KeyError:
            raise ValueError(
                f"Backbone '{backbone}' is not supported. "
                f"Choose from {list(factories.keys())}."
            )

        self._replace_classifier(num_classes)

    def _replace_classifier(self, num_classes: int):
        """
        Replace the final classification layer for common torchvision backbones.
        - ResNet:   model.fc
        - VGG/Alex: model.classifier[6]
        """
        if hasattr(self.backbone, 'fc') and isinstance(self.backbone.fc, nn.Linear):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, 'classifier') and isinstance(self.backbone.classifier, nn.Sequential):
            # Assume last layer is a Linear head
            if not isinstance(self.backbone.classifier[-1], nn.Linear):
                raise TypeError("Expected the last classifier layer to be nn.Linear.")
            in_features = self.backbone.classifier[-1].in_features
            new_classifier = list(self.backbone.classifier)
            new_classifier[-1] = nn.Linear(in_features, num_classes)
            self.backbone.classifier = nn.Sequential(*new_classifier)
        else:
            raise TypeError("Unsupported backbone head structure for automatic replacement.")

    def forward(self, x):
        return self.backbone(x)