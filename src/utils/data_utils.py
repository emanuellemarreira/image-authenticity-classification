import os

from enum import Enum
from typing import Tuple
from kagglehub import dataset_download
from torchvision import datasets, transforms


# Detalhe
# Pytorch    -> [canal, altura, largura]
# Tensorflow -> [altura, largura, canal]

__ONLINE_DATASET_URI = "birdy654/cifake-real-and-ai-generated-synthetic-images"
__BASE_LOCAL_DATA_PATHS = [
    ("../data/train/REAL", 50000),
    ("../data/train/FAKE", 50000),
    ("../data/test/REAL", 10000),
    ("../data/test/FAKE", 10000)
]

class ModelType(Enum):
    GENERAL = "general"
    EFFICIENTNET_B0 = "efficientnet_b0"
    RESNET50 = "resnet50"
    XCEPTION = "xception"


def _check_folder(path: str) -> Tuple[bool, int]:
    if not os.path.isdir(path):
        return False, 0

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return True, len(files)


def _is_dataset_present() -> bool:
    """
    Verifica se todas as pastas esperadas existem e contêm o número correto de arquivos.
    """
    for path, expected_count in __BASE_LOCAL_DATA_PATHS:
        exists, count = _check_folder(path)
        if not exists or count != expected_count:
            return False
    
    return True


def _to_keras_format(x):
    return x.permute(1, 2, 0)


def _get_preprocessing_transforms(model_type: ModelType,
                                  resize_to: Tuple[int, int],
                                  keras_format: bool = False):
    
    transform_list = [transforms.Resize(resize_to)]
    
    if model_type == ModelType.EFFICIENTNET_B0:
        # EfficientNet-B0 precisa de 224x224
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif model_type == ModelType.RESNET50:
        # ResNet50 usa 224x224 como padrão
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif model_type == ModelType.XCEPTION:
        # Xception precisa de 299x299 idealmente
        # Xception usa normalização com valores entre -1 e 1
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    else:
        transform_list.extend([
            transforms.ToTensor()
        ])
    
    # Converter para formato Keras se necessário (CHW -> HWC)
    if keras_format:
        transform_list.append(transforms.Lambda(_to_keras_format))
    
    return transforms.Compose(transform_list)


def load_and_preprocess_data(resize_to: Tuple[int, int] = (32, 32),
                             keras_format: bool = False,
                             model_type: ModelType = ModelType.GENERAL):
    """
    Carrega e faz o pré-processamento do CIFAKE dataset. Caso não esteja presente, baixa do kaggle.

    :param resize_to: Tamanho alvo da imagem durante o pré-processamento.
    :return: Tuple[train_dataset, test_dataset].
    """
    
    train_path = "../data/train"
    test_path = "../data/test"
    
    if not _is_dataset_present():
        path = dataset_download(__ONLINE_DATASET_URI)
        train_path = f"{path}/cifake/train"
        test_path = f"{path}/cifake/test"

    transform = _get_preprocessing_transforms(model_type, resize_to, keras_format)

    return (
        datasets.ImageFolder(root=train_path, transform=transform),
        datasets.ImageFolder(root=test_path, transform=transform)
    )