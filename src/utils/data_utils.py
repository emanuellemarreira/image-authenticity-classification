import os

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


def load_and_preprocess_data(resize_to: Tuple[int, int] = (32, 32), keras_format: bool = False):
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

    transform_list = [
        transforms.Resize(resize_to),
        transforms.ToTensor()
    ]

    if keras_format:
        transform_list.append(transforms.Lambda(lambda x: x.permute(1, 2, 0)))

    transform = transforms.Compose(transform_list)

    return (
        datasets.ImageFolder(root=train_path, transform=transform),
        datasets.ImageFolder(root=test_path, transform=transform)
    )