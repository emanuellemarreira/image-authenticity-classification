import os
import tensorflow as tf

from enum import Enum
from typing import Tuple, Callable
from kagglehub import dataset_download


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
    for path, expected_count in __BASE_LOCAL_DATA_PATHS:
        exists, count = _check_folder(path)
        if not exists or count != expected_count:
            return False

    return True


def _get_tf_preprocess_fn(model_type: ModelType,
                          resize_to: Tuple[int, int]) -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """
    Retorna função de preprocessing para ser aplicada no map() do tf.data.Dataset
    """
    def preprocess(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.resize(image, resize_to)
        image = tf.cast(image, tf.float32) / 255.0

        if model_type in [ModelType.EFFICIENTNET_B0, ModelType.RESNET50]:
            mean = tf.constant([0.485, 0.456, 0.406])
            std = tf.constant([0.229, 0.224, 0.225])
            image = (image - mean) / std

        elif model_type == ModelType.XCEPTION:
            # Normalização para [-1, 1]
            image = (image - 0.5) * 2.0

        return image, label

    return preprocess


def load_and_preprocess_data_tf(resize_to: Tuple[int, int] = (224, 224),
                                batch_size: int = 32,
                                model_type: ModelType = ModelType.GENERAL) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    train_path = "../data/train"
    test_path = "../data/test"

    if not _is_dataset_present():
        path = dataset_download(__ONLINE_DATASET_URI)
        train_path = f"{path}/cifake/train"
        test_path = f"{path}/cifake/test"

    preprocess_fn = _get_tf_preprocess_fn(model_type, resize_to)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=resize_to,
        batch_size=batch_size,
        shuffle=True
    ).map(preprocess_fn).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        image_size=resize_to,
        batch_size=batch_size,
        shuffle=False
    ).map(preprocess_fn).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds
