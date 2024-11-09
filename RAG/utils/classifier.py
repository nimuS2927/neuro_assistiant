import os

from pathlib import Path
from typing import List
from huggingface_hub import login

from auth.authentication_in_hf import authenticate_hf
from core_config import c_hf, c_basic
from transformers import pipeline


class Classifier:
    """
    Используется для классификации вопроса пользователя по категориям базы данных
    для определения индекса(-ов) в котором будет происходить поиск документов.
    """

    def __init__(
        self,
        model_name: str = "joeddav/xlm-roberta-large-xnli",
        specialization: str = "zero-shot-classification",
        path_to_dataset_dir: Path = c_basic.path_to_dataset_dir,
    ):
        authenticate_hf()
        self.model_name: str = model_name
        self.specialization: str = specialization
        self.path_to_dataset_dir: Path = path_to_dataset_dir
        self.categories: List[str] = self.get_categories()
        self.pipeline = None
        self.load_pipeline()

    # Названия папок внутри датасета будут являться названиями категорий на которые поделен датасет
    def get_categories(self):
        return os.listdir(self.path_to_dataset_dir)

    def load_pipeline(self):
        self.pipeline = pipeline(
            self.specialization,
            model=self.model_name,
            use_fast=False,
            device_map="auto",
        )


classifier_ = Classifier()
# sequence_to_classify = "какие существуют правила при использование такси?"
# response: Dict[str, Union[str, List[str]]] = classifier.pipeline(
#     sequence_to_classify, classifier.categories
# )
#
# for i in range(len(response["labels"])):
#     print(f"{response['labels'][i]} - {response['scores'][i]:.2f}")
