import json
import os
import shutil
from pathlib import Path
import logging

from llama_index.core.base.base_retriever import BaseRetriever

import logging_config
from sentence_transformers import SentenceTransformer
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from tqdm import tqdm

from core_config import c_basic
from readers import MarkDownReader
from llama_index.core.schema import Document
from model import li_saiga_mistral_7b_gguf
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    load_indices_from_storage,
)
from llama_index.core.llms.llm import LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import set_global_tokenizer

Settings.llm = li_saiga_mistral_7b_gguf.llm

Settings.embed_model = HuggingFaceEmbedding(model_name="cointegrated/rubert-tiny2")
Settings.context_window = 4000
Settings.num_output = 512
set_global_tokenizer(li_saiga_mistral_7b_gguf.tokenizer.encode)

logger = logging.getLogger(os.path.basename(__file__))


def remove_directory(path):
    # Проверяем, существует ли директория
    if os.path.exists(path) and os.path.isdir(path):
        try:
            shutil.rmtree(path)  # Удаляем директорию и всё её содержимое
            print(f"Удалена папка: {path}")
        except Exception as e:
            print(f"Ошибка при удалении папки {path}: {e}")
    else:
        print(f"Папка {path} не существует или это не директория.")


class IndexHelper:
    _indexes = {}

    def __init__(
        self,
    ):
        self.reader_md = MarkDownReader(
            chunk_size=500, overlap_size=100, expected_formats=[".md"]
        )

    @property
    def get_indexes(self):
        return self.__class__._indexes

    def get_retrievers_and_categories(self):
        path_to_config_file: Path = Path.joinpath(
            c_basic.project_dir, "index_config.json"
        )
        if not path_to_config_file.exists():
            raise FileNotFoundError("Файл index_config.json не обнаружен")
        with open(path_to_config_file, "r", encoding="utf-8") as f:
            config_indexes = json.load(f)

        retrievers: dict[str, BaseRetriever] = {
            index_dict["name"]: self.get_indexes[
                index_dict["folder_name"]
            ].as_retriever(similarity_top_k=5)
            for index_dict in config_indexes
        }
        categories: dict[str, str] = {
            index_dict["name"]: index_dict["name"] + "\n" + index_dict["description"]
            for index_dict in config_indexes
        }
        return retrievers, categories

    def create_and_save_index(
        self,
        categories: list[str] = None,
        verbose: bool = False,
    ):
        if categories is None:
            categories = os.listdir(c_basic.path_to_dataset_dir)
        for cat in tqdm(categories, desc="Обработка категорий"):
            index_path = Path.joinpath(c_basic.path_to_indexes_dir, cat)

            documents = []
            path_to_files = Path.joinpath(c_basic.path_to_dataset_dir, cat)
            files = os.listdir(path_to_files)

            # for file in tqdm(files, desc="Обработка файлов"):
            for file in tqdm(files, desc="Обработка файлов"):
                if file.endswith(".md"):
                    path_to_file = Path.joinpath(path_to_files, file)
                    print(f"Парсинг:\nпапка {cat}\nфайл {file}")
                    documents.extend(self.reader_md.read_file(path_to_file))
            if verbose:
                print(
                    f"Начат процесс создания индекса:\nкатегория - {cat}\nЧисло документов - {len(documents)}"
                )
            storage_context = StorageContext.from_defaults()
            logger.info("Старт создания индекса %r", cat)
            index: VectorStoreIndex = VectorStoreIndex(
                documents, storage_context=storage_context
            )
            index.storage_context.persist()
            logger.info("Индекса %r создан", cat)
            self.get_indexes[cat] = index
            if verbose:
                print(
                    f"Индекс добавлен в словарь:\nкатегория - {cat}\nЧисло документов - {len(documents)}"
                )
            storage_context.persist(persist_dir=index_path)

    def load_index(self, categories=None, reload=False, verbose: bool = False):
        if categories is None:
            categories = os.listdir(c_basic.path_to_dataset_dir)
        indexes_dirs = os.listdir(c_basic.path_to_indexes_dir)
        if not reload:
            # проверяем какие категории уже присутствуют в папке с индексами и создаем список для загрузки
            indexes_for_load = [
                category for category in categories if category in indexes_dirs
            ]
            logger.info("обнаружено %d индексов для загрузки", len(indexes_for_load))
            if verbose:
                print("Индексы для загрузки:\n%s" % "\n".join(indexes_for_load))
            # проверяем какие категории отсутствуют в папке с индексами и создаем список для создания
            missing_indexes = [
                category for category in categories if category not in indexes_dirs
            ]
            logger.info(
                "обнаружено %d папка(-ок) с данными для создания индекса(-ов)",
                len(indexes_for_load),
            )
            if verbose:
                print(
                    "Категории для создания индексов:\n%s" % "\n".join(missing_indexes)
                )

            # загружаем существующие индексы
            for index_name in indexes_for_load:
                index_path = Path.joinpath(c_basic.path_to_indexes_dir, index_name)
                try:
                    storage_context = StorageContext.from_defaults(
                        persist_dir=str(index_path)
                    )
                    index = load_index_from_storage(storage_context=storage_context)
                    self.get_indexes[index_name] = index
                    logger.info("Загружен индекс %s", index_name)
                except FileNotFoundError:
                    missing_indexes.append(index_name)
                    remove_directory(index_path)
                    logger.info(
                        "Индекс %r поврежден, он удален и будет создан заново",
                        index_name,
                    )
            # создаем и сохраняем недостающие индексы
            self.create_and_save_index(categories=missing_indexes)
        else:
            remove_directory(c_basic.path_to_indexes_dir)
            c_basic.path_to_indexes_dir.mkdir(parents=True, exist_ok=True)
            self.create_and_save_index(categories=categories)


in_helper = IndexHelper()
in_helper.load_index()
print(in_helper.get_indexes)
