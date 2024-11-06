import os
import shutil
from pathlib import Path

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

    def create_and_save_index(
        self,
        categories=None,
        reload=False,
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
            print(
                f"Начат процесс создания индекса:\nкатегория - {cat}\nЧисло документов - {len(documents)}"
            )
            storage_context = StorageContext.from_defaults()
            index: VectorStoreIndex = VectorStoreIndex(
                documents, storage_context=storage_context
            )
            index.storage_context.persist()
            self.get_indexes[cat] = index
            print(
                f"Индекс добавлен в словарь:\nкатегория - {cat}\nЧисло документов - {len(documents)}"
            )
            storage_context.persist(persist_dir=index_path)

    def load_index(
        self,
        categories=None,
        reload=False,
    ):
        if categories is None:
            categories = os.listdir(c_basic.path_to_dataset_dir)
        indexes_dirs = os.listdir(c_basic.path_to_indexes_dir)
        if not reload:
            # проверяем какие категории уже присутствуют в папке с индексами и создаем список для загрузки
            indexes_for_load = [
                category for category in categories if category in indexes_dirs
            ]
            # проверяем какие категории отсутствуют в папке с индексами и создаем список для создания
            missing_indexes = [
                category for category in categories if category not in indexes_dirs
            ]
            # загружаем существующие индексы
            for index_name in indexes_for_load:
                index_path = Path.joinpath(c_basic.path_to_indexes_dir, index_name)
                try:
                    storage_context = StorageContext.from_defaults(
                        persist_dir=str(index_path)
                    )
                    index = load_index_from_storage(storage_context=storage_context)
                    self.get_indexes[index_name] = index
                except FileNotFoundError:
                    missing_indexes.append(index_name)
                    remove_directory(index_path)

            # создаем и сохраняем недостающие индексы
            self.create_and_save_index(categories=missing_indexes)
        else:
            remove_directory(c_basic.path_to_indexes_dir)
            c_basic.path_to_indexes_dir.mkdir(parents=True, exist_ok=True)
            self.create_and_save_index(categories=categories)


in_helper = IndexHelper()
in_helper.load_index(reload=True)
