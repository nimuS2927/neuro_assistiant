import json
import os
import logging
import logging_config
from datetime import datetime
from pathlib import Path

from core_config import c_basic

logger = logging.getLogger(os.path.basename(__file__))


def create_index_config_file(
    path_to_config_file: Path = None,
    path_to_dataset_dir: Path = None,
    categories: list[str] = None,
    is_old_delete: bool = True,
):
    path_to_config_file = path_to_config_file or c_basic.project_dir
    path_to_dataset_dir = path_to_dataset_dir or c_basic.path_to_dataset_dir
    if not categories:
        categories = os.listdir(path_to_dataset_dir)
        logger.info(
            "Файл будет сохранен:\n%s\nКатегории индексов взяты из папки датасета:\n%s\nВсего обнаружено %-4d категорий",
            path_to_config_file,
            path_to_dataset_dir,
            len(categories),
        )
    else:
        logger.info(
            "Файл будет сохранен:\n%s\nКатегории индексов переданы пользователем.\nВсего обнаружено %-4d категорий",
            path_to_config_file,
            len(categories),
        )

    config_ = []
    for category in categories:
        config_.append(
            {
                "folder_name": category,
                "name": "example",
                "description": "some description",
            }
        )

    path_to_file = Path.joinpath(path_to_config_file, "index_config.json")
    # Если найден файл с конфигом, то переименуем его
    if path_to_file.exists():
        if is_old_delete:
            path_to_file.unlink(missing_ok=True)
            logger.info(
                "Обнаружен равнее созданный файл index_config.json. Файл УДАЛЕН."
            )
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_file_name = f"{path_to_file.stem}_{timestamp}{path_to_file.suffix}"
            new_file_path = path_to_file.with_name(new_file_name)
            path_to_file.rename(new_file_path)

            logger.info(
                "Обнаружен равнее созданный файл index_config.json\nФайл переименован: %s",
                new_file_name,
            )

    with open(path_to_file, "w", encoding="utf-8") as f:
        json.dump(config_, f, indent=4, ensure_ascii=False)
        logger.info("Записан файл index_config.json")


if __name__ == "__main__":
    create_index_config_file(is_old_delete=False)
