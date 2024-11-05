import os
import logging
from logging.config import dictConfig
from logging.handlers import TimedRotatingFileHandler
from core_config import c_basic

# Конфигурация логирования
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO" if c_basic.debug else "WARNING",
        },
        "file": (
            {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "formatter": "default",
                "level": "INFO" if c_basic.debug else "WARNING",
                "filename": os.path.join(c_basic.path_to_log_files, "project.log"),
                "when": "midnight",  # Логи ротации каждый день в полночь
                "backupCount": 7,  # Хранить 7 файлов (неделя)
                "encoding": "utf-8",  # Для поддержки юникода в логах
            }
            if not c_basic.debug
            else None
        ),
    },
    "root": {
        "handlers": ["console", "file"] if not c_basic.debug else ["console"],
        "level": "INFO" if c_basic.debug else "WARNING",
    },
}

# Применение конфигурации логирования
if c_basic.debug:
    logging_config["handlers"].pop(
        "file", None
    )  # Убираем file-обработчик в режиме отладки
dictConfig(logging_config)
