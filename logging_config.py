import os
import logging
from logging.config import dictConfig
from core_config import c_basic


# Определение пользовательского фильтра
class BelowWarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno < logging.WARNING


# Конфигурация логирования
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        },
    },
    "filters": {
        "below_warning": {
            "()": BelowWarningFilter,
        },
    },
    "handlers": {
        "console_stdout": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
            "filters": ["below_warning"],
        },
        "console_stderr": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "WARNING",
            "stream": "ext://sys.stderr",
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
        "handlers": (
            ["file"] if not c_basic.debug else ["console_stdout", "console_stderr"]
        ),
        "level": "DEBUG" if c_basic.debug else "WARNING",
    },
}

# Применение конфигурации логирования
if c_basic.debug:
    logging_config["handlers"].pop(
        "file", None
    )  # Убираем file-обработчик в режиме отладки
else:
    logging_config["handlers"].pop(
        "console_stdout", None
    )  # Убираем console_stdout-обработчик в режиме продакшен
    logging_config["handlers"].pop(
        "console_stderr", None
    )  # Убираем console_stderr-обработчик в режиме продакшен
dictConfig(logging_config)
