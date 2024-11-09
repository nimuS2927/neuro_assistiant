import os
from typing import Optional

from huggingface_hub import login, HfApi, HfFolder

import logging
import logging_config
from core_config import c_hf

logger = logging.getLogger(os.path.basename(__file__))


def check_token_validity(token: Optional[str]):
    if not token:
        token = HfFolder.get_token()
    if token is None:
        logger.info("Токен отсутствует")
        return False

    api = HfApi()
    try:
        # Проверка через вызов профиля пользователя
        user = api.whoami(token=token)
        logger.info("Токен действителен. Пользователь: %s", user["name"])
        return True
    except Exception as e:
        logger.info("Ошибка аутентификации", exc_info=e)

        return False


def authenticate_hf(validate: bool = False):
    # Проверка наличия токена в текущем окружении
    token = HfFolder.get_token()
    if token is None:
        # Запрос на логин пользователя, если токен отсутствует
        logger.info("Токен отсутствует. Выполняем аутентификацию.")
        login(
            token=c_hf.token, add_to_git_credential=True
        )  # Вызов метода для логина; откроется окно для ввода токена
    else:
        logger.info("Токен обнаружен.")
        if validate:
            logger.info("Проверяем валидность.")
            if not check_token_validity(token):
                login(token=c_hf.token, add_to_git_credential=True)
