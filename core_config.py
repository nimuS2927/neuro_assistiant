import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

DEBUG = False


class CoreConfigBasic(object):
    _instance = None

    def __new__(cls):
        if not cls._instance:
            instance = super(CoreConfigBasic, cls).__new__(cls)
            cls._instance = instance
        return cls._instance

    def __init__(self):
        self.__PROJECT_DIR: Path = Path(__file__).parent
        self.__PATH_TO_FILES: Path = Path.joinpath(
            self.__PROJECT_DIR, "library", "files"
        )
        self.__PATH_TO_FILES.mkdir(parents=True, exist_ok=True)
        self.__PATH_TO_MODELS: Path = Path.joinpath(
            self.__PROJECT_DIR, "library", "models"
        )
        self.__PATH_TO_MODELS.mkdir(parents=True, exist_ok=True)
        self.__DATASET_DIR: str = (
            os.getenv("DATASET_DIR") if os.getenv("DATASET_DIR") else "yandex_files"
        )
        self.__PATH_TO_DATASET_DIR: Path = Path.joinpath(
            self.__PATH_TO_FILES, self.__DATASET_DIR
        )
        self.__PATH_TO_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        self.__PATH_TO_INDEXES_DIR: Path = Path.joinpath(
            self.__PATH_TO_FILES, "indexes"
        )
        self.__PATH_TO_INDEXES_DIR.mkdir(parents=True, exist_ok=True)
        self.__DEBUG = os.getenv("DEBUG", "False").lower() == "true"
        self.__PATH_TO_LOG_FILES: Path = Path.joinpath(self.__PATH_TO_FILES, "log")
        self.__PATH_TO_LOG_FILES.mkdir(parents=True, exist_ok=True)

    # region Functions to getting basic settings
    @property
    def project_dir(self) -> Path:
        return self.__PROJECT_DIR

    @property
    def path_to_files(self) -> Path:
        return self.__PATH_TO_FILES

    @property
    def path_to_models(self) -> Path:
        return self.__PATH_TO_MODELS

    @property
    def path_to_dataset_dir(self) -> Path:
        return self.__PATH_TO_DATASET_DIR

    @property
    def path_to_indexes_dir(self) -> Path:
        return self.__PATH_TO_INDEXES_DIR

    @property
    def debug(self) -> bool:
        return self.__DEBUG

    @property
    def path_to_log_files(self) -> Path:
        return self.__PATH_TO_LOG_FILES

    # endregion


c_basic = CoreConfigBasic()


class HFConfig(object):
    _instance = None

    def __new__(cls):
        if not cls._instance:
            instance = super(HFConfig, cls).__new__(cls)
            cls._instance = instance
        return cls._instance

    def __init__(self):
        self.__TOKEN: str = os.getenv("HF_TOKEN")
        self.__N_CTX = os.getenv("N_CTX")

    # region Functions to getting hf settings
    @property
    def token(self) -> str:
        return self.__TOKEN

    @property
    def n_ctx(self) -> int:
        return int(self.__N_CTX)

    # endregion


c_hf = HFConfig()


class ConfigProject(object):
    _instance = None

    def __new__(cls):
        if not cls._instance:
            instance = super(ConfigProject, cls).__new__(cls)
            cls._instance = instance
        return cls._instance

    def __init__(self):
        self.__basic = c_basic
        self.__hf = c_hf

    # region Functions to getting project settings
    @property
    def basic(self) -> CoreConfigBasic:
        return self.__basic

    @property
    def db(self) -> HFConfig:
        return self.__hf

    # endregion


c_project = ConfigProject()
