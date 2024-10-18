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
        self.__PATH_TO_FILES: Path = Path.joinpath(self.__PROJECT_DIR, 'library', 'files')
        self.__PATH_TO_FILES.mkdir(parents=True, exist_ok=True)

    # region Functions to getting basic settings
    @property
    def project_dir(self) -> Path:
        return self.__PROJECT_DIR

    @property
    def path_to_files(self) -> Path:
        return self.__PATH_TO_FILES
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

    # region Functions to getting hf settings
    @property
    def token(self) -> str:
        return self.__TOKEN
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
