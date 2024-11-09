import os
from typing import Sequence

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from transformers import AutoTokenizer
import requests
from auth.authentication_in_hf import authenticate_hf
from core_config import c_hf, c_basic

from llama_index.core.llms import ChatMessage

from model.conversation import MODELS_CONFIG

import logging
import logging_config


logger = logging.getLogger(os.path.basename(__file__))


class LISaigaMistral7BGguf:
    def __init__(
        self,
        model_name: str = "IlyaGusev/saiga_mistral_7b_gguf",
        is_cuda: bool = False,
        model_version: str = "model-q4_K.gguf",
        n_ctx: int = 4000,
        top_k=30,
        top_p=0.9,
        temperature=0.2,
        repeat_penalty=1.1,
    ):
        authenticate_hf()

        self.model_name = model_name
        self.model_version = model_version
        self.is_cuda = is_cuda
        logger.info("Инициализация модели LlamaCPP")
        self.llm = LlamaCPP(
            model_path=self.get_model_path(),
            context_window=n_ctx,
            model_kwargs={
                "n_gpu_layers": 50,
                "use_gpu": True,
            },
            generate_kwargs={
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "repeat_penalty": repeat_penalty,
                "max_tokens": 1024,
            },
            system_prompt=MODELS_CONFIG[model_name]["default_system_prompt"],
            messages_to_prompt=self.messages_to_prompt,
            completion_to_prompt=self.completion_to_prompt,
            verbose=True,
        )
        logger.info("Инициализация токенизатора")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "IlyaGusev/saiga_mistral_7b_lora", use_fast=False
        )

    def get_model_path(self):
        model_path = os.path.join(c_basic.path_to_models, self.model_version)
        if not os.path.exists(model_path):
            print("Скачивание модели")
            url = f"https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf/resolve/main/{self.model_version}"
            response = requests.get(url)

            if response.status_code == 200:
                # Сохранение файла
                with open(model_path, "wb") as file:
                    file.write(response.content)
                print("Скачивание завершено успешно.")
                return model_path
            else:
                raise ValueError(
                    f"Ошибка при скачивании: получен статус {response.status_code}"
                )
        return model_path

    @staticmethod
    def messages_to_prompt(messages: Sequence[ChatMessage], context_str: str = None):
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"<s>{message.role}\n{message.content}</s>\n"
            elif message.role == "user":
                prompt += f"<s>{message.role}\n{message.content}</s>\n"
            elif message.role == "bot":
                prompt += f"<s>{message.role}\n{message.content}</s>\n"

        # ensure we start with a system prompt, insert blank if needed
        if not prompt.startswith("<s>system\n"):
            prompt = "<s>system\n</s>\n" + prompt

        # add final assistant prompt
        prompt = prompt + "<s>bot\n"
        return prompt

    @staticmethod
    def completion_to_prompt(completion: str):
        return f"<s>system\n</s>\n<s>user\n{completion}</s>\n<s>bot\n"


li_saiga_mistral_7b_gguf = LISaigaMistral7BGguf()
