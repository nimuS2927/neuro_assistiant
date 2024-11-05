import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from transformers import AutoTokenizer
import requests
from huggingface_hub import login
from core_config import c_hf, c_basic

from llama_index.core.llms import ChatMessage

from model.conversation import MODELS_CONFIG


#
# messages = [
#     ChatMessage(
#         role="system", content="You are a pirate with a colorful personality"
#     ),
#     ChatMessage(role="user", content="What is your name"),
# ]


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
        login(token=c_hf.token, add_to_git_credential=True)
        self.model_name = model_name
        self.model_version = model_version
        self.is_cuda = is_cuda
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
                "temp": temperature,
                "repeat_penalty": repeat_penalty,
            },
            system_prompt=MODELS_CONFIG[model_name]["default_system_prompt"],
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "IlyaGusev/saiga_mistral_7b_lora", use_fast=False
        )

    def get_model_path(self):
        model_path = os.path.join(c_basic.path_to_models, self.model_version)
        if not os.path.exists(model_path):
            print("Скачивание модели")
            url = "https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf/resolve/main/model-q4_K.gguf"
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
    def messages_to_prompt(messages: list[ChatMessage]):
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"<s>{message.role}\n{message.content}</s>\n"
            elif message.role == "user":
                prompt += f"<s>{message.role}\n{message.content}</s>\n"
            elif message.role == "bot":
                prompt += f"<s>bot\n"

        # ensure we start with a system prompt, insert blank if needed
        if not prompt.startswith("<s>system\n"):
            prompt = "<s>system\n</s>\n" + prompt

        # add final assistant prompt
        prompt = prompt + "<s>bot\n"
        return prompt

    # @staticmethod
    # def completion_


li_saiga_mistral_7b_gguf = LISaigaMistral7BGguf()
