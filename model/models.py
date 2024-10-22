import os
from typing import Dict
import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
    GPT2Tokenizer,
    T5ForConditionalGeneration,
)
from huggingface_hub import login
from core_config import c_hf, c_basic
from model.conversation import Conversation


class ModelBase:
    def __init__(
        self,
        model_name: str,
        is_cuda: bool = False,
    ):
        login(token=c_hf.token)
        self.model_name = model_name
        self.is_cuda = is_cuda


class ModelSaigaMistral7BLora(ModelBase):
    def __init__(
        self,
        model_name: str = "IlyaGusev/saiga_mistral_7b",
        is_cuda: bool = False,
    ):
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.device = None
        super().__init__(
            model_name,
            is_cuda=is_cuda,
        )

    def load_model(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.is_cuda else "cpu"
        )
        peft_config = PeftConfig.from_pretrained(self.model_name)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=c_basic.path_to_models,
        )
        self.model = PeftModel.from_pretrained(
            model,
            self.model_name,
            torch_dtype=torch.float16,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,
            cache_dir=c_basic.path_to_models,
        )

        self.generation_config = GenerationConfig.from_pretrained(self.model_name)

    def generate(self, prompt):
        data = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(self.model.device) for k, v in data.items()}
        output_ids = self.model.generate(
            **data, generation_config=self.generation_config
        )[0]
        output_ids = output_ids[len(data["input_ids"][0]) :]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output.strip()


class ModelFREDT5117B(ModelBase):
    def __init__(
        self,
        model_name: str = "ai-forever/FRED-T5-1.7B",
        is_cuda: bool = True,
    ):
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.device = None
        super().__init__(
            model_name,
            is_cuda=is_cuda,
        )

    def load_model(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.model_name,
            eos_token="</s>",
            cache_dir=c_basic.path_to_models,
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name,
            cache_dir=c_basic.path_to_models,
        )
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.is_cuda else "cpu"
        )
        self.model.to(self.device)

    def generate(self, prompt):
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        print("/" * 50)
        print(input_ids)
        print("/" * 50)
        outputs = self.model.generate(
            input_ids,
            eos_token_id=self.tokenizer.eos_token_id,
            early_stopping=True,
            # num_beams=3,
            # max_new_tokens=256,
            # repetition_penalty=1.2,
            # do_sample=True,
            # temperature=1.2,
        )
        print("-" * 50)
        print(outputs)
        print("-" * 50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class ModelSaigaMistral7BGguf(ModelBase):
    def __init__(
        self,
        model_name: str = "IlyaGusev/saiga_mistral_7b_gguf",
        is_cuda: bool = False,
        model_version: str = "model-q4_K.gguf",
    ):
        self.model = None
        self.model_version = model_version
        self.tokenizer = None
        self.generation_config = None
        self.device = None
        super().__init__(
            model_name,
            is_cuda=is_cuda,
        )

    def load_model(
        self,
        new_conversation: Conversation = None,
        n_ctx: int = 4000,
    ):
        import requests
        from llama_cpp import Llama

        if not new_conversation:
            new_conversation = Conversation(model_name=self.model_name)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.is_cuda else "cpu"
        )
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
            else:
                raise ValueError(
                    f"Ошибка при скачивании: получен статус {response.status_code}"
                )
        else:
            print("Используем ранее загруженную версию модели")
        self.model = Llama(
            model_path=model_path,
            split_mode=0,
            n_gpu_layers=-1,
            use_gpu=True,
            n_ctx=n_ctx,
            n_parts=1,
        )
        system_prompt = new_conversation.get_system_prompt()
        system_tokens = self.model.tokenize(
            system_prompt.encode("utf-8"), special=False
        )

        self.model.eval(system_tokens)

    def generate(
        self,
        prompt,
        top_k=30,
        top_p=0.9,
        temperature=0.2,
        repeat_penalty=1.1,
    ):
        tokens = self.model.tokenize(prompt.encode("utf-8"), special=False)
        generator = self.model.generate(
            tokens,
            top_k=top_k,
            top_p=top_p,
            temp=temperature,
            repeat_penalty=repeat_penalty,
        )
        for token in generator:
            token_str = self.model.detokenize([token]).decode("utf-8", errors="ignore")
            tokens.append(token)
            if token == self.model.token_eos():
                break
            print(token_str, end="", flush=True)
        print()


# fred = ModelFREDT5117B()
# fred.load_model()
# # conversation = Conversation(model_name="ai-forever/FRED-T5-1.7B")
# # conversation.add_user_message("реши пример и дай ответ 5+4=")
# # prompt_i = conversation.get_prompt()
# # print("*" * 50)
# # print(prompt_i)
# # print("*" * 50)
# print(fred.generate(prompt="ты можешь решать математические примеры?"))

saiga_gguf = ModelSaigaMistral7BGguf()
conversation = Conversation(saiga_gguf.model_name)
saiga_gguf.load_model(new_conversation=conversation, n_ctx=8000)
# conversation.add_user_message("У тебя есть имя?")
# result_prompt = conversation.get_prompt()
# saiga_gguf.generate(result_prompt)

while True:
    user_message = input("User: ")
    conversation.add_user_message(user_message)
    result_prompt = conversation.get_prompt()
    saiga_gguf.generate(result_prompt)
