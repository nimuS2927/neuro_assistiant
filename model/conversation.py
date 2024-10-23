from typing import Dict

MODELS_CONFIG = {
    "IlyaGusev/saiga_mistral_7b": {
        "default_message_template": "<s>{role}\n{content}</s>",
        "default_response_template": "<s{role}\n",
        "default_system_prompt": """
                                Ты — Сайга, русскоязычный автоматический ассистент.
                                 Ты разговариваешь с людьми и помогаешь им.
                                """,
    },
    "IlyaGusev/saiga_mistral_7b_gguf": {
        "default_message_template": "<s>{role}\n{content}</s>",
        "default_response_template": "<s>{role}\n",
        "default_system_prompt": """Ты — Сайга, русскоязычный автоматический ассистент.
Ты разговариваешь с людьми и помогаешь им.
Думай шаг за шагом, чтобы быть уверенным в своем ответе.""",
    },
    "ai-forever/FRED-T5-1.7B": {
        "default_message_template": "<s>{role}\n{content}</s>",
        "default_response_template": "<s>{role}\n",
        "default_system_prompt": "Ты — персональный ассистент помощник, который называется НейроЛаборант отвечающий "
        "на вопросы пользователя,",
    },
}

LANGUAGES = ["en", "ru"]
LANGUAGES_VOCAB = {
    "en": {
        "user": "user",
        "assistant": "assistant",
        "system": "system",
        "document": "document",
    },
    "ru": {
        "user": "пользователь",
        "assistant": "ассистент",
        "system": "инструкция",
        "document": "документ",
    },
}

# Думай шаг за шагом и используй Documents для точного ответа на вопросы пользователя.
# Если ты не нашел ответа в документах, то прямо об этом и скажи.


class Conversation:
    def __init__(
        self,
        model_name: str,
        model_or_tokenizer=None,
        is_tokenizer: bool = True,
        config: Dict = None,
        language: str = "en",
    ):
        if config is None:
            config = MODELS_CONFIG
        self.model_name = model_name
        self.default_message_template = config.get(model_name).get(
            "default_message_template"
        )
        self.default_response_template = config.get(model_name).get(
            "default_response_template"
        )
        self.default_system_prompt = config.get(model_name).get("default_system_prompt")
        self.language = language.lower() if language.lower() in LANGUAGES else "en"
        self.language_vocab = LANGUAGES_VOCAB[self.language]
        self.messages = [{"role": "system:", "content": self.default_system_prompt}]
        self.is_tokenizer = is_tokenizer
        self.model_or_tokenizer = model_or_tokenizer

    def add_user_message(self, message):
        self.messages.append({"role": self.language_vocab["user"], "content": message})

    def add_bot_message(self, message):
        self.messages.append(
            {"role": self.language_vocab["assistant"], "content": message}
        )

    def add_documents(self, content):
        self.messages.append(
            {"role": self.language_vocab["document"], "content": content}
        )

    def get_system_prompt(self):
        return self.default_message_template.format(
            role=self.language_vocab["system"], content=self.default_system_prompt
        )

    def get_prompt(
        self,
        with_system: bool = True,
        delete_documents: bool = False,
    ):
        final_text = ""
        for message in self.messages:
            if message["role"] == "system" and not with_system:
                continue
            final_text += self.default_message_template.format(**message)
        final_text += self.default_response_template.format(
            role=self.language_vocab["assistant"]
        )
        if delete_documents:
            self.messages = [
                message
                for message in self.messages
                if not message["role"] == self.language_vocab["document"]
            ]
        if self.is_tokenizer:
            tokens = self.model_or_tokenizer(
                final_text.strip().encode("utf-8"), special=False
            )
        else:
            tokens = self.model_or_tokenizer.tokenize(
                final_text.strip().encode("utf-8"), special=False
            )
        return tokens
