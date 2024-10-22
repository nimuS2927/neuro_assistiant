from typing import Dict

MODELS_CONFIG = {
    "IlyaGusev/saiga_mistral_7b": {
        "default_message_template": "<s>{role}\n{content}</s>",
        "default_response_template": "<s>bot\n",
        "default_system_prompt": """
                                Ты — Сайга, русскоязычный автоматический ассистент.
                                 Ты разговариваешь с людьми и помогаешь им.
                                """,
    },
    "IlyaGusev/saiga_mistral_7b_gguf": {
        "default_message_template": "<s>{role}\n{content}</s>",
        "default_response_template": "<s>bot\n",
        "default_system_prompt": """Ты — Сайга, русскоязычный автоматический ассистент.
Ты разговариваешь с людьми и помогаешь им.
Думай шаг за шагом, чтобы быть уверенным в своем ответе.""",
    },
    "ai-forever/FRED-T5-1.7B": {
        "default_message_template": "<s>{role}\n{content}</s>",
        "default_response_template": "<s>bot\n",
        "default_system_prompt": "Ты — персональный ассистент помощник, который называется НейроЛаборант отвечающий "
        "на вопросы пользователя,",
    },
}


# Думай шаг за шагом и используй Documents для точного ответа на вопросы пользователя.
# Если ты не нашел ответа в документах, то прямо об этом и скажи.


class Conversation:
    def __init__(
        self,
        model_name: str,
        config: Dict = None,
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
        self.messages = [{"role": "system:", "content": self.default_system_prompt}]

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_bot_message(self, message):
        self.messages.append({"role": "assistant", "content": message})

    def add_documents(self, content):
        self.messages.append({"role": "Documents", "content": content})

    def get_system_prompt(self):
        return self.default_message_template.format(
            role="system", content=self.default_system_prompt
        )

    def get_prompt(self):
        final_text = ""
        len_messages = len(self.messages)
        documents_context = None
        if self.messages[-1].get("role") == "Documents":
            last = len_messages - 1
            documents_context = self.messages.pop(-1)
        else:
            last = len_messages

        for i, message in enumerate(self.messages[:last]):
            if i == last - 1 and documents_context:
                content = message.get("content")
                content += f"\nDocuments\n{documents_context.get('content')}"
                message_text = self.default_message_template.format(
                    role="User", content=content
                )
                final_text += message_text
                # else:
                #     content = message.get("content")
                #     content += f"\nDocuments\nОтсутствуют"
                #     message_text = self.default_message_template.format(
                #         role="User", content=content
                #     )
                #     final_text += message_text
            else:
                message_text = self.default_message_template.format(**message)
                final_text += message_text
        final_text += self.default_response_template
        return final_text.strip()
