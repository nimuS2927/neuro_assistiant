import json
import logging
import os
from pathlib import Path
from typing import Sequence

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core import PromptTemplate
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core import Settings
from core_config import c_basic
from indexes import in_helper

from utils.key_extrator import keyextractor

logger = logging.getLogger(os.path.basename(__file__))


class EngineHelper:
    def __init__(
        self,
        path_to_config_file: Path = None,
        query_engine_str: str = None,
    ):
        self.path_to_config_file: Path = path_to_config_file or Path.joinpath(
            c_basic.project_dir, "index_config.json"
        )
        if not self.path_to_config_file.exists():
            raise FileNotFoundError("Файл index_config.json не обнаружен")
        self.query_engine = None
        if not self.query_engine:
            self.get_sub_question_query_engine()

    def get_sub_question_query_engine(self):
        self.query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=self.get_query_engine_tools()
        )
        logger.info("SubQuestionQueryEngine успешно инициализирован")

    def get_query_engine_tools(self):
        query_engine_tools = []

        with open(self.path_to_config_file, "r", encoding="utf-8") as f:
            config_ = json.load(f)

        for tool in config_:
            if not in_helper.get_indexes.get(tool["folder_name"]):
                logger.info("Индекс: %s, не обнаружен.", tool["folder_name"])
                continue

            query_engine_tools.append(
                QueryEngineTool(
                    query_engine=in_helper.get_indexes.get(
                        tool["folder_name"]
                    ).as_query_engine(similarity_top_k=3),
                    metadata=ToolMetadata(
                        name=tool["name"],
                        description=tool["description"],
                    ),
                )
            )
            logger.info("Индекс: %s, добавлен в QueryEngineTools", tool["folder_name"])

        return query_engine_tools


# eng_helper = EngineHelper()

qa_prompt = PromptTemplate(
    "Контекстная информация ниже.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Используя контекстную информацию, а не предыдущие знания,"
    "ответь на вопрос.\n"
    "Вопрос: {query_str}\n"
    "Ответ: "
)


class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    response_synthesizer: BaseSynthesizer
    llm: LlamaCPP
    qa_prompt: PromptTemplate
    retrievers: dict[str, BaseRetriever]
    categories: dict[str, str]

    def custom_query(self, query_str: str):
        retriever_names_with_scores = keyextractor.cosine_scores(
            query_str=query_str, categories_=self.categories, verbose=True
        )
        names = [name for name, score in retriever_names_with_scores if score >= 0.5]
        if not names:
            names = [name for name, score in retriever_names_with_scores[:3]]
        nodes_list = []
        for name in names:
            nodes = self.retrievers[name].retrieve(query_str)
            nodes_list.extend(nodes)
        # Тут можно написать проверку на размер контекста и его ограничение
        # context_window = Settings.context_window
        context_str = "\n\n".join([n.node.get_content() for n in nodes_list])
        logger.info("Сформированный контекст:\n%s", context_str)
        response = self.llm.complete(
            qa_prompt.format(context_str=context_str, query_str=query_str),
        )

        return str(response)


synthesizer = get_response_synthesizer(text_qa_template=qa_prompt)
retrievers, categories = in_helper.get_retrievers_and_categories()
query_engine = RAGStringQueryEngine(
    retrievers=retrievers,
    categories=categories,
    response_synthesizer=synthesizer,
    llm=Settings.llm,
    qa_prompt=qa_prompt,
)
response_ = query_engine.query(
    "как правила для разработчиков при использовании ваших API?"
)

print(str(response_))
