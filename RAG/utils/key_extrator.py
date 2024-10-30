import inspect
import re
from pathlib import Path

import pymorphy2
from huggingface_hub import login
from keybert import KeyBERT
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

from core_config import c_hf, c_basic
import nltk

nltk.download("stopwords")


class KeyExtractor:
    """
    Извлекает ключевые слова из документов и приводит их к Лемме
    """

    _instance = None

    def __new__(cls):
        if not cls._instance:
            instance = super(KeyExtractor, cls).__new__(cls)
            cls._instance = instance
        return cls._instance

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        stop_words=None,
    ):
        if stop_words is None:
            stop_words = list(
                set(stopwords.words("russian") + stopwords.words("english"))
            )
        self.stop_words = stop_words
        login(token=c_hf.token, add_to_git_credential=True)
        self.model_name = model_name
        self.model = KeyBERT(SentenceTransformer(self.model_name))
        self.morph = pymorphy2.MorphAnalyzer()

    def lemmatize_keywords(
        self,
        keywords: list[str] | list[tuple[str, float]],
        include_scores: bool = False,
    ) -> list[str] | list[tuple[str, float]]:
        if include_scores:
            return [
                (self.morph.parse(word)[0].normal_form, score)
                for word, score in keywords
            ]
        else:
            return [self.morph.parse(word)[0].normal_form for word in keywords]

    def get_keywords_from_document(
        self,
        document: str,
        keyphrase_ngram_range: tuple[int, int] = (1, 1),
        stop_words=None,
        top_n: int = 5,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        is_lemmatize: bool = True,
        include_scores: bool = False,
    ) -> list[str] | list[tuple[str, float]]:

        stack = inspect.stack()
        many_documents = (
            False
            if len(stack) == 1
            or (len(stack) > 1 and stack[1].function != "get_keywords_from_documents")
            else True
        )

        keywords: list[tuple[str, float]] = self.model.extract_keywords(
            docs=document.replace("_", " "),
            stop_words=stop_words if stop_words else self.stop_words,
            keyphrase_ngram_range=keyphrase_ngram_range,
            top_n=top_n,
            use_maxsum=use_maxsum,
            use_mmr=use_mmr,
            diversity=diversity,
        )
        keywords = [
            (re.sub(r"[^a-zA-Zа-яА-Я]", " ", keyword).strip(), score)
            for keyword, score in keywords
        ]
        only_keywords: list[str] = [keyword for keyword, score in keywords]
        if include_scores:
            if is_lemmatize:
                return (
                    self.lemmatize_keywords(
                        keywords,
                        include_scores=include_scores,
                    )
                    if many_documents
                    else self.sort_and_delete_duplicate(
                        self.lemmatize_keywords(
                            keywords,
                            include_scores=include_scores,
                        ),
                        include_scores=include_scores,
                    )
                )
            else:
                return (
                    keywords
                    if many_documents
                    else self.sort_and_delete_duplicate(
                        keywords,
                        include_scores=include_scores,
                    )
                )
        else:
            if is_lemmatize:
                return (
                    self.lemmatize_keywords(only_keywords)
                    if many_documents
                    else self.sort_and_delete_duplicate(
                        self.lemmatize_keywords(only_keywords)
                    )
                )
            else:
                return (
                    only_keywords
                    if many_documents
                    else self.sort_and_delete_duplicate(only_keywords)
                )

    @staticmethod
    def sort_and_delete_duplicate(
        keywords: list[str] | list[tuple[str, float]],
        include_scores: bool = False,
    ) -> list[str] | list[tuple[str, float]]:
        if not include_scores:
            return sorted(set(keywords))
        else:
            words_dict = {}
            for word, score in keywords:
                if word not in words_dict:
                    words_dict[word] = score
                else:
                    words_dict[word] = (
                        score if score > words_dict[word] else words_dict[word]
                    )
            return sorted(
                [(word, score) for word, score in words_dict.items()],
                key=lambda x: x[1],
                reverse=True,
            )

    def get_keywords_from_documents(
        self,
        documents: list[str],
        keyphrase_ngram_range: tuple[int, int] = (1, 1),
        stop_words=None,
        top_n: int = 5,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        is_lemmatize: bool = True,
        include_scores: bool = False,
    ) -> list[str] | list[tuple[str, float]]:
        keywords_all: list[str] | list[tuple[str, float]] = []
        for doc in documents:
            keywords = self.get_keywords_from_document(
                document=doc,
                stop_words=stop_words,
                keyphrase_ngram_range=keyphrase_ngram_range,
                top_n=top_n,
                use_maxsum=use_maxsum,
                use_mmr=use_mmr,
                diversity=diversity,
                is_lemmatize=is_lemmatize,
                include_scores=include_scores,
            )
            keywords_all.extend(keywords)
        return self.sort_and_delete_duplicate(
            keywords_all,
            include_scores=include_scores,
        )


# with open(
#     Path.joinpath(
#         c_basic.path_to_dataset_dir,
#         "Авто.ру",
#         "Условия предоставления опции сервиса Auto.ru по информационн_autoru_bot_tender.md",
#     ),
#     "r",
#     encoding="utf-8",
# ) as f:
#     f_str = f.read()

# key_extractor = KeyExtractor(
#     top_n=10,
#     is_lemmatize=False,
#     use_mmr=False,
#     use_maxsum=True,
#     include_scores=True,
#     diversity=0.2,
# )
# words = key_extractor.get_keywords_from_document(f_str)
# print(words)
