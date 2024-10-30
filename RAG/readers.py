import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from llama_index.core.schema import Document
from exceptions import InvalidFileFormatError

from RAG import KeyExtractor
from rusenttokenize import ru_sent_tokenize


# import nltk
#
# nltk.download("punkt_tab")


class BaseReader(ABC):

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Возвращает размер чанков на которые делится документ"""
        pass

    @property
    @abstractmethod
    def overlap_size(self) -> int:
        """
        Возвращает размер перекрытия между чанками, применяется только если чанки относятся к одному абзацу,
        также число является не абсолютным, а приблизительным размером перекрытия, т.к. при перекрытии используются
        целые предложения
        """
        pass

    @property
    @abstractmethod
    def expected_formats(self) -> list[str]:
        """Возвращает поддерживаемые форматы файлов"""
        pass

    @abstractmethod
    def create_document(self, chunks: list[dict[str, str | dict[str, Any]]]):
        """Создает документ из полученного чанков"""

    @abstractmethod
    def read_file(self, path_to_file: Path):
        """Читает файл и обрабатывает его"""
        pass


class MarkDownReader(BaseReader):
    def __init__(
        self,
        chunk_size: int,
        overlap_size: int,
        expected_formats: list[str],
    ):
        self.__chunk_size = chunk_size
        self.__overlap_size = overlap_size
        self.__expected_formats = expected_formats
        self.key_extractor = KeyExtractor()

    @property
    def chunk_size(self) -> int:
        """Возвращает размер чанков на которые делится документ"""
        return self.__chunk_size

    @property
    def overlap_size(self) -> int:
        """
        Возвращает размер перекрытия между чанками, также число является не абсолютным,
        а приблизительным размером перекрытия, применяется в 2-х случаях:
        1. Если чанки относятся к одному абзацу, при перекрытии используются целые предложения.
        2. При делении предложения больше self.chunk_size, при перекрытии используются целые слова.
        """
        return self.__overlap_size

    @property
    def expected_formats(self) -> list[str]:
        """Возвращает поддерживаемые форматы файлов"""
        return self.__expected_formats

    def create_document(self, chunks):
        """Создает документ из полученного чанков"""
        return [
            Document(text=chunk["text"], metadata=chunk["metadata"]) for chunk in chunks
        ]

    def append_chunk(self, chunks, text, header_1=None, header_2=None):
        text = f"{header_1}\n{header_2}\n{text}".strip()
        keywords = self.key_extractor.get_keywords_from_document(
            document=text.replace("\n\n", "\n"),
            keyphrase_ngram_range=(1, 1),
            use_maxsum=True,
        )
        chunks.append({"text": text, "metadata": {"keywords": keywords}})
        return chunks

    @staticmethod
    def split_text(path_to_file):
        """Делит текст по заголовкам"""
        chunks = []
        text = ""
        header_1_pattern = re.compile(r"^=+\n*")
        header_2_pattern = re.compile(r"^-+\n*")
        header_1 = None
        header_2 = None

        with open(path_to_file, "r", encoding="utf-8") as file:
            buffer = ""
            for line in file:
                line = line.replace(r"\.", ".")
                if not buffer:
                    buffer = line
                    continue
                if re.match(header_1_pattern, line):
                    if text:
                        chunks.append(
                            {
                                "text": text.strip(),
                                "header_1": header_1,
                                "header_2": header_2,
                                "len": len(text.strip()),
                            }
                        )
                        text = ""
                    header_1 = buffer
                    buffer = ""
                    header_2 = None
                elif re.match(header_2_pattern, line):
                    if text:
                        chunks.append(
                            {
                                "text": text.strip(),
                                "header_1": header_1,
                                "header_2": header_2,
                                "len": len(text.strip()),
                            }
                        )
                        text = ""
                    header_2 = buffer
                    buffer = ""
                else:
                    text += buffer
                    buffer = line
        return chunks

    @staticmethod
    def split_paragraphs(text):
        """Делит текст на параграфы и абзацы"""
        # 1. Разделение текста на параграфы по символу \n\n
        paragraphs = text.split("\n\n")

        # 2. Обработка каждого параграфа, разделяя его на абзацы по символу \n
        result = []
        for paragraph in paragraphs:
            # 3. Разделяем параграф на абзацы
            lines = paragraph.split("\n")

            # Обработка абзацев
            new_paragraph = ""
            for line in lines:
                # Если длина нового абзаца + текущая строка меньше 150 символов
                if len(new_paragraph) + len(line) < 150:
                    new_paragraph += line + "\n"  # Объединяем с предыдущим
                else:
                    # Сохраняем новый абзац в результат, если он не пуст
                    if new_paragraph:
                        result.append(new_paragraph.strip())
                    new_paragraph = line + " "  # Начинаем новый абзац

            # Добавляем последний абзац, если он не пуст
            if new_paragraph:
                result.append(new_paragraph.strip())

        return result

    def split_sentence(self, chunk: dict):
        """Делит предложение на части по словам добавляя overlap если он задан"""
        text = chunk["text"]
        start = 0
        texts = []
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text) and text[end] != " ":
                pos_space = text[start:end].rfind(" ")
                if pos_space != -1:
                    texts.append(text[start:pos_space].strip())
                    if self.overlap_size > 0:
                        next_start = text[
                            start : (pos_space - self.overlap_size)
                        ].rfind(" ")
                        start = next_start if next_start != -1 else end
                    else:
                        start = pos_space
                else:
                    texts.append(text[start:end].strip())
                    start = end
            else:
                texts.append(text[start:end].strip())
                start = end
        return [
            {
                "text": t,
                "header_1": chunk["header_1"],
                "header_2": chunk["header_2"],
                "len": len(t),
            }
            for t in texts
        ]

    def split_chunk(self, chunk: dict):
        text = " ".join(ru_sent_tokenize(chunk["text"]))
        if len(text) < self.chunk_size:
            return [
                {
                    "text": text.strip(),
                    "header_1": chunk["header_1"],
                    "header_2": chunk["header_2"],
                    "len": len(text.strip()),
                }
            ]
        sentences = ru_sent_tokenize(chunk["text"])
        chunks = []
        text = ""
        i = 0
        count_sent = 0
        while i < len(sentences):
            if len(sentences[i]) > self.chunk_size:
                if text:
                    chunks.append(
                        {
                            "text": text.strip(),
                            "header_1": chunk["header_1"],
                            "header_2": chunk["header_2"],
                            "len": len(text.strip()),
                        }
                    )
                    text = ""
                    count_sent = 0
                new_chunks = self.split_sentence(
                    {
                        "text": sentences[i],
                        "header_1": chunk["header_1"],
                        "header_2": chunk["header_2"],
                        "len": len(sentences[i]),
                    }
                )
                chunks.extend(new_chunks)
                del new_chunks
                i += 1
            elif len(text) + len(sentences[i]) > self.chunk_size:
                chunks.append(
                    {
                        "text": text.strip(),
                        "header_1": chunk["header_1"],
                        "header_2": chunk["header_2"],
                        "len": len(text.strip()),
                    }
                )
                # Собираем overlap
                if len(text) > self.overlap_size:
                    if count_sent == 2:
                        overlap = sentences[i - 1]
                        count_sent = 1
                        text = overlap
                    elif count_sent > 2:
                        step = 1
                        overlap = sentences[i - step]
                        count_sent = 1
                        while (
                            len(overlap) + len((sentences[i - step - 1]))
                            < self.overlap_size
                        ):
                            step += 1
                            overlap = sentences[i - step] + " " + overlap
                            count_sent += 1
                        if abs(len(overlap) - self.overlap_size) > abs(
                            len(overlap)
                            + len((sentences[i - step - 1]))
                            - self.overlap_size
                        ):
                            step += 1
                            overlap = sentences[i - step] + " " + overlap
                            count_sent += 1
                        text = overlap
                    else:  # Случай когда в тексте только 1 предложение, overlap отсутствует
                        count_sent = 0
                        text = ""

            else:
                sentence = sentences[i]
                text += " " + sentence
                text = text.strip()
                count_sent += 1
                i += 1
        if text:
            chunks.append(
                {
                    "text": text.strip(),
                    "header_1": chunk["header_1"],
                    "header_2": chunk["header_2"],
                    "len": len(text.strip()),
                }
            )
        return chunks

    def read_file(self, path_to_file: Path, return_doc: bool = True):
        """
        Читает файл и обрабатывает его
        """
        if (
            not path_to_file.is_file()
            or path_to_file.suffix not in self.expected_formats
        ):
            raise InvalidFileFormatError(
                file_path=path_to_file, expected_formats=self.expected_formats
            )
        chunks = self.split_text(path_to_file)
        new_chunks = []
        for chunk in chunks:
            new_texts = self.split_paragraphs(chunk["text"])
            for t in new_texts:
                new_chunks.append(
                    {
                        "text": t,
                        "header_1": chunk["header_1"],
                        "header_2": chunk["header_2"],
                        "len": len(t),
                    }
                )
        chunks = []
        for chunk in new_chunks:
            splitted_chunks = self.split_chunk(chunk)
            chunks.extend(splitted_chunks)
        if return_doc:
            return self.create_document(chunks)
        return chunks


# mdr = MarkDownReader(chunk_size=500, overlap_size=100, expected_formats=[".md"])
#
# docs = mdr.read_file(
#     Path(
#         r"C:\Users\vsumi\PycharmProjects\neuro_assistiant\library\files\yandex_files\Дистрибуция продуктов Яндекса\Правила использования Партнерского интерфейса_distribution_interface.md"
#     )
# )
