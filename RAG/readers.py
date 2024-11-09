import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from llama_index.core.schema import Document
from tqdm import tqdm

from exceptions import InvalidFileFormatError

from utils.key_extrator import keyextractor
from rusenttokenize import ru_sent_tokenize


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
    def create_documents(self, chunks: list[dict[str, str | dict[str, Any]]]):
        """Создает документ из полученного чанков"""
        pass

    @abstractmethod
    def create_document(self, chunk):
        """Создает документ из полученного чанков"""
        pass

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
        self.key_extractor = keyextractor

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

    def create_documents(self, chunks: list[dict]):
        """Создает документ из полученного чанков"""
        return [
            Document(text=chunk["text"], metadata=chunk["metadata"]) for chunk in chunks
        ]

    def create_document(self, chunk: dict):
        """Создает документ из полученного чанков"""
        return Document(text=chunk["text"], metadata=chunk["metadata"])

    def append_chunk(self, text, return_doc, header_1=None, header_2=None):
        text = f"{header_1}\n{header_2}\n{text}".strip().replace("\n\n", "\n")
        keywords = self.key_extractor.get_keywords_from_document(
            document=text,
            keyphrase_ngram_range=(1, 1),
            use_maxsum=True,
        )
        if return_doc:
            return self.create_document(
                {"text": text, "metadata": {"keywords": keywords}}
            )
        return {"text": text, "metadata": {"keywords": keywords}}

    @staticmethod
    def split_text(path_to_file):
        """Делит текст по заголовкам"""
        header_1_pattern = re.compile(r"^=+\n*")
        header_2_pattern = re.compile(r"^-+\n*")
        header_1, header_2, text, buffer = None, None, "", ""

        with open(path_to_file, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                line = line.strip()
                # print(f"Строка №{i+1}")
                if line == "\n":
                    continue
                line = line.replace(r"\.", ".")
                if not buffer:
                    buffer = line
                    continue
                if re.match(header_1_pattern, line):
                    if text:
                        yield {
                            "text": text.strip(),
                            "header_1": header_1,
                            "header_2": header_2,
                            # "len": len(text.strip()),
                        }
                        text = ""
                    header_1, header_2 = buffer, None
                    buffer = ""
                elif re.match(header_2_pattern, line):
                    if text:
                        yield {
                            "text": text.strip(),
                            "header_1": header_1,
                            "header_2": header_2,
                            # "len": len(text.strip()),
                        }

                        text = ""
                    header_2 = buffer
                    buffer = ""
                else:
                    text += buffer
                    buffer = line
        if text:
            text += buffer
            yield {
                "text": text.strip(),
                "header_1": header_1,
                "header_2": header_2,
            }

    @staticmethod
    def split_paragraphs(text):
        """Делит текст на параграфы и абзацы"""
        # 1. Разделение текста на параграфы по символу \n\n
        paragraphs = text.split("\n\n")

        # 2. Обработка каждого параграфа, разделяя его на абзацы по символу \n
        for paragraph in paragraphs:
            # 3. Разделяем параграф на абзацы
            lines = paragraph.split("\n")

            # Обработка абзацев
            new_paragraph = []
            for line in lines:
                # Если текущий абзац меньше 150 символов
                if len(" ".join(new_paragraph)) < 150:
                    new_paragraph.append(line)  # Объединяем с предыдущим
                else:
                    # Сохраняем новый абзац в результат, если он не пуст
                    if new_paragraph:
                        yield ". ".join(new_paragraph).strip()
                    new_paragraph = [line]  # Начинаем новый абзац

            # Добавляем последний абзац, если он не пуст
            if new_paragraph:
                yield ". ".join(new_paragraph).strip()

    def split_sentence(self, chunk: dict):
        """Делит предложение на части по словам добавляя overlap если он задан"""
        text = chunk["text"]
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text) and text[end] != " ":
                pos_space = text[start:end].rfind(" ")
                if pos_space != -1:
                    pos_space += start
                    if abs(pos_space - end) > 100:
                        pos_space = end
                    yield text[start:pos_space].strip()
                    if self.overlap_size > 0:
                        next_start = text[
                            start : (pos_space - self.overlap_size)
                        ].rfind(" ")
                        if abs((pos_space - self.overlap_size) - next_start) > 50:
                            next_start = pos_space - self.overlap_size
                        start = next_start + start if next_start != -1 else end
                    else:
                        start = pos_space
                else:
                    yield text[start:end].strip()
                    start = end
            else:
                yield text[start:end].strip()
                start = end

    def split_chunk(self, chunk: dict, return_doc: bool):
        text = " ".join(ru_sent_tokenize(chunk["text"]))
        if len(text) < self.chunk_size:
            yield self.append_chunk(
                text.strip(),
                return_doc,
                chunk["header_1"],
                chunk["header_2"],
            )
        else:
            if text:
                del text  # Очищаем память, хотя далее и присваиваем переменной пустое значение. Явное лучше не явного
            sentences = ru_sent_tokenize(chunk["text"])
            text = ""
            count_sent = 0
            for i, sentence in enumerate(sentences):
                if len(sentence) > self.chunk_size:
                    if text:
                        yield self.append_chunk(
                            text.strip(),
                            chunk["header_1"],
                            chunk["header_2"],
                        )
                        text = ""
                        count_sent = 0
                    for sentence_chunk in self.split_sentence(
                        {
                            "text": sentence,
                            "header_1": chunk["header_1"],
                            "header_2": chunk["header_2"],
                        }
                    ):
                        yield self.append_chunk(
                            sentence_chunk.strip(),
                            chunk["header_1"],
                            chunk["header_2"],
                        )
                elif len(text) + len(sentence) > self.chunk_size:
                    yield self.append_chunk(
                        text.strip(),
                        chunk["header_1"],
                        chunk["header_2"],
                    )
                    # Собираем overlap
                    if len(text) > self.overlap_size:
                        if count_sent == 1:
                            count_sent = 0
                            text = sentence
                        else:
                            overlap_sentences = []
                            step = 1
                            while (
                                len(
                                    " ".join(overlap_sentences)
                                    + " "
                                    + sentences[i - step - 1]
                                )
                                < self.overlap_size
                                and i - step > 0
                            ):
                                overlap_sentences.insert(0, sentences[i - step])
                                step += 1
                            if i - step - 1 >= 0:
                                if abs(
                                    len(" ".join(overlap_sentences)) - self.overlap_size
                                ) > abs(
                                    len(
                                        " ".join(overlap_sentences)
                                        + " "
                                        + sentences[i - step - 1]
                                    )
                                    - self.overlap_size
                                ):
                                    overlap_sentences.insert(0, sentences[i - step - 1])
                            text = " ".join(overlap_sentences) + " " + sentence
                    else:
                        text = sentence
                else:
                    text += " " + sentence
                    count_sent += 1
            if text:
                yield self.append_chunk(
                    text.strip(),
                    chunk["header_1"],
                    chunk["header_2"],
                )

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
        chunks = []
        for i, chunk in enumerate(self.split_text(path_to_file)):
            # print(f"chunk №{i}")
            for j, paragraph in enumerate(self.split_paragraphs(chunk["text"])):
                # print(f"paragraph №{j}")
                sub_chunk = {
                    "text": paragraph,
                    "header_1": chunk["header_1"],
                    "header_2": chunk["header_2"],
                }
                for sub2_chunk in self.split_chunk(sub_chunk, return_doc):
                    chunks.append(sub2_chunk)

        return chunks

    # region Старый код, большие затраты памяти!!! (сохранено только для примера как было и чего можно достичь)
    # @staticmethod
    # def split_text(path_to_file):
    #     """Делит текст по заголовкам"""
    #     chunks = []
    #     text = ""
    #     header_1_pattern = re.compile(r"^=+\n*")
    #     header_2_pattern = re.compile(r"^-+\n*")
    #     header_1 = None
    #     header_2 = None
    #
    #     with open(path_to_file, "r", encoding="utf-8") as file:
    #         buffer = ""
    #         for line in file:
    #             line = line.replace(r"\.", ".")
    #             if not buffer:
    #                 buffer = line
    #                 continue
    #             if re.match(header_1_pattern, line):
    #                 if text:
    #                     chunks.append(
    #                         {
    #                             "text": text.strip(),
    #                             "header_1": header_1,
    #                             "header_2": header_2,
    #                             "len": len(text.strip()),
    #                         }
    #                     )
    #                     text = ""
    #                 header_1 = buffer
    #                 buffer = ""
    #                 header_2 = None
    #             elif re.match(header_2_pattern, line):
    #                 if text:
    #                     chunks.append(
    #                         {
    #                             "text": text.strip(),
    #                             "header_1": header_1,
    #                             "header_2": header_2,
    #                             "len": len(text.strip()),
    #                         }
    #                     )
    #                     text = ""
    #                 header_2 = buffer
    #                 buffer = ""
    #             else:
    #                 text += buffer
    #                 buffer = line
    #     return chunks
    #
    # @staticmethod
    # def split_paragraphs(text):
    #     """Делит текст на параграфы и абзацы"""
    #     # 1. Разделение текста на параграфы по символу \n\n
    #     paragraphs = text.split("\n\n")
    #
    #     # 2. Обработка каждого параграфа, разделяя его на абзацы по символу \n
    #     result = []
    #     for paragraph in paragraphs:
    #         # 3. Разделяем параграф на абзацы
    #         lines = paragraph.split("\n")
    #
    #         # Обработка абзацев
    #         new_paragraph = ""
    #         for line in lines:
    #             # Если длина нового абзаца + текущая строка меньше 150 символов
    #             if len(new_paragraph) + len(line) < 150:
    #                 new_paragraph += line + "\n"  # Объединяем с предыдущим
    #             else:
    #                 # Сохраняем новый абзац в результат, если он не пуст
    #                 if new_paragraph:
    #                     result.append(new_paragraph.strip())
    #                 new_paragraph = line + " "  # Начинаем новый абзац
    #
    #         # Добавляем последний абзац, если он не пуст
    #         if new_paragraph:
    #             result.append(new_paragraph.strip())
    #
    #     return result
    #
    # def split_sentence(self, chunk: dict):
    #     """Делит предложение на части по словам добавляя overlap если он задан"""
    #     text = chunk["text"]
    #     start = 0
    #     texts = []
    #     while start < len(text):
    #         end = min(start + self.chunk_size, len(text))
    #         if end < len(text) and text[end] != " ":
    #             pos_space = text[start:end].rfind(" ")
    #             if pos_space != -1:
    #                 texts.append(text[start:pos_space].strip())
    #                 if self.overlap_size > 0:
    #                     next_start = text[
    #                                  start: (pos_space - self.overlap_size)
    #                                  ].rfind(" ")
    #                     start = next_start if next_start != -1 else end
    #                 else:
    #                     start = pos_space
    #             else:
    #                 texts.append(text[start:end].strip())
    #                 start = end
    #         else:
    #             texts.append(text[start:end].strip())
    #             start = end
    #     return [
    #         {
    #             "text": t,
    #             "header_1": chunk["header_1"],
    #             "header_2": chunk["header_2"],
    #             "len": len(t),
    #         }
    #         for t in texts
    #     ]
    #
    # def split_chunk(self, chunk: dict):
    #     text = " ".join(ru_sent_tokenize(chunk["text"]))
    #     if len(text) < self.chunk_size:
    #         return [
    #             {
    #                 "text": text.strip(),
    #                 "header_1": chunk["header_1"],
    #                 "header_2": chunk["header_2"],
    #                 "len": len(text.strip()),
    #             }
    #         ]
    #     sentences = ru_sent_tokenize(chunk["text"])
    #     chunks = []
    #     text = ""
    #     i = 0
    #     count_sent = 0
    #     while i < len(sentences):
    #         if len(sentences[i]) > self.chunk_size:
    #             if text:
    #                 chunks.append(
    #                     {
    #                         "text": text.strip(),
    #                         "header_1": chunk["header_1"],
    #                         "header_2": chunk["header_2"],
    #                         "len": len(text.strip()),
    #                     }
    #                 )
    #                 text = ""
    #                 count_sent = 0
    #             new_chunks = self.split_sentence(
    #                 {
    #                     "text": sentences[i],
    #                     "header_1": chunk["header_1"],
    #                     "header_2": chunk["header_2"],
    #                     "len": len(sentences[i]),
    #                 }
    #             )
    #             chunks.extend(new_chunks)
    #             del new_chunks
    #             i += 1
    #         elif len(text) + len(sentences[i]) > self.chunk_size:
    #             chunks.append(
    #                 {
    #                     "text": text.strip(),
    #                     "header_1": chunk["header_1"],
    #                     "header_2": chunk["header_2"],
    #                     "len": len(text.strip()),
    #                 }
    #             )
    #             # Собираем overlap
    #             if len(text) > self.overlap_size:
    #                 if count_sent == 2:
    #                     overlap = sentences[i - 1]
    #                     count_sent = 1
    #                     text = overlap
    #                 elif count_sent > 2:
    #                     step = 1
    #                     overlap = sentences[i - step]
    #                     count_sent = 1
    #                     while (
    #                             len(overlap) + len((sentences[i - step - 1]))
    #                             < self.overlap_size
    #                     ):
    #                         step += 1
    #                         overlap = sentences[i - step] + " " + overlap
    #                         count_sent += 1
    #                     if abs(len(overlap) - self.overlap_size) > abs(
    #                             len(overlap)
    #                             + len((sentences[i - step - 1]))
    #                             - self.overlap_size
    #                     ):
    #                         step += 1
    #                         overlap = sentences[i - step] + " " + overlap
    #                         count_sent += 1
    #                     text = overlap
    #                 else:  # Случай когда в тексте только 1 предложение, overlap отсутствует
    #                     count_sent = 0
    #                     text = ""
    #
    #         else:
    #             sentence = sentences[i]
    #             text += " " + sentence
    #             text = text.strip()
    #             count_sent += 1
    #             i += 1
    #     if text:
    #         chunks.append(
    #             {
    #                 "text": text.strip(),
    #                 "header_1": chunk["header_1"],
    #                 "header_2": chunk["header_2"],
    #                 "len": len(text.strip()),
    #             }
    #         )
    #     return chunks
    #
    # def read_file(self, path_to_file: Path, return_doc: bool = True):
    #     """
    #     Читает файл и обрабатывает его
    #     """
    #     if (
    #             not path_to_file.is_file()
    #             or path_to_file.suffix not in self.expected_formats
    #     ):
    #         raise InvalidFileFormatError(
    #             file_path=path_to_file, expected_formats=self.expected_formats
    #         )
    #     chunks = self.split_text(path_to_file)
    #     new_chunks = []
    #     for chunk in chunks:
    #         new_texts = self.split_paragraphs(chunk["text"])
    #         for t in new_texts:
    #             new_chunks.append(
    #                 {
    #                     "text": t,
    #                     "header_1": chunk["header_1"],
    #                     "header_2": chunk["header_2"],
    #                     "len": len(t),
    #                 }
    #             )
    #     chunks = []
    #     for chunk in new_chunks:
    #         splitted_chunks = self.split_chunk(chunk)
    #         chunks.extend(splitted_chunks)
    #
    #     result_chunks = []
    #     headers_1_lvl = []
    #     for chunk in chunks:
    #         self.append_chunk(
    #             result_chunks, chunk["text"], chunk["header_1"], chunk["header_2"]
    #         )
    #         if chunk["header_1"]:
    #             headers_1_lvl.append(chunk["header_1"])
    #
    #     if return_doc:
    #         return self.create_document(result_chunks)
    #     return result_chunks
    # endregion


# mdr = MarkDownReader(chunk_size=500, overlap_size=100, expected_formats=[".md"])

# docs = mdr.read_file(
#     Path(
#         r"C:\Users\vsumi\PycharmProjects\neuro_assistiant\library\files\yandex_files\Кинопоиск\Пользовательское соглашение сайта «Кинопоиск»_kinopoisk_termsofuse.md"
#     )
# )
# print(len(docs))
