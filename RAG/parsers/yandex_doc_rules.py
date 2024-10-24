import json
import os
import random
from pathlib import Path
import re

import requests
from bs4 import BeautifulSoup
from core_config import c_basic
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
from markdownify import markdownify as md
from tqdm import tqdm
import time


def get_all_main_links(output_file: str) -> dict[str, list[tuple[str, str]]]:
    if os.path.exists(output_file_for_links):
        print("Найден файл с ссылками")
        with open(output_file_for_links, "r", encoding="utf-8") as f:
            sections = json.load(f)
        return sections

    # URL страницы
    url = "https://yandex.ru/legal/"

    # Отправка запроса для получения содержимого страницы
    response = requests.get(url)

    # Создаем структуру для хранения ссылок по группам
    sections = {}

    # Проверка успешности запроса
    try:
        # Парсинг содержимого страницы с использованием BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Находим все группы ссылок
        groups = soup.find_all("div", class_="groups__group")

        # Проходим по каждой группе
        for group in groups:
            # Извлекаем заголовок группы
            group_title = group.find("div", class_="groups__title").text.strip()

            # Извлекаем все блоки, содержащие ссылки
            groups_links = group.find_all("div", class_="groups__link")

            # Извлекаем ссылки из каждого блока
            links = []
            for link_block in groups_links:
                link_tag = link_block.find("a", href=True)  # Находим <a> внутри блока
                if link_tag:
                    link_text = link_tag.text.strip()  # Текст ссылки
                    link_href = f"https://yandex.ru{link_tag['href']}"  # URL ссылки
                    links.append((link_text, link_href))

            # Сохраняем ссылки под названием группы
            sections[group_title] = links

        # # Вывод результатов
        # for section, links in sections.items():
        #     print(f"\n{section}:")
        #     for name, href in links:
        #         print(f"  {name}: {href}")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sections, f, ensure_ascii=False, indent=4)

        return sections
    # Обработка ошибок с HTTP
    except requests.exceptions.HTTPError as http_err:
        raise SystemExit(f"HTTP ошибка: {http_err}")
    # Обработка других ошибок, например сетевых
    except requests.exceptions.RequestException as err:
        raise SystemExit(f"Ошибка запроса: {err}")


def initialize_driver():
    """
    Инициализирует ChromeDriver с необходимыми настройками.

    :return: Экземпляр WebDriver
    """
    chrome_options = Options()
    # chrome_options.add_argument(
    #     "--headless"
    # )  # Запуск в фоновом режиме без открытия окна браузера
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    # Делаем наш драйвер похоим на человека
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    )
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument("accept-language=ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--incognito")

    # Путь к chromedriver, если он не в PATH
    path_to_chromedriver = Path.joinpath(
        c_basic.project_dir, "RAG", "parsers", "chromedriver-win64", "chromedriver.exe"
    )
    driver_service = Service(executable_path=str(path_to_chromedriver))

    driver = webdriver.Chrome(service=driver_service, options=chrome_options)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )

    return driver


def html_to_markdown(html_string):
    """
    Преобразует HTML строку в Markdown формат.

    :param html_string: Строка HTML кода
    :return: Строка в формате Markdown
    """
    markdown_text = md(html_string)
    return markdown_text


def zero(html_content):
    return str(html_content)


def first(html_content):
    return str(html_content.next)


def second(html_content):
    return str(html_content.next.next)


ARTICLE_SEARCH = [
    {
        "class": "article",
        "attr": {
            "role": "article",
            "class": "doc-c-article",
            "aria-labelledby": "ariaid-title1",
        },
        "func": zero,
    },
    {
        "class": "main",
        "attr": {
            "class": "dc-doc-page__content",
        },
        "func": zero,
    },
]


def parse_multiple_links(links_data):
    # Инициализируем драйвер
    driver = initialize_driver()

    path_to_yandex_files = Path.joinpath(c_basic.path_to_files, "yandex_files")
    path_to_yandex_files.mkdir(parents=True, exist_ok=True)
    # Парсим все ссылки и создаем из их контента документы в формате MarkDown
    bad_links = []
    for group, links in tqdm(links_data.items(), desc="Обработка групп"):
        # Создаем папку группы
        path_to_group_files = Path.joinpath(path_to_yandex_files, group)
        path_to_group_files.mkdir(parents=True, exist_ok=True)
        for name, href in tqdm(links, desc="Парсинг ссылок"):
            driver.execute_cdp_cmd(
                "Network.setExtraHTTPHeaders", {"headers": {"Referer": href}}
            )
            name = name.replace("/", "-").replace("\\", "-")
            name = name[:60] if len(name) > 80 else name
            invalid_chars_pattern = r'[<>:"/\\|?*\x00-\x1F]'
            full_name = f"{name}_{href.split('/')[-2]}.md"
            full_name = re.sub(invalid_chars_pattern, " ", full_name)
            path_to_file = Path.joinpath(path_to_group_files, full_name)
            if Path.exists(path_to_file):
                print(f"Файл {path_to_file} обнаружен, перейдем к следующему...")
                continue
            try:

                # Открываем страницу
                driver.get(href)
                # Задержка, чтобы страница успела полностью загрузиться
                time.sleep(3)

                # Получаем HTML содержимое страницы
                html_content = driver.page_source
                # Парсим HTML с помощью BeautifulSoup
                soup = BeautifulSoup(html_content, "html.parser")
                is_pdf = soup.find(
                    "embed",
                    {"type": "application/pdf"},
                )
                if is_pdf:
                    pdf_url = driver.current_url
                    response = requests.get(pdf_url)
                    # Проверка, что запрос выполнен успешно
                    if response.status_code == 200:
                        # Сохранение файла на диск
                        path_to_file = Path.joinpath(
                            path_to_group_files, f"{pdf_url.split('/')[-1]}"
                        )
                        with open(path_to_file, "wb") as f:
                            f.write(response.content)
                        print(f"Файл {pdf_url.split('/')[-1]} успешно загружен.")
                        continue
                    else:
                        print(
                            f"Ошибка: Не удалось скачать файл, url {pdf_url} статус {response.status_code}"
                        )
                        continue
                is_404 = soup.find(
                    "h1",
                    {"class": "content__title"},
                )
                if is_404:
                    if is_404.text == "Ошибка 404. Нет такой страницы":
                        bad_links.append(href)
                        continue
                # Ищем определенный div по классу
                markdown_content = None
                for i_dict in ARTICLE_SEARCH:
                    div_article = soup.find(
                        i_dict["class"],
                        i_dict["attr"],
                    )
                    if div_article:
                        # Преобразуем HTML в Markdown
                        markdown_content = html_to_markdown(i_dict["func"](div_article))
                        break
                if not markdown_content:
                    raise ValueError(f"Страница не распарсилась. \n{href}")
                with open(path_to_file, "w", encoding="utf-8") as file_md:
                    file_md.write(markdown_content)
            # Обработка ошибок с HTTP
            except requests.exceptions.HTTPError as http_err:
                raise SystemExit(f"HTTP ошибка: {http_err}")
            # Обработка других ошибок, например сетевых
            except requests.exceptions.RequestException as err:
                raise SystemExit(f"Ошибка запроса: {err}")
            time.sleep(random.uniform(1, 3))
    with open("bad_links.txt", "w", encoding="utf-8") as file_bad_links:
        file_bad_links.write("\n".join(bad_links))


output_file_for_links = "yandex_legal_links.json"
links_dict = get_all_main_links(output_file_for_links)
parse_multiple_links(links_dict)
