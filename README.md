Для работы проекта необходимо выполнить следующие действия

1. Установить в ручную через pip библиотеки pytorch с поддержкой куда
```bash  
 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
2. Установить в ручную через pip llama-cpp-python
```bash  
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```
3. Скачать драйвер для работы парсера, ссылка для скачивания chromedriver https://storage.googleapis.com/chrome-for-testing-public/130.0.6723.69/win64/chromedriver-win64.zip

4. Внеси исправление в файл pymorphy2 .venv\Lib\site-packages\pymorphy2\units\base.py

  замени метод getargspec в строке №70:

  ```python  
  args, varargs, kw, default = inspect.getargspec(cls.__init__)
  ```

  на метод getfullargspec и добавь срез в конце [:4]:
  ```python  
  args, varargs, kw, default = inspect.getfullargspec(cls.__init__)[:4]
  ```

  Без этих изменений функции библиотека не работает, поэтому леммантизация ключевых слов будет не доступна
