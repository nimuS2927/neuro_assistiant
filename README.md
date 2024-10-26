Для работы проекта необходимо установить в ручную через pip библиотеки pytorch с поддержкой куда

```bash  
 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

Ссылка для скачивания chromedriver https://storage.googleapis.com/chrome-for-testing-public/130.0.6723.69/win64/chromedriver-win64.zip

Внеси исправление в фай pymorphy2 .venv\Lib\site-packages\pymorphy2\units\base.py

замени метод getargspec в строке №70:

```python  
args, varargs, kw, default = inspect.getargspec(cls.__init__)
```

на метод getfullargspec и добавь срез в конце [:4]:
```python  
args, varargs, kw, default = inspect.getfullargspec(cls.__init__)[:4]
```