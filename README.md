# CNN Classifier

Интерактивный дашборд для обучения и сравнения свёрточных нейронных сетей на датасетах MNIST и CIFAR-10. Всё на чистом Python. Немного фронта (для красоты)
---

## Для работы надо

**1. Клонировать репозиторий**

```bash
git clone https://github.com/YOUR_USERNAME/cnn-classifier.git
cd cnn-classifier
```

**2. Создать виртуальное окружение**

```bash
python -m venv .venv
```

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

Mac / Linux:
```bash
source .venv/bin/activate
```

**3. Установить зависимости**

```bash
pip install -r requirements.txt
```

**4. Запустить дашборд**

```bash
streamlit run app.py
```

Открой браузер: `http://localhost:8501`

---

## Что умеет дашборд

- Выбор датасета: MNIST или CIFAR-10
- Выбор архитектуры: Simple, Medium или Deep CNN
- Настройка гиперпараметров: количество эпох, размер батча
- Live-обучение — графики accuracy и loss обновляются после каждой эпохи
- Вкладка с результатами — сравнение всех обученных моделей

В будущем наверное я бы хотела сохранять обученные модели и уже в чат условно к ним добавлять фотки на тематики датасетов MNIST/CIFAR-10 и смотреть как отвечает моделька
---

## Архитектуры

| Архитектура | Описание | Параметры | Подходит для |
|---|---|---|---|
| Simple | 2 свёрточных блока | ~93K | MNIST, быстрый старт |
| Medium | BatchNorm + Dropout | ~300K | Стабильное обучение |
| Deep | 5 свёрточных блоков | ~1.2M | CIFAR-10, лучшая точность |

---

## Структура проекта
cnn-classifier/
├── src/
│ ├── _init_.py
│ ├── model.py — архитектуры CNN
│ └── train.py — загрузка данных и обучение
├── results/ — JSON с результатами экспериментов
├── app.py — Streamlit дашборд
├── main.py — CLI запуск обучения
├── requirements.txt
├── CNN_THEORY.md — теоретическая справка по CNN
└── README.md


---

## CLI запуск

Если не нужен дашборд, можно обучать прямо из терминала:

```bash
# одна архитектура
python main.py --dataset mnist --arch simple --epochs 5

# все архитектуры сразу
python main.py --dataset cifar10 --epochs 15

# с кастомными параметрами
python main.py --dataset cifar10 --arch deep --epochs 20 --batch 128
```

Результаты сохраняются в `results/<dataset>_<arch>.json`.

---

## Примерные результаты

| Датасет | Архитектура | Test Accuracy |
|---|---|---|
| MNIST | Simple | ~99.1% |
| MNIST | Medium | ~99.3% |
| CIFAR-10 | Simple | ~65% |
| CIFAR-10 | Deep | ~82% |

---

## Стек

- TensorFlow / Keras — построение и обучение моделей
- Streamlit — веб-дашборд на чистом Python
- NumPy — работа с массивами
- Pandas — таблицы и графики

---

## Теория

Подробная справка по CNN, MaxPooling, Dropout и BatchNormalization — в файле [CNN_THEORY.md](CNN_THEORY.md).