import os
import json
import glob
import time
import streamlit as st
import pandas as pd
from tensorflow import keras
from src.model import ARCHITECTURES

st.set_page_config(page_title="CNN Classifier", page_icon="🧠", layout="wide")
st.title("🧠 CNN Classifier Dashboard")
st.caption("Обучай и сравнивай свёрточные нейронные сети прямо в браузере")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_results():
    files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    results = []
    for f in files:
        with open(f) as fp:
            results.append(json.load(fp))
    return results


class StreamlitCallback(keras.callbacks.Callback):
    def __init__(self, epochs, log_placeholder, chart_placeholder):
        super().__init__()
        self.total_epochs = epochs
        self.log_placeholder = log_placeholder
        self.chart_placeholder = chart_placeholder
        self.history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
        self.logs_text = []

    def on_epoch_end(self, epoch, logs=None):
        acc      = logs.get("accuracy", 0)
        val_acc  = logs.get("val_accuracy", 0)
        loss     = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)

        self.history["train_acc"].append(round(acc, 4))
        self.history["val_acc"].append(round(val_acc, 4))
        self.history["train_loss"].append(round(loss, 4))
        self.history["val_loss"].append(round(val_loss, 4))

        # Лог-строка
        line = (
            f"Эпоха {epoch+1}/{self.total_epochs} | "
            f"accuracy: {acc:.4f} | val_accuracy: {val_acc:.4f} | "
            f"loss: {loss:.4f} | val_loss: {val_loss:.4f}"
        )
        self.logs_text.append(line)

        # Обновляем лог в UI
        self.log_placeholder.code("\n".join(self.logs_text))

        # Обновляем графики в реальном времени
        epochs_range = list(range(1, epoch + 2))
        df_acc = pd.DataFrame({
            "Train Accuracy": self.history["train_acc"],
            "Val Accuracy":   self.history["val_acc"],
        }, index=epochs_range)
        df_loss = pd.DataFrame({
            "Train Loss": self.history["train_loss"],
            "Val Loss":   self.history["val_loss"],
        }, index=epochs_range)

        with self.chart_placeholder.container():
            c1, c2 = st.columns(2)
            c1.line_chart(df_acc)
            c2.line_chart(df_loss)


tab1, tab2 = st.tabs(["🚀 Обучить модель", "📊 Результаты"])


# ТАБ 1: Запуск обучения
with tab1:
    st.header("Настройки обучения")

    col_l, col_r = st.columns(2)

    with col_l:
        dataset = st.selectbox(
            "📦 Датасет",
            ["mnist", "cifar10"],
            help="MNIST — цифры 0-9 (проще). CIFAR-10 — 10 классов объектов (сложнее)."
        )
        arch = st.selectbox(
            "🏗 Архитектура",
            ["simple", "medium", "deep"],
            help="simple — 2 Conv блока | medium — BatchNorm + Dropout | deep — 5 блоков"
        )

    with col_r:
        epochs = st.slider("⏱ Количество эпох", min_value=1, max_value=30, value=5)
        batch_size = st.select_slider(
            "📦 Размер батча",
            options=[32, 64, 128, 256],
            value=64,
            help="Сколько изображений обрабатывается за один шаг. Больше = быстрее, но больше памяти."
        )

    # Описание выбранной архитектуры
    arch_info = {
        "simple": "**Simple CNN** — 2 свёрточных блока, ~93K параметров. Быстро обучается, хорошо для MNIST.",
        "medium": "**Medium CNN** — BatchNorm + Dropout, ~300K параметров. Стабильное обучение, меньше переобучения.",
        "deep":   "**Deep CNN** — 5 блоков, ~1.2M параметров. Лучшая точность на CIFAR-10, дольше обучается.",
    }
    st.info(arch_info[arch])

    st.divider()

    if st.button("▶️ Запустить обучение", type="primary", use_container_width=True):

        import numpy as np

        # Загрузка данных
        with st.spinner("Загрузка датасета..."):
            if dataset == "cifar10":
                (x_tr, y_tr), (x_te, y_te) = keras.datasets.cifar10.load_data()
                input_shape = (32, 32, 3)
            else:
                (x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
                x_tr = x_tr[..., np.newaxis]
                x_te = x_te[..., np.newaxis]
                input_shape = (28, 28, 1)

            x_tr = x_tr.astype("float32") / 255.0
            x_te = x_te.astype("float32") / 255.0
            y_tr = keras.utils.to_categorical(y_tr, 10)
            y_te = keras.utils.to_categorical(y_te, 10)

        st.success(f"✅ Датасет загружен: {x_tr.shape[0]} train / {x_te.shape[0]} test")

        # Строим модель
        model = ARCHITECTURES[arch](input_shape, num_classes=10)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        total_params = model.count_params()
        st.caption(f"Параметров в модели: **{total_params:,}**")

        # Плейсхолдеры для live-обновления
        st.subheader("📡 Обучение в реальном времени")
        chart_placeholder = st.empty()
        log_placeholder   = st.empty()

        # Запуск
        cb = StreamlitCallback(epochs, log_placeholder, chart_placeholder)
        model.fit(
            x_tr, y_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[cb],
            verbose=0,  # отключаем вывод в терминал — всё идёт в UI
        )

        # Финальная оценка
        test_loss, test_acc = model.evaluate(x_te, y_te, verbose=0)
        st.success(f"🏁 Обучение завершено! Test Accuracy: **{test_acc*100:.2f}%**")

        # Сохраняем результат
        result = {
            "dataset":    dataset,
            "arch":       arch,
            "epochs":     epochs,
            "batch_size": batch_size,
            "train_acc":  cb.history["train_acc"],
            "val_acc":    cb.history["val_acc"],
            "train_loss": cb.history["train_loss"],
            "val_loss":   cb.history["val_loss"],
            "test_acc":   round(float(test_acc), 4),
            "test_loss":  round(float(test_loss), 4),
            "params":     total_params,
        }
        out_path = os.path.join(RESULTS_DIR, f"{dataset}_{arch}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        st.caption(f"💾 Сохранено в `{out_path}`")
        st.balloons()


# ТАБ 2: Результаты
with tab2:
    results = load_results()

    if not results:
        st.info("Пока нет результатов. Обучи первую модель во вкладке 🚀")
    else:
        # KPI карточки
        cols = st.columns(min(len(results), 4))
        for col, r in zip(cols, results):
            col.metric(
                label=f"{r['dataset']} / {r['arch']}",
                value=f"{r['test_acc']*100:.2f}%",
                delta=f"{r['params']:,} params"
            )

        st.divider()

        # Сравнение accuracy
        st.subheader("🏆 Test Accuracy по моделям")
        df_bar = pd.DataFrame([{
            "Модель": f"{r['dataset']}/{r['arch']}",
            "Test Accuracy %": round(r['test_acc'] * 100, 2)
        } for r in results])
        st.bar_chart(df_bar.set_index("Модель"))

        st.divider()

        # Кривые выбранной модели
        st.subheader("📈 Кривые обучения")
        labels   = [f"{r['dataset']} / {r['arch']}" for r in results]
        selected = st.selectbox("Выбери модель:", labels)
        r        = results[labels.index(selected)]
        ep       = list(range(1, len(r['train_acc']) + 1))

        c1, c2 = st.columns(2)
        c1.line_chart(pd.DataFrame({"Train Accuracy": r['train_acc'], "Val Accuracy": r['val_acc']}, index=ep))
        c2.line_chart(pd.DataFrame({"Train Loss": r['train_loss'], "Val Loss": r['val_loss']}, index=ep))

        st.divider()

        # Таблица
        st.subheader("📋 Все эксперименты")
        st.dataframe(pd.DataFrame([{
            "Датасет":       r['dataset'],
            "Архитектура":   r['arch'],
            "Эпохи":         r['epochs'],
            "Test Accuracy": f"{r['test_acc']*100:.2f}%",
            "Параметры":     f"{r['params']:,}",
        } for r in results]), use_container_width=True)