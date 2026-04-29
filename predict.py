"""
Модуль промышленного инференса и генерации графиков.
Загружает веса обученных моделей, строит прогнозы на тестовом отрезке
и формирует сравнительные дашборды для оценки точности.
"""

import os
import pickle
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from config import (
    DATASET_PATH, TARGET_COL, DATE_COL, PRED_LEN,
    SEQ_LEN, WEIGHTS_DIR, PLOTS_DIR, SEASONALITY, ACTIVE_DATASET
)
from utils import engineer_features
from models import DLinear
from hybrid_smart_arima_lstm import ContextAwareLSTM
from autoformer import AutoformerFull

# Настройка логирования и стилей графиков
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.figsize': (16, 8),
    'axes.titlesize': 15,
    'axes.labelsize': 12,
    'legend.fontsize': 12,
    'legend.frameon': True,
    'legend.shadow': True,
    'lines.linewidth': 2.5
})

CLIP_MIN = None if ACTIVE_DATASET == 'ett' else 0.0

def run_comparative_inference() -> None:
    """
    Основная функция инференса. Загружает предысторию, прогоняет через все
    доступные обученные модели и генерирует графики (индивидуальные и общий дашборд).
    """
    logging.info("Инициализация сравнительного дашборда...")
    df = pd.read_csv(DATASET_PATH, parse_dates=[DATE_COL], index_col=DATE_COL).sort_index()

    SHIFT_DAYS = 10
    OFFSET = SEASONALITY * SHIFT_DAYS

    history_start = -(SEQ_LEN + PRED_LEN + OFFSET)
    history_end = -(PRED_LEN + OFFSET)

    history_series = df[TARGET_COL].iloc[history_start: history_end]
    history_df = df.iloc[history_start: history_end].copy()

    if OFFSET == 0:
        actual_future = df[TARGET_COL].iloc[-PRED_LEN:]
    else:
        actual_future = df[TARGET_COL].iloc[-(PRED_LEN + OFFSET): -OFFSET]

    actual_dates = actual_future.index

    # Инициализация переменных прогноза
    base_forecast: Optional[np.ndarray] = None
    dlinear_forecast: Optional[np.ndarray] = None
    hybrid_forecast: Optional[np.ndarray] = None
    autoformer_forecast: Optional[np.ndarray] = None

    # --- 1. SARIMA ---
    try:
        sarima_path = os.path.join(WEIGHTS_DIR, 'best_sarima_model.pkl')
        with open(sarima_path, 'rb') as f:
            sarima_model = pickle.load(f)
        updated_sarima = sarima_model.apply(history_series)
        base_forecast = np.clip(updated_sarima.forecast(steps=PRED_LEN).values, a_min=CLIP_MIN, a_max=None)
    except Exception as e:
        logging.error(f"Ошибка SARIMA: {e}")

    # Подготовка универсального тензора для нейросетей
    try:
        scaler_path = os.path.join(WEIGHTS_DIR, 'global_scaler.pkl')
        global_scaler = joblib.load(scaler_path)
        history_tensor = torch.FloatTensor(
            global_scaler.transform(history_series.values.reshape(-1, 1))
        ).unsqueeze(0)
    except Exception as e:
        logging.error(f"Ошибка загрузки скейлера: {e}")
        history_tensor = None

    # --- 2. DLinear ---
    try:
        if history_tensor is not None:
            dlinear_model = DLinear(seq_len=SEQ_LEN, pred_len=PRED_LEN)
            dlinear_weights = os.path.join(WEIGHTS_DIR, 'best_clean_dlinear.pth')
            dlinear_model.load_state_dict(torch.load(dlinear_weights, map_location='cpu', weights_only=True))
            dlinear_model.eval()
            with torch.no_grad():
                dlinear_out_scaled = dlinear_model(history_tensor).cpu().numpy().reshape(-1, 1)
            dlinear_forecast = np.clip(global_scaler.inverse_transform(dlinear_out_scaled).flatten(), a_min=CLIP_MIN, a_max=None)
    except Exception as e:
        logging.error(f"Ошибка DLinear: {e}")

    # --- 3. Гибрид (ARIMA-LSTM) ---
    try:
        # Загружаем контекст и скейлеры для LSTM
        context_data = engineer_features(history_df, TARGET_COL)
        context_features = context_data.drop(columns=[TARGET_COL])

        scaler_context = joblib.load(os.path.join(WEIGHTS_DIR, 'scaler_context.pkl'))
        scaler_resid = joblib.load(os.path.join(WEIGHTS_DIR, 'scaler_resid.pkl'))

        # Загружаем и инициализируем саму сеть LSTM
        lstm_model = ContextAwareLSTM(input_size=context_features.shape[1] + 1)
        lstm_weights = os.path.join(WEIGHTS_DIR, 'best_hybrid_lstm.pth')
        lstm_model.load_state_dict(torch.load(lstm_weights, map_location='cpu', weights_only=True))
        lstm_model.eval()

        # Загружаем ARIMA
        arima_path = os.path.join(WEIGHTS_DIR, 'best_arima_for_hybrid.pkl')
        with open(arima_path, 'rb') as f:
            arima_model = pickle.load(f)

        # Прогоняем предысторию через ARIMA
        history_arima_result = arima_model.apply(history_series)
        history_arima_fitted = history_arima_result.fittedvalues

        # Считаем остатки от ARIMA
        history_residuals = (history_series.values - history_arima_fitted.values).reshape(-1, 1)
        history_residuals_scaled = scaler_resid.transform(history_residuals)

        # Формируем входной тензор для LSTM
        lstm_features = np.hstack((history_residuals_scaled, scaler_context.transform(context_features)))

        # Инференс LSTM
        with torch.no_grad():
            lstm_output_scaled = lstm_model(
                torch.tensor(lstm_features, dtype=torch.float32).unsqueeze(0)
            ).cpu().numpy().reshape(-1, 1)

        arima_base_forecast = np.clip(history_arima_result.forecast(steps=PRED_LEN).values, a_min=CLIP_MIN, a_max=None)

        # Прогноз ARIMA + прогноз остатков от LSTM
        hybrid_forecast = np.clip(
            arima_base_forecast + scaler_resid.inverse_transform(lstm_output_scaled).flatten(),
            a_min=CLIP_MIN, a_max=None
        )

    except Exception as e:
        logging.error(f"Ошибка Гибрида: {e}")

    # --- 4. Autoformer ---
    try:
        if history_tensor is not None:
            autoformer_model = AutoformerFull(seq_len=SEQ_LEN, pred_len=PRED_LEN)
            autoformer_weights = os.path.join(WEIGHTS_DIR, 'best_autoformer.pth')
            autoformer_model.load_state_dict(torch.load(autoformer_weights, map_location='cpu', weights_only=True))
            autoformer_model.eval()
            with torch.no_grad():
                autoformer_out_scaled = autoformer_model(history_tensor).cpu().numpy().reshape(-1, 1)
            autoformer_forecast = np.clip(global_scaler.inverse_transform(autoformer_out_scaled).flatten(), a_min=CLIP_MIN, a_max=None)
    except Exception as e:
        logging.error(f"Ошибка Autoformer: {e}")

    def plot_individual(model_name: str, forecast: np.ndarray, color: str, filename: str, marker: str) -> None:
        """Вспомогательная функция для отрисовки индивидуального графика модели."""
        plt.figure(figsize=(14, 7))
        plot_hist_len = SEASONALITY * 3
        zoom_dates = history_series.index[-plot_hist_len:]
        zoom_values = history_series.values[-plot_hist_len:]

        plt.plot(zoom_dates, zoom_values, label="Предыстория", color='grey', alpha=0.6, linewidth=1.5)
        plt.plot(actual_dates, actual_future.values, label="Факт", color='#1f77b4', marker='o')
        plt.plot(actual_dates, forecast, label=f"Прогноз ({model_name})", color=color, linestyle='-', marker=marker)

        plt.axvline(x=zoom_dates[-1], color='black', linestyle=':', linewidth=1.5)
        plt.title(f"Прогноз энергопотребления ({model_name})")

        plt.ylabel("Потребление электроэнергии (кВт⋅ч)")

        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b, %H:%M'))
        plt.gcf().autofmt_xdate()
        plt.legend(loc='upper left')

        plot_path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    logging.info("Генерация графиков...")
    if base_forecast is not None:
        plot_individual("SARIMA", base_forecast, '#ff7f0e', f'{ACTIVE_DATASET}_sarima.png', 'x')
    if dlinear_forecast is not None:
        plot_individual("DLinear", dlinear_forecast, '#2ca02c', f'{ACTIVE_DATASET}_dlinear.png', 'x')
    if hybrid_forecast is not None:
        plot_individual("ARIMA-LSTM", hybrid_forecast, '#d62728', f'{ACTIVE_DATASET}_hybrid.png', 'x')
    if autoformer_forecast is not None:
        plot_individual("Autoformer", autoformer_forecast, '#9467bd', f'{ACTIVE_DATASET}_autoformer.png', 'x')

    # === ОБЩИЙ ДАШБОРД ===
    plt.figure()
    plot_hist_len = SEASONALITY * 2
    zoom_dates = history_series.index[-plot_hist_len:]
    zoom_values = history_series.values[-plot_hist_len:]

    plt.plot(zoom_dates, zoom_values, label="Факт (предыстория 2 дня)", color='grey', linewidth=2)
    plt.plot(actual_dates, actual_future.values, label="Истинное потребление (Факт)", color='black',marker='o',markersize=7, linewidth=2.5, alpha=0.5)

    if base_forecast is not None:
        plt.plot(actual_dates, base_forecast, label="SARIMA", color='#ff7f0e', linestyle='--',marker='d',markersize=7, linewidth=2, alpha=0.8)
    if hybrid_forecast is not None:
        plt.plot(actual_dates, hybrid_forecast, label="ARIMA-LSTM", color='#d62728', marker='x', markersize=7, linewidth=1.5, alpha=0.9)
    if dlinear_forecast is not None:
        plt.plot(actual_dates, dlinear_forecast, label="DLinear", color='#2ca02c', linestyle='-',marker='s',markersize=7, linewidth=2.5)
    if autoformer_forecast is not None:
        plt.plot(actual_dates, autoformer_forecast, label="Autoformer", color='#9467bd', linestyle='-.',marker='^',markersize=7, linewidth=2, alpha=0.8)

    plt.axvline(x=zoom_dates[-1], color='black', linestyle=':', linewidth=2)
    plt.title("Сравнительный дашборд моделей прогнозирования")

    plt.ylabel("Потребление электроэнергии (кВт⋅ч)")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b, %H:%M'))
    plt.gcf().autofmt_xdate()
    plt.legend(loc='upper left')

    dashboard_path = os.path.join(PLOTS_DIR, f'{ACTIVE_DATASET}_production_comparative_dashboard.png')
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    logging.info(f"Графики и дашборд успешно сохранены в папку {PLOTS_DIR}")


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    run_comparative_inference()