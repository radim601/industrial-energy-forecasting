"""
Модуль для обучения и оценки базовой модели SARIMA.
Служит бенчмарком для сравнения с более сложными нейросетевыми архитектурами.
"""

import os
import json
import pickle
import logging
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Отключаем сообщения от statsmodels при сходимости матриц
warnings.simplefilter('ignore', ConvergenceWarning)

# Импортируем общую логику
from utils import set_seed, calculate_and_log_metrics
from config import (
    DATASET_PATH, TARGET_COL, DATE_COL, PRED_LEN,
    DATA_SLICE_SIZE, WEIGHTS_DIR, SEASONALITY,ACTIVE_DATASET
)
from sarima_grid_search import run_grid_search


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
set_seed(42)


def run_sarima() -> None:
    """
    Запускает пайплайн базовой модели SARIMA:
    загрузка данных, поиск параметров (или загрузка из памяти), обучение,
    прогнозирование и расчет метрик.
    """
    logging.info("=== ЗАПУСК ОБУЧЕНИЯ: Базовая модель SARIMA ===")

    # Загружаем данные. Берем срез (DATA_SLICE_SIZE) для ускорения работы SARIMA
    df = pd.read_csv(DATASET_PATH, parse_dates=[DATE_COL], index_col=DATE_COL).sort_index()
    data = df[TARGET_COL]

    train_size = min(DATA_SLICE_SIZE, len(data) - PRED_LEN)
    train = data.iloc[:train_size]
    test = data.iloc[train_size:train_size + PRED_LEN]
    actuals = test.values

    # =========================================================================
    # Автоматический подбор гиперпараметров (Grid Search)
    # =========================================================================
    params_file = os.path.join(WEIGHTS_DIR, 'best_sarima_params.json')

    if os.path.exists(params_file):
        logging.info("Загрузка оптимальных гиперпараметров...")
        with open(params_file, 'r', encoding='utf-8') as f:
            best_params = json.load(f)

        best_order: Tuple[int, int, int] = tuple(best_params['order'])
        best_seasonal: Tuple[int, int, int, int] = tuple(best_params['seasonal'])
    else:
        logging.info("Параметры не найдены. Запуск поиска Grid Search...")
        best_order, best_seasonal = run_grid_search(DATASET_PATH, train_size, SEASONALITY)

        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump({'order': best_order, 'seasonal': best_seasonal}, f)
        logging.info(f"Параметры сохранены: {params_file}")

    # =========================================================================
    # Обучение и сохранение весов
    # =========================================================================
    logging.info(f"Инициализация SARIMA с архитектурой {best_order} x {best_seasonal}")
    model = sm.tsa.SARIMAX(train, order=best_order, seasonal_order=best_seasonal)
    fitted_model = model.fit(disp=False)

    model_path = os.path.join(WEIGHTS_DIR, 'best_sarima_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(fitted_model, f)

    # =========================================================================
    # Инференс и оценка (Бенчмарк для сравнения с нейросетями)
    # =========================================================================
    forecast = fitted_model.forecast(steps=PRED_LEN).values
    forecast = np.clip(forecast, a_min=(None if ACTIVE_DATASET == 'ett' else 0.0), a_max=None)  # Исключаем отрицательные прогнозы

    calculate_and_log_metrics(actuals, forecast, "SARIMA", y_train=train.values)
    logging.info("=== Модуль SARIMA успешно завершен ===")


if __name__ == "__main__":
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    run_sarima()