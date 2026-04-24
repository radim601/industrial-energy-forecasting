"""Модуль автоматизированного поиска оптимальных гиперпараметров для модели SARIMA."""

import logging
from typing import Tuple

import pandas as pd
import pmdarima as pm

from config import SEASONALITY, DATA_SLICE_SIZE, DATASET_PATH, TARGET_COL, DATE_COL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_grid_search(
    data_path: str,
    slice_size: int,
    seasonality: int
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    """
    Выполняет автоматический подбор гиперпараметров для модели SARIMA
    с помощью алгоритма Хиндмана-Кандакара (stepwise search).

    Args:
        data_path (str): Путь к CSV-файлу с датасетом.
        slice_size (int): Количество записей с начала ряда, используемых для подбора.
        seasonality (int): Период сезонности (m).

    Returns:
        Tuple: Два кортежа. Первый — order (p, d, q), второй — seasonal_order (P, D, Q, m).
               В случае ошибки возвращает дефолтные параметры (1, 1, 1) и (1, 0, 0, m).
    """
    logging.info(f"Загрузка данных из {data_path} ({slice_size} записей)...")

    df = pd.read_csv(data_path, parse_dates=[DATE_COL], index_col=DATE_COL).sort_index()

    # Берем срез данных с начала ряда
    y_train = df[TARGET_COL].iloc[:slice_size]

    logging.info("Запуск pmdarima (алгоритм Хиндмана-Кандакара)...")
    try:
        model = pm.auto_arima(
            y_train,
            start_p=0, start_q=0, max_p=2, max_q=2,  # Ограничили AR/MA компоненты для предотвращения переобучения
            m=seasonality,
            start_P=0, start_Q=0, max_P=1, max_Q=1,
            d=1, D=1,                                # Фиксируем дифференцирование (для приведения к стационарности)
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,                           # Пошаговый поиск (быстрее, чем полный grid search)
            n_jobs=1                                 # При stepwise=True распараллеливание не поддерживается
        )

        logging.info(f"ПОДБОР ПАРАМЕТРОВ ЗАВЕРШЕН! Лучший AIC: {model.aic()}")
        logging.info(f"Оптимальные параметры: order={model.order}, seasonal={model.seasonal_order}")

        return model.order, model.seasonal_order

    except Exception as e:
        logging.error(f"Ошибка при поиске параметров: {e}")
        return (1, 1, 1), (1, 0, 0, seasonality)


if __name__ == "__main__":
    run_grid_search(DATASET_PATH, DATA_SLICE_SIZE, SEASONALITY)