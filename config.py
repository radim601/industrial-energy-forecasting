"""
Конфигурационный файл с глобальными гиперпараметрами.
Управляет выбором датасета, настройками архитектур и процессом обучения.
"""

import os
from typing import Dict, Any

# =========================================================================
# ВЫБОР АКТИВНОГО ДАТАСЕТА
# =========================================================================
ACTIVE_DATASET: str = 'ett'

DATASETS: Dict[str, Dict[str, Any]] = {
    'steel': {
        'path': 'data/steel_hourly.csv',       # Сжатый файл (1 час)
        'target': 'Usage_kWh',
        'date_col': 'date',
        'freq': '1h',
        'seasonality': 24,                     # Суточная сезонность
        'seq_len': 336,                        # История: 14 дней (24 * 14)
        'pred_len': 72,                        # Прогноз: 3 дня (24 * 3)
        'data_slice': 2000                     # Срез для базовой SARIMA
    },
    'ett': {
        'path': 'data/ETTh1.csv',              # Набор данных силовых трансформаторов
        'target': 'HUFL',                      # High Useful Load (Активная полезная нагрузка)
        'date_col': 'date',
        'freq': '1h',
        'seasonality': 24,
        'seq_len': 336,
        'pred_len': 72,
        'data_slice': 2000
    }
}

# Инициализация параметров датасета
current = DATASETS[ACTIVE_DATASET]

DATASET_PATH: str = current['path']
TARGET_COL: str = current['target']
DATE_COL: str = current['date_col']
SEASONALITY: int = current['seasonality']
SEQ_LEN: int = current['seq_len']
PRED_LEN: int = current['pred_len']
DATA_SLICE_SIZE: int = current['data_slice']
FREQ: str = current['freq']

# =========================================================================
# НАСТРОЙКА ДИРЕКТОРИЙ
# =========================================================================
WEIGHTS_DIR: str = f"weights_{ACTIVE_DATASET}/"
PLOTS_DIR: str = f"plots_{ACTIVE_DATASET}/"

# Гарантируем наличие папок для сохранения результатов
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# =========================================================================
# ГИПЕРПАРАМЕТРЫ АРХИТЕКТУР (DLinear, Autoformer, LSTM)
# =========================================================================
KERNEL_SIZE: int = 25      # Обязательно нечетное число для декомпозиции
HIDDEN_SIZE: int = 256
DROPOUT_RATE: float = 0.2

# =========================================================================
# ГИПЕРПАРАМЕТРЫ ОБУЧЕНИЯ
# =========================================================================
BATCH_SIZE: int = 64
LEARNING_RATE: float = 0.001
VAL_RATIO: float = 0.15    # Единая доля валидации для всех моделей

# Настройки для DLinear и Autoformer
EPOCHS: int = 30
PATIENCE: int = 5

# Специфичные настройки для гибрида ARIMA-LSTM
HYBRID_EPOCHS: int = 100
HYBRID_PATIENCE: int = 20