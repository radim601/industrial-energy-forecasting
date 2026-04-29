"""
Модуль для обучения и оценки модели DLinear.
Использует прямой проход (без рекуррентности) и декомпозицию сигнала
для быстрого и точного прогнозирования временных рядов.
"""

import os
import logging

import numpy as np
import joblib
import torch
import torch.nn as nn

from utils import set_seed, calculate_and_log_metrics, get_prepared_dataloaders, train_pytorch_model
from models import DLinear
from config import (
    DATASET_PATH, TARGET_COL, DATE_COL, SEQ_LEN, PRED_LEN,
    WEIGHTS_DIR, BATCH_SIZE, LEARNING_RATE, EPOCHS, PATIENCE,
    VAL_RATIO, DATA_SLICE_SIZE, ACTIVE_DATASET
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
set_seed(42)


def run_dlinear() -> None:
    """
    Запускает полный пайплайн модели DLinear:
    загрузка и подготовка данных, инициализация архитектуры,
    обучение с ранней остановкой, инференс на тестовом окне и расчет метрик.
    """
    logging.info("=== ЗАПУСК МОДУЛЯ: DLinear ===")

    # Убедимся, что директория для весов существует
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # =========================================================================
    # Загрузка данных и препроцессинг
    # =========================================================================
    train_loader, val_loader, test_dataset, scaler, train_data_inv = get_prepared_dataloaders(
        DATASET_PATH, TARGET_COL, DATE_COL, SEQ_LEN, PRED_LEN,
        BATCH_SIZE, VAL_RATIO, DATA_SLICE_SIZE
    )

    # Сохраняем скейлер для последующего использования в модуле predict.py
    scaler_path = os.path.join(WEIGHTS_DIR, 'global_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logging.info(f"Скейлер успешно сохранен: {scaler_path}")

    # =========================================================================
    # Инициализация и обучение модели
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Используемое вычислительное устройство: {device}")

    model = DLinear(seq_len=SEQ_LEN, pred_len=PRED_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    weights_path = os.path.join(WEIGHTS_DIR, 'best_clean_dlinear.pth')

    logging.info("Старт обучения DLinear...")
    train_pytorch_model(
        model, train_loader, val_loader, criterion, optimizer,
        device, EPOCHS, PATIENCE, weights_path
    )

    # =========================================================================
    # Инференс и расчет метрик
    # =========================================================================
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    # Берем первое окно из тестового датасета
    sample_idx = 0
    test_x, test_y = test_dataset[sample_idx]

    with torch.no_grad():
        test_tensor = test_x.unsqueeze(0).to(device)
        pred_y = model(test_tensor).cpu().numpy()[0]

    # Обратное масштабирование
    true_y_inv = scaler.inverse_transform(test_y.numpy())
    pred_y_inv = np.clip(scaler.inverse_transform(pred_y), a_min=(None if ACTIVE_DATASET == 'ett' else 0.0), a_max=None)

    calculate_and_log_metrics(
        true_y_inv[:, 0],
        pred_y_inv[:, 0],
        "DLinear",
        y_train=train_data_inv[:, 0]
    )

    logging.info("=== Модуль DLinear успешно завершен ===")


if __name__ == "__main__":
    run_dlinear()