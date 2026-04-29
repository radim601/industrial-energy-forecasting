"""
Модуль реализации гибридной модели прогнозирования временных рядов (ARIMA-LSTM).
Реализует асимметричный подход: статистическое моделирование базового тренда
с последующей нейросетевой компенсацией нелинейных остатков.
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

# Импорт вспомогательных модулей
from utils import set_seed, calculate_and_log_metrics, engineer_features, train_pytorch_model
from config import (
    DATASET_PATH, TARGET_COL, DATE_COL, SEQ_LEN, PRED_LEN,
    DATA_SLICE_SIZE, WEIGHTS_DIR, BATCH_SIZE, LEARNING_RATE,
    VAL_RATIO, HYBRID_EPOCHS, HYBRID_PATIENCE, ACTIVE_DATASET
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
set_seed(42)


class ContextAwareLSTM(nn.Module):
    """
    Нейросетевая модель на базе LSTM для прогнозирования нелинейных остатков.
    Учитывает как исторические значения самих остатков, так и экзогенный контекст.
    """
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.0)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, PRED_LEN)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели.

        Args:
            x (torch.Tensor): Тензор признаков размерности (Batch, Seq_len, Features)

        Returns:
            torch.Tensor: Прогноз остатков размерности (Batch, Pred_len)
        """
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        return self.regressor(last_time_step_out)


def run_hybrid() -> None:
    """
    Запускает полный цикл обучения гибридной модели:
    1. Обучение/загрузка базовой ARIMA и извлечение остатков.
    2. Подготовка датасета с контекстными признаками.
    3. Обучение сети LSTM прогнозировать будущие остатки.
    4. Финальный инференс и расчет метрик.
    """
    logging.info("=== ЗАПУСК ОБУЧЕНИЯ: Композитная модель ARIMA-LSTM ===")

    df = pd.read_csv(DATASET_PATH, parse_dates=[DATE_COL], index_col=DATE_COL).sort_index()
    df = engineer_features(df, TARGET_COL)

    data_slice = df.iloc[:DATA_SLICE_SIZE]
    y = data_slice[TARGET_COL]

    train_size = len(y) - PRED_LEN
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # =========================================================================
    # Базовое статистическое моделирование (ARIMA)
    # =========================================================================
    arima_weights = os.path.join(WEIGHTS_DIR, 'best_arima_for_hybrid.pkl')
    try:
        with open(arima_weights, 'rb') as f:
            arima_model = pickle.load(f)
        logging.info("Базовая компонента (ARIMA) успешно загружена из памяти")
    except FileNotFoundError:
        logging.info("Инициализация обучения базовой компоненты ARIMA...")
        arima_model = ARIMA(endog=y_train, order=(2, 1, 2)).fit()
        with open(arima_weights, 'wb') as f:
            pickle.dump(arima_model, f)

    # Используем fittedvalues вместо predict для корректной работы с d=1
    train_forecast = arima_model.fittedvalues
    residuals = y_train.values - train_forecast.values

    # =========================================================================
    # Нейросетевая компенсация (LSTM)
    # =========================================================================
    logging.info("Подготовка набора данных для обучения компенсирующей модели LSTM...")
    X_context = data_slice.drop(columns=[TARGET_COL]).iloc[:train_size]

    # Масштабируем по обучающей части
    val_split_idx = int(len(residuals) * (1 - VAL_RATIO))

    scaler_context = StandardScaler()
    scaler_context.fit(X_context.iloc[:val_split_idx])
    X_context_scaled = scaler_context.transform(X_context)

    scaler_resid = StandardScaler()
    scaler_resid.fit(residuals[:val_split_idx].reshape(-1, 1))
    residuals_scaled = scaler_resid.transform(residuals.reshape(-1, 1))

    joblib.dump(scaler_context, os.path.join(WEIGHTS_DIR, 'scaler_context.pkl'))
    joblib.dump(scaler_resid, os.path.join(WEIGHTS_DIR, 'scaler_resid.pkl'))

    X_lstm, y_lstm = [], []
    for i in range(len(residuals_scaled) - SEQ_LEN - PRED_LEN + 1):
        context_window = X_context_scaled[i : i+SEQ_LEN]
        resid_window = residuals_scaled[i : i+SEQ_LEN]

        # Горизонтальная конкатенация: остатки + признаки времени (синусы/косинусы)
        combined_features = np.hstack((resid_window, context_window))
        X_lstm.append(combined_features)
        y_lstm.append(residuals_scaled[i+SEQ_LEN : i+SEQ_LEN+PRED_LEN].flatten())

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    split = int((1 - VAL_RATIO) * len(X_lstm))
    X_train_t = torch.tensor(X_lstm[:split], dtype=torch.float32)
    y_train_t = torch.tensor(y_lstm[:split], dtype=torch.float32)
    X_val_t = torch.tensor(X_lstm[split:], dtype=torch.float32)
    y_val_t = torch.tensor(y_lstm[split:], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_lstm.shape[2]
    model = ContextAwareLSTM(input_size=input_size).to(device)

    # HuberLoss устойчив к выбросам в значениях энергопотребления
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    weights_path = os.path.join(WEIGHTS_DIR, 'best_hybrid_lstm.pth')

    logging.info("Обучение компенсирующей сети LSTM...")
    train_pytorch_model(
        model, train_loader, val_loader, criterion, optimizer,
        device, HYBRID_EPOCHS, HYBRID_PATIENCE, weights_path
    )

    # =========================================================================
    # Инференс
    # =========================================================================
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    context_test = scaler_context.transform(
        data_slice.drop(columns=[TARGET_COL]).iloc[train_size-SEQ_LEN : train_size]
    )
    recent_residuals = residuals[-SEQ_LEN:].reshape(-1, 1)
    recent_residuals_scaled = scaler_resid.transform(recent_residuals)

    combined_test = np.hstack((recent_residuals_scaled, context_test))

    with torch.no_grad():
        test_tensor = torch.tensor(combined_test, dtype=torch.float32).unsqueeze(0).to(device)
        pred_resid_scaled = model(test_tensor).cpu().numpy().reshape(-1, 1)

    pred_resid_inv = scaler_resid.inverse_transform(pred_resid_scaled).flatten()
    base_forecast = arima_model.forecast(steps=PRED_LEN).values

    # Итоговый прогноз: базовый тренд + предсказанные остатки
    final_hybrid_forecast = np.clip(base_forecast + pred_resid_inv, a_min=(None if ACTIVE_DATASET == 'ett' else 0.0), a_max=None)

    actual_values = y_test.values[:PRED_LEN]
    calculate_and_log_metrics(actual_values, final_hybrid_forecast, "ARIMA-LSTM Hybrid", y_train=y_train.values)

    logging.info("=== Модуль гибридного прогнозирования успешно завершен ===")


if __name__ == "__main__":
    run_hybrid()