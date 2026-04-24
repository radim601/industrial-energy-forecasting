"""
Модуль вспомогательных функций для подготовки данных, расчета метрик
и обучения моделей прогнозирования временных рядов.
"""

import random
import logging
from typing import Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def set_seed(seed: int = 42) -> None:
    """
    Фиксирует генераторы псевдослучайных чисел для обеспечения воспроизводимости экспериментов.

    Args:
        seed (int): Значение seed для random, numpy и torch. По умолчанию 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет симметричную среднюю абсолютную ошибку в процентах (sMAPE).

    Args:
        y_true (np.ndarray): Фактические значения.
        y_pred (np.ndarray): Предсказанные значения.

    Returns:
        float: Значение метрики sMAPE (в процентах).
    """
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: Optional[np.ndarray] = None) -> float:
    """
    Вычисляет среднюю абсолютную масштабированную ошибку (MASE).
    Метрика устойчива к масштабу данных и сравнивает модель с наивным прогнозом.

    Args:
        y_true (np.ndarray): Фактические значения.
        y_pred (np.ndarray): Предсказанные значения.
        y_train (Optional[np.ndarray]): Обучающая выборка для расчета наивной ошибки.
            Если не передана, наивная ошибка считается по y_true.

    Returns:
        float: Значение MASE. Возвращает np.nan, если наивная ошибка равна 0.
    """
    if y_train is not None and len(y_train) > 1:
        naive_error = np.mean(np.abs(np.diff(y_train)))
    else:
        naive_error = np.mean(np.abs(np.diff(y_true)))

    if naive_error == 0:
        return np.nan

    return mean_absolute_error(y_true, y_pred) / naive_error


def calculate_and_log_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    y_train: Optional[np.ndarray] = None
) -> Tuple[float, float, float, float, float, float]:
    """
    Рассчитывает основной набор метрик регрессии и выводит их в лог.

    Args:
        y_true (np.ndarray): Истинные значения.
        y_pred (np.ndarray): Прогнозы модели.
        model_name (str): Название модели для заголовка логов.
        y_train (Optional[np.ndarray]): Данные обучения для расчета MASE.

    Returns:
        Tuple: Кортеж из метрик (MSE, RMSE, MAE, MASE, sMAPE, R2).
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)
    mase = calculate_mase(y_true, y_pred, y_train)
    r2 = r2_score(y_true, y_pred)

    logging.info(f"\n--- МЕТРИКИ: {model_name} ---")
    logging.info(f"MSE:   {mse:.3f}")
    logging.info(f"RMSE:  {rmse:.3f}")
    logging.info(f"MAE:   {mae:.3f}")
    logging.info(f"MASE:  {mase:.3f}")
    logging.info(f"sMAPE: {smape:.2f} %")
    logging.info(f"R^2:   {r2:.3f}\n")

    return mse, rmse, mae, mase, smape, r2


def get_prepared_dataloaders(
    data_path: str,
    target_col: str,
    date_col: str,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    val_ratio: float = 0.15,
    data_slice_size: int = 2000
) -> Tuple[DataLoader, DataLoader, "TimeSeriesDataset", MinMaxScaler, np.ndarray]:
    """
    Загружает данные, масштабирует их и разбивает на обучающий, валидационный
    и тестовый DataLoader'ы.

    Args:
        data_path (str): Путь к CSV-файлу с данными.
        target_col (str): Название целевой переменной.
        date_col (str): Название колонки с датой/временем (используется как индекс).
        seq_len (int): Длина исторического окна.
        pred_len (int): Горизонт прогнозирования.
        batch_size (int): Размер батча.
        val_ratio (float): Доля данных для валидации от общего объема обучающей выборки.
        data_slice_size (int): Ограничение размера датасета для синхронизации тестов.

    Returns:
        Tuple: Обучающий лоадер, валидационный лоадер, тестовый датасет,
               обученный scaler и инвертированные данные обучения (для MASE).
    """
    df = pd.read_csv(data_path, parse_dates=[date_col], index_col=date_col).sort_index()
    data = df[[target_col]].values

    # Синхронизация тестовой выборки с SARIMA
    total_len = min(data_slice_size + pred_len, len(data))
    data = data[:total_len]

    train_size = len(data) - pred_len
    val_size = int(train_size * val_ratio)
    train_end = train_size - val_size

    scaler = MinMaxScaler()
    scaler.fit(data[:train_end])
    data_scaled = scaler.transform(data)

    # Train: от начала до train_end
    train_dataset = TimeSeriesDataset(data_scaled[:train_end], seq_len, pred_len)

    # Val: окно для валидации
    val_data = data_scaled[train_end - seq_len: train_size]
    val_dataset = TimeSeriesDataset(val_data, seq_len, pred_len)

    # Test: Строго ОДНО окно, предсказывающее PRED_LEN (как в SARIMA)
    test_data = data_scaled[train_size - seq_len: train_size + pred_len]
    test_dataset = TimeSeriesDataset(test_data, seq_len, pred_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Извлекаем оригинальные обучающие данные для корректного расчета MASE
    train_data_inv = scaler.inverse_transform(data_scaled[:train_size])

    return train_loader, val_loader, test_dataset, scaler, train_data_inv


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset для формирования последовательностей временных рядов.
    Генерирует пары (X, y) с использованием скользящего окна.
    """

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        """
        Args:
            data (np.ndarray): Одномерный или многомерный массив признаков.
            seq_len (int): Количество исторических шагов (X).
            pred_len (int): Количество шагов для прогноза (y).
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def engineer_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Генерирует циклические временные признаки (синусы и косинусы) на основе индекса.
    Отлично подходит для фиксации суточной и недельной сезонности (например, пиков энергопотребления).

    Args:
        df (pd.DataFrame): Исходный датафрейм (индекс должен быть типа Datetime).
        target_col (str): Название колонки с целевой переменной.

    Returns:
        pd.DataFrame: Датафрейм, содержащий исходную целевую переменную и новые признаки.
    """
    hours = df.index.hour + df.index.minute / 60.0
    days = df.index.dayofweek

    new_features = pd.DataFrame({
        'hour_sin': np.sin(2 * np.pi * hours / 24.0),
        'hour_cos': np.cos(2 * np.pi * hours / 24.0),
        'day_sin': np.sin(2 * np.pi * days / 7.0),
        'day_cos': np.cos(2 * np.pi * days / 7.0)
    }, index=df.index)

    return pd.concat([df[[target_col]], new_features], axis=1).astype(np.float32)


def train_pytorch_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    weights_path: str
) -> None:
    """
    Универсальный цикл обучения для нейросетевых архитектур (DLinear, Autoformer и гибридов).
    Включает расчет Loss-функции, проход по эпохам и механизм Early Stopping.

    Args:
        model (torch.nn.Module): Инициализированная модель PyTorch.
        train_loader (DataLoader): Загрузчик обучающих данных.
        val_loader (DataLoader): Загрузчик валидационных данных.
        criterion (torch.nn.modules.loss._Loss): Функция потерь (например, MSELoss).
        optimizer (torch.optim.Optimizer): Оптимизатор (например, Adam).
        device (torch.device): Устройство для вычислений ('cpu' или 'cuda').
        epochs (int): Максимальное количество эпох обучения.
        patience (int): Количество эпох без улучшений до ранней остановки.
        weights_path (str): Путь для сохранения лучших весов модели.
    """
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x.to(device))
            loss = criterion(out, batch_y.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                out = model(batch_x.to(device))
                val_loss += criterion(out, batch_y.to(device)).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if epoch % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            logging.info(f"Эпоха {epoch + 1:03d}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.warning(f"Ранняя остановка на эпохе {epoch + 1}")
                break