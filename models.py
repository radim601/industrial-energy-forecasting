"""
Модуль с архитектурами нейросетевых моделей для прогнозирования временных рядов.
Содержит реализацию декомпозиционных блоков и модели DLinear.
"""

import torch
import torch.nn as nn

from config import KERNEL_SIZE, SEQ_LEN, PRED_LEN


class SeriesDecomposition(nn.Module):
    """
    Блок декомпозиции временного ряда на глобальный тренд и сезонность.
    Использует усреднение скользящим окном.
    """

    def __init__(self, kernel_size: int = KERNEL_SIZE):
        super(SeriesDecomposition, self).__init__()

        # ядро обязательно должно быть нечетным для симметричного паддинга
        assert kernel_size % 2 != 0, f"Ошибка: kernel_size должен быть нечетным (получено {kernel_size})."

        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Разделяет сигнал на тренд и высокочастотный остаток (сезонность).

        Args:
            x (torch.Tensor): Входной тензор размерности (Batch, Seq_len, Channels).

        Returns:
            tuple: Кортеж из двух тензоров (сезонная компонента, трендовая компонента).
        """
        # Динамический паддинг для сохранения исходной длины последовательности
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)

        # Выделение тренда через AvgPool1d
        trend_init = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)

        # Вычисление сезонности как разницы между оригиналом и трендом
        seasonal_init = x - trend_init

        return seasonal_init, trend_init


class DLinear(nn.Module):
    """
    Декомпозиционная линейная модель (DLinear) для прямого прогнозирования.
    Архитектура лишена рекуррентности и механизмов внимания, опираясь на независимые линейные слои.
    """

    def __init__(self, seq_len: int = SEQ_LEN, pred_len: int = PRED_LEN):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.decomposition = SeriesDecomposition(KERNEL_SIZE)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели DLinear. Проецирует историческое окно напрямую в горизонт прогноза.

        Args:
            x (torch.Tensor): Историческое окно размерности (Batch, Seq_len, Channels).

        Returns:
            torch.Tensor: Сгенерированный прогноз размерности (Batch, Pred_len, Channels).
        """
        # Декомпозиция сигнала
        seasonal_init, trend_init = self.decomposition(x)

        # Подготовка тензоров для линейного слоя (меняем местами измерения)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # Независимое линейное проецирование
        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)

        # Аддитивное объединение компонент
        output = seasonal_output + trend_output

        return output.permute(0, 2, 1)