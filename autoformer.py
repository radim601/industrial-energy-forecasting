"""
Модуль с архитектурой Autoformer для прогнозирования временных рядов.
Основан на механизме автокорреляции (AutoCorrelation) и внутренней декомпозиции сигнала.
"""

import os
import math
import logging

import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

# Подтягиваем общую логику из utils
from utils import set_seed, calculate_and_log_metrics, get_prepared_dataloaders, train_pytorch_model
from config import (
    DATASET_PATH, TARGET_COL, DATE_COL, SEQ_LEN, PRED_LEN,
    WEIGHTS_DIR, BATCH_SIZE, LEARNING_RATE, KERNEL_SIZE,
    EPOCHS, PATIENCE, VAL_RATIO, DATA_SLICE_SIZE,ACTIVE_DATASET
)
from models import SeriesDecomposition

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
set_seed(42)


# =========================================================================
# БЛОК ЭМБЕДДИНГОВ
# =========================================================================

class PositionalEmbedding(nn.Module):
    """
    Классическое синусоидальное позиционное кодирование (embedding).
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """
    Векторное представление (embedding) значений с использованием одномерной свертки (1D Conv).
    """

    def __init__(self, c_in: int, d_model: int):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model,
            kernel_size=3, padding=1, padding_mode='circular', bias=False
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding_wo_pos(nn.Module):
    """
    Векторное представление данных без позиционного кодирования.
    """

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.value_embedding(x)
        return self.dropout(x)


# =========================================================================
# БЛОК AUTOCORRELATION С MULTI-HEAD ОБЕРТКОЙ
# =========================================================================

class AutoCorrelation(nn.Module):
    """
    Механизм автокорреляции, включающий две фазы:
    (1) поиск зависимостей на основе периодов;
    (2) агрегация временных задержек.
    Этот блок может бесшовно заменить классическое семейство механизмов self-attention.
    """

    def __init__(self, mask_flag: bool = True, factor: int = 1, scale: float = None,
                 attention_dropout: float = 0.1, output_attention: bool = False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        Фаза агрегации временных задержек для режима обучения.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        # поиск top-k значений
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)

        # обновление матрицы корреляций
        tmp_corr = torch.softmax(weights, dim=-1)

        # агрегация
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        Ускоренная версия автокорреляции (выполненная в стиле batch-normalization).
        Предназначена для фазы вывода (инференса).
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        # инициализация индексов
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0) \
            .repeat(batch, head, channel, 1).to(values.device)

        # поиск top-k значений
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)

        # обновление матрицы корреляций
        tmp_corr = torch.softmax(weights, dim=-1)

        # агрегация
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def forward(self, queries: torch.Tensor, keys: torch.Tensor,
                values: torch.Tensor, attn_mask: torch.Tensor = None) -> tuple:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # поиск зависимостей на основе периодов через Быстрое преобразование Фурье (FFT)
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)

        # агрегация временных задержек
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    """
    Обертка для многоголового (Multi-head) слоя автокорреляции.
    Выполняет проецирование векторов запросов, ключей и значений.
    """

    def __init__(self, correlation: nn.Module, d_model: int, n_heads: int,
                 d_keys: int = None, d_values: int = None):
        super(AutoCorrelationLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries: torch.Tensor, keys: torch.Tensor,
                values: torch.Tensor, attn_mask: torch.Tensor = None) -> tuple:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


# =========================================================================
# БАЗОВЫЕ КОМПОНЕНТЫ И СБОРКА АРХИТЕКТУРЫ
# =========================================================================

class AutoformerEncoderLayer(nn.Module):
    """
    Слой энкодера Autoformer со встроенной декомпозицией ряда.
    """

    def __init__(self, d_model: int, n_heads: int = 8, factor: int = 1, dropout: float = 0.1):
        super().__init__()
        self.attention = AutoCorrelationLayer(
            AutoCorrelation(mask_flag=False, factor=factor, attention_dropout=dropout, output_attention=False),
            d_model, n_heads)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model * 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_model * 4, out_channels=d_model, kernel_size=1, bias=False)

        self.decomp1 = SeriesDecomposition(kernel_size=KERNEL_SIZE)
        self.decomp2 = SeriesDecomposition(kernel_size=KERNEL_SIZE)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_x, _ = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)

        y = x
        y = self.dropout(F.relu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        x, _ = self.decomp2(x + y)
        return x


class AutoformerDecoderLayer(nn.Module):
    """
    Слой декодера Autoformer со встроенной декомпозицией ряда и накоплением тренда.
    """

    def __init__(self, d_model: int, c_out: int, n_heads: int = 8, factor: int = 1, dropout: float = 0.1):
        super().__init__()
        self.self_attention = AutoCorrelationLayer(
            AutoCorrelation(mask_flag=True, factor=factor, attention_dropout=dropout, output_attention=False),
            d_model, n_heads)
        self.cross_attention = AutoCorrelationLayer(
            AutoCorrelation(mask_flag=False, factor=factor, attention_dropout=dropout, output_attention=False),
            d_model, n_heads)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model * 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_model * 4, out_channels=d_model, kernel_size=1, bias=False)

        self.decomp1 = SeriesDecomposition(KERNEL_SIZE)
        self.decomp2 = SeriesDecomposition(KERNEL_SIZE)
        self.decomp3 = SeriesDecomposition(KERNEL_SIZE)
        self.dropout = nn.Dropout(dropout)

        # Сглаживающая проекция тренда
        self.projection = nn.Conv1d(
            in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1,
            padding=1, padding_mode='circular', bias=False
        )

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> tuple:
        x_att, _ = self.self_attention(x, x, x)
        x = x + self.dropout(x_att)
        x, trend1 = self.decomp1(x)

        c_att, _ = self.cross_attention(x, cross, cross)
        x = x + self.dropout(c_att)
        x, trend2 = self.decomp2(x)

        y = x
        y = self.dropout(F.relu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)

        return x, residual_trend


class AutoformerFull(nn.Module):
    """
    Полная архитектура модели Autoformer для прогнозирования временных рядов.
    """

    def __init__(self, seq_len: int, pred_len: int, d_model: int = 256,
                 n_heads: int = 8, c_out: int = 1, e_layers: int = 2, d_layers: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = seq_len // 2

        self.decomp = SeriesDecomposition(KERNEL_SIZE)

        self.enc_embedding = DataEmbedding_wo_pos(c_in=c_out, d_model=d_model)
        self.dec_embedding = DataEmbedding_wo_pos(c_in=c_out, d_model=d_model)

        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model, n_heads=n_heads) for _ in range(e_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            AutoformerDecoderLayer(d_model, c_out, n_heads=n_heads) for _ in range(d_layers)
        ])

        self.out_projection = nn.Linear(d_model, c_out)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели.
        """
        # Вычисляем среднее по историческому окну для инициализации будущего тренда
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[2]], device=x_enc.device)

        # Декомпозиция входа
        seasonal_init, trend_init = self.decomp(x_enc)

        # Подготовка входов для декодера
        dec_trend = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        dec_seasonal = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # Проход через Энкодер
        enc_out = self.enc_embedding(x_enc)
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)

        # Проход через Декодер
        dec_out = self.dec_embedding(dec_seasonal)
        trend_part = dec_trend  # Инициализируем тренд

        for layer in self.decoder_layers:
            dec_out, trend_update = layer(dec_out, enc_out)
            trend_part = trend_part + trend_update  # Накапливаем изменения тренда

        # 6. Финальная сборка
        seasonal_part = self.out_projection(dec_out)
        final_out = trend_part + seasonal_part

        return final_out[:, -self.pred_len:, :]  # [B, L, D]


# =========================================================================
# ЦИКЛ ОБУЧЕНИЯ И ТЕСТИРОВАНИЯ
# =========================================================================

def run_autoformer() -> None:
    """
    Запускает полный пайплайн модели Autoformer:
    подготовка данных, обучение, инференс и логирование метрик.
    """
    logging.info("=== ЗАПУСК ОБУЧЕНИЯ: Autoformer ===")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    loaders_data = get_prepared_dataloaders(
        DATASET_PATH, TARGET_COL, DATE_COL, SEQ_LEN, PRED_LEN,
        BATCH_SIZE, VAL_RATIO, DATA_SLICE_SIZE
    )

    train_loader, val_loader, test_dataset, scaler, train_data_inv = loaders_data

    scaler_path = os.path.join(WEIGHTS_DIR, 'global_scaler.pkl')
    joblib.dump(scaler, scaler_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Используемое устройство: {device}")

    model = AutoformerFull(seq_len=SEQ_LEN, pred_len=PRED_LEN, d_model=256, n_heads=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    weights_path = os.path.join(WEIGHTS_DIR, 'best_autoformer.pth')

    logging.info("Старт обучения Autoformer...")
    train_pytorch_model(
        model, train_loader, val_loader, criterion, optimizer,
        device, EPOCHS, PATIENCE, weights_path
    )

    # Инференс
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    sample_idx = 0
    test_x, test_y = test_dataset[sample_idx]
    with torch.no_grad():
        test_tensor = test_x.unsqueeze(0).to(device)
        pred_y = model(test_tensor).cpu().numpy()[0]

    true_y_inv = scaler.inverse_transform(test_y.numpy())
    pred_y_inv = np.clip(scaler.inverse_transform(pred_y), a_min=(None if ACTIVE_DATASET == 'ett' else 0.0), a_max=None)

    calculate_and_log_metrics(
        true_y_inv[:, 0],
        pred_y_inv[:, 0],
        "Autoformer",
        y_train=train_data_inv[:, 0]
    )

    logging.info("=== Модуль Autoformer успешно завершен ===")


if __name__ == "__main__":
    run_autoformer()