"""
Главный управляющий скрипт пайплайна прогнозирования энергопотребления.
Автоматизирует загрузку данных, обучение моделей и генерацию сравнительных дашбордов.
"""

import os
import sys
import zipfile
import subprocess
import logging

import requests
import pandas as pd

# Импортируем настройки из конфигурационного файла
from config import DATASETS, ACTIVE_DATASET


# Настройка логирования для вывода в консоль
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)


def download_ett(target_path: str) -> bool:
    """
    Автоматическое скачивание ETTh1 с официального репозитория ETDataset.

    Args:
        target_path (str): Путь для сохранения итогового CSV-файла.

    Returns:
        bool: True, если скачивание прошло успешно, иначе False.
    """
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    logging.info("Скачивание эталонного датасета ETTh1...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(target_path, 'wb') as f:
            f.write(response.content)

        logging.info(f"Датасет ETTh1 успешно скачан и готов к работе: {target_path}")
        return True
    except Exception as e:
        logging.error(f"Ошибка при скачивании ETTh1: {e}")
        return False


def download_uci_steel(target_path: str) -> bool:
    """
    Автоматическое скачивание Steel Industry через официальный API UCI
    и агрегация до 1 часа.

    Args:
        target_path (str): Путь для сохранения итогового CSV-файла.

    Returns:
        bool: True, если загрузка и обработка прошли успешно, иначе False.
    """
    logging.info("Попытка загрузки Steel Industry через библиотеку ucimlrepo...")
    try:
        from ucimlrepo import fetch_ucirepo

        # Скачиваем датасет по его ID
        steel = fetch_ucirepo(id=851)

        # Забираем оригинальный полный датафрейм со всеми колонками
        df = steel.data.original

        # Преобразуем колонку с датами в правильный формат и делаем её индексом
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
        df.set_index('date', inplace=True)

        logging.info("Сжатие сырых 15-минутных данных до 1 часа...")
        # Оставляем только числовые колонки, усредняем и заполняем возможные пропуски
        numeric_cols = df.select_dtypes(include='number').columns
        df_hourly = df[numeric_cols].resample('1h').mean().ffill()

        # Сохраняем результат в target_path
        df_hourly.to_csv(target_path)
        logging.info(f"Датасет успешно загружен через API, сжат и готов к работе: {target_path}")
        return True

    except ImportError:
        logging.error("Библиотека ucimlrepo не найдена! Установите ее: pip install ucimlrepo")
        return False
    except Exception as e:
        logging.error(f"Ошибка при обработке Steel Industry: {e}")
        return False


def check_and_download_datasets() -> bool:
    """
    Проверка наличия данных для активного профиля и их автозагрузка при необходимости.

    Returns:
        bool: True, если данные готовы к работе, False в случае критической ошибки.
    """
    logging.info("=== АВТОЗАГРУЗКА И ПРОВЕРКА ДАТАСЕТОВ ===")
    logging.info(f"Активный профиль конфигурации: [{ACTIVE_DATASET.upper()}]")

    os.makedirs("data", exist_ok=True)

    if ACTIVE_DATASET not in DATASETS:
        logging.error(f"Профиль '{ACTIVE_DATASET}' не найден в config.py!")
        return False

    dataset_info = DATASETS[ACTIVE_DATASET]
    file_path = dataset_info['path']

    if os.path.exists(file_path):
        logging.info(f"Датасет {ACTIVE_DATASET} найден: {file_path}")
        return True

    logging.warning(f"Готовый файл {file_path} не найден. Запуск подготовки...")

    if ACTIVE_DATASET == 'ett':
        download_ett(file_path)
    elif ACTIVE_DATASET == 'steel':
        download_uci_steel(file_path)
    else:
        logging.error(f"Для профиля {ACTIVE_DATASET} не настроена обработка")
        return False

    # Финальная проверка
    if os.path.exists(file_path):
        logging.info(f"Файл {file_path} успешно подготовлен")
        return True
    else:
        logging.error(f"Не удалось подготовить датасет {ACTIVE_DATASET}. Проверьте пути")
        return False


def run_script(script_name: str) -> bool:
    """
    Запуск python-скриптов с обработкой ошибок.
    Использует текущее виртуальное окружение.

    Args:
        script_name (str): Имя исполняемого скрипта (например, 'dlinear.py').

    Returns:
        bool: True, если скрипт завершился успешно (код 0), иначе False.
    """
    logging.info(f"=== ЗАПУСК МОДУЛЯ: {script_name} ===")
    try:
        # Используем sys.executable, чтобы запускать скрипты в текущем виртуальном окружении
        subprocess.run([sys.executable, script_name], check=True)
        return True
    except subprocess.CalledProcessError:
        logging.error(f"КРИТИЧЕСКАЯ ОШИБКА: Модуль {script_name} завершился сбоем")
        return False


def main() -> None:
    """
    Главная функция запуска полного цикла исследования.
    Последовательно проверяет данные и запускает обучение всех архитектур.
    """
    logging.info("СТАРТ ПОЛНОГО ЦИКЛА ИССЛЕДОВАНИЯ")

    # Проверка данных
    if not check_and_download_datasets():
        logging.error("Остановка пайплайна: нет данных для обучения")
        sys.exit(1)

    logging.info("=== ДАННЫЕ ГОТОВЫ. ПЕРЕХОД К ПОСЛЕДОВАТЕЛЬНОМУ ОБУЧЕНИЮ ===")

    # Список модулей для запуска в строгом порядке
    pipeline_scripts = [
        "sarima_baseline.py",         # Базовая статистика
        "dlinear.py",                 # Линейная модель
        "hybrid_smart_arima_lstm.py", # Гибрид
        "autoformer.py",              # Трансформер
        "predict.py"                  # Генерация графиков и дашбордов
    ]

    # Запуск
    for script in pipeline_scripts:
        if not os.path.exists(script):
            logging.warning(f"Файл {script} не найден в директории, пропуск шага")
            continue

        success = run_script(script)
        if not success:
            logging.error("Выполнение пайплайна прервано из-за ошибки в дочернем модуле")
            sys.exit(1)

    logging.info("=== ПАЙПЛАЙН ИССЛЕДОВАНИЯ УСПЕШНО ЗАВЕРШЕН ===")
    logging.info("Все графики и итоговый дашборд сохранены в папку результатов")


if __name__ == "__main__":
    main()