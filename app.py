#!/usr/bin/env python3
"""
Точка входа приложения для предобработки данных hh.ru.

Использование:
    python app.py путь/к/hh.csv
    
Результат:
    x_data.npy - матрица признаков
    y_data.npy - вектор целевой переменной (зарплаты)
"""

import sys
import logging
import numpy as np
from pathlib import Path
from pipeline import DataPipeline


# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Основная точка входа приложения.
    
    Парсит аргументы командной строки, запускает пайплайн, сохраняет результаты.
    """
    if len(sys.argv) != 2:
        logger.error("Использование: python app.py путь/к/hh.csv")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        logger.error(f"Файл не найден: {csv_path}")
        sys.exit(1)
    
    try:
        logger.info(f"Обработка файла {csv_path}...")
        pipeline = DataPipeline()
        x_data, y_data = pipeline.process(str(csv_path))
        
        output_dir = csv_path.parent
        np.save(output_dir / "x_data.npy", x_data)
        np.save(output_dir / "y_data.npy", y_data)
        
        logger.info(f"✓ Сохранены x_data.npy ({x_data.shape}) и y_data.npy ({y_data.shape})")
        logger.info(f"  Признаков: {x_data.shape[1]} столбцов")
        logger.info(f"  Выборок: {len(y_data)} резюме обработано")
        
    except Exception as e:
        logger.exception(f"Ошибка обработки: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()