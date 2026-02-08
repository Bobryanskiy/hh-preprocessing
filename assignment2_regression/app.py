#!/usr/bin/env python3
"""
Точка входа для предсказания зарплат.

Использование (опционально):
    python app.py путь/к/x_data.npy
    
Если путь не указан — ищет x_data.npy автоматически.
Вывод: список зарплат, по одной на строку (только числа в stdout).
"""

import sys
import logging
import numpy as np
from pathlib import Path
from model import LinearRegressionModel


logging.basicConfig(
    level=logging.ERROR,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def find_x_data() -> Path | None:
    """Найти x_data.npy в корне или в папке задания №1."""
    candidates = [
        Path("../x_data.npy"),
        Path("../../x_data.npy"),
        Path("../assignment1_preprocessing/x_data.npy"),
        Path("x_data.npy"),
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    return None


def main() -> None:
    # Определение пути к данным
    if len(sys.argv) == 2:
        x_path = Path(sys.argv[1])
    else:
        x_path = find_x_data()
        if x_path is None:
            logger.error("Файл x_data.npy не найден.")
            logger.error("Укажите путь явно: python app.py путь/к/x_data.npy")
            logger.error("Или сначала выполните задание №1")
            sys.exit(1)
    
    if not x_path.exists():
        logger.error(f"Файл не найден: {x_path}")
        sys.exit(1)
    
    # Загрузка данных
    try:
        X = np.load(x_path)
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        sys.exit(1)
    
    # Загрузка модели
    model = LinearRegressionModel()
    try:
        model.load("resources")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        logger.error("Сначала обучите модель: python train.py")
        sys.exit(1)
    
    # Предсказание и вывод РЕЗУЛЬТАТА в stdout (только числа!)
    try:
        y_pred = model.predict(X)
        for salary in y_pred:
            print(f"{salary:.2f}")
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()