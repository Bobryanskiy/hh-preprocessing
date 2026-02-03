#!/usr/bin/env python3
"""
Точка входа для предсказания зарплат.

Использование:
    python app.py путь/к/x_data.npy
    
Вывод:
    Список зарплат в рублях, по одной на строку (только числа в stdout).
"""

import sys
import logging
import numpy as np
from pathlib import Path
from model import LinearRegressionModel


# Настройка логгера (только для ошибок, в stderr)
logging.basicConfig(
    level=logging.ERROR,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) != 2:
        logger.error("Использование: python app.py путь/к/x_data.npy")
        sys.exit(1)
    
    x_path = Path(sys.argv[1])
    if not x_path.exists():
        logger.error(f"Файл не найден: {x_path}")
        sys.exit(1)
    
    # Загрузка данных
    try:
        X = np.load(x_path)
    except Exception as e:
        logger.exception(f"Ошибка загрузки данных: {e}")
        sys.exit(1)
    
    # Загрузка модели
    model = LinearRegressionModel()
    try:
        model.load("resources")
    except Exception as e:
        logger.exception(f"Ошибка загрузки модели: {e}")
        logger.error("Сначала обучите модель: python train.py")
        sys.exit(1)
    
    # Предсказание и вывод РЕЗУЛЬТАТА в stdout (только числа!)
    try:
        y_pred = model.predict(X)
        for salary in y_pred:
            # ВАЖНО: зарплаты выводим через print() в stdout
            # Это требование задания — нейросеть ожидает чистые числа
            print(f"{salary:.2f}")
    except Exception as e:
        logger.exception(f"Ошибка предсказания: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()