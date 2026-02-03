#!/usr/bin/env python3
"""
Скрипт обучения модели линейной регрессии.

Сохраняет веса модели в папку resources/.
"""

import sys
import logging
import numpy as np
from pathlib import Path
from model import LinearRegressionModel


# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def main() -> None:
    x_path = Path("./assignment1_preprocessing/x_data.npy")
    y_path = Path("./assignment1_preprocessing/y_data.npy")
    
    if not x_path.exists() or not y_path.exists():
        logger.error("Файлы x_data.npy и y_data.npy не найдены")
        sys.exit(1)
    
    try:
        logger.info("Загрузка данных...")
        X = np.load(x_path)
        y = np.load(y_path)
        
        logger.info(f"Обучение модели на {len(X)} образцах...")
        model = LinearRegressionModel()
        model.fit(X, y)
        
        logger.info("Сохранение весов в resources/...")
        model.save("resources")
        
        logger.info("✓ Модель успешно обучена")
    except Exception as e:
        logger.exception(f"Ошибка обучения: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()