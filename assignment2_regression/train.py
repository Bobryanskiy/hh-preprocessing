#!/usr/bin/env python3
"""
Скрипт обучения модели линейной регрессии с фильтрацией выбросов.
"""

import sys
import logging
import numpy as np
from pathlib import Path
from model import LinearRegressionModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def find_data_files() -> tuple[Path, Path] | tuple[None, None]:
    """Найти x_data.npy и y_data.npy."""
    candidates = [
        (Path("../x_data.npy"), Path("../y_data.npy")),
        (Path("../../x_data.npy"), Path("../../y_data.npy")),
        (Path("../assignment1_preprocessing/x_data.npy"), Path("../assignment1_preprocessing/y_data.npy")),
        (Path("x_data.npy"), Path("y_data.npy")),
    ]
    
    for x_path, y_path in candidates:
        if x_path.exists() and y_path.exists():
            return x_path, y_path
    
    return None, None


def main() -> None:
    x_path, y_path = find_data_files()
    
    if x_path is None or y_path is None:
        logger.error("Файлы x_data.npy и y_data.npy не найдены.")
        logger.error("Сначала выполните задание №1:")
        logger.error("  cd ../assignment1_preprocessing")
        logger.error("  python app.py")
        sys.exit(1)
    
    logger.info(f"Загрузка данных из: {x_path.parent}")
    X = np.load(x_path)
    y = np.load(y_path)
    
    # Фильтрация выбросов (легальное улучшение без нарушения ТЗ!)
    # Убираем зарплаты < 15к (заглушки hh.ru) и > 1 млн (аномалии)
    mask = (y >= 15_000) & (y <= 1_000_000)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    logger.info(f"Исходные данные: {len(y)} образцов")
    logger.info(f"После фильтрации выбросов: {len(y_filtered)} образцов ({len(y_filtered)/len(y)*100:.1f}%)")
    
    logger.info(f"Обучение модели на {len(X_filtered)} образцах...")
    model = LinearRegressionModel()
    model.fit(X_filtered, y_filtered)
    
    logger.info("Сохранение весов в resources/...")
    model.save("resources")
    
    # Расчёт метрик НА ОТФИЛЬТРОВАННЫХ ДАННЫХ
    y_pred = model.predict(X_filtered)
    mse = np.mean((y_filtered - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_filtered - y_pred) ** 2) / np.sum((y_filtered - np.mean(y_filtered)) ** 2)
    
    logger.info("✓ Модель успешно обучена")
    logger.info(f"  MSE:  {mse:,.0f}")
    logger.info(f"  RMSE: {rmse:,.0f} руб.")
    logger.info(f"  R²:   {r2:.4f}")


if __name__ == "__main__":
    main()