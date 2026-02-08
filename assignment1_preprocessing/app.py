#!/usr/bin/env python3
"""
Точка входа в приложение обработки данных hh.ru.
Если путь не указан — ищет hh.csv в корне репозитория.
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path
from pipeline import DataPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def find_hh_csv() -> Path | None:
    """Автоматически найти hh.csv в корне репозитория."""
    candidates = [
        Path("../hh.csv"),
        Path("../../hh.csv"),
        Path("hh.csv"),
        Path.home() / "Downloads" / "hh.csv"
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def main() -> None:
    # Определяем путь к CSV
    if len(sys.argv) == 2:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = find_hh_csv()
        if csv_path is None:
            logger.error("Файл hh.csv не найден. Укажите путь явно:")
            logger.error("  python app.py путь/к/hh.csv")
            sys.exit(1)
        logger.info(f"Автоматически найден hh.csv: {csv_path}")
    
    if not csv_path.exists():
        logger.error(f"Файл не найден: {csv_path}")
        sys.exit(1)
    
    try:
        pipeline = DataPipeline()
        x_data, y_data = pipeline.process(str(csv_path))
        
        output_dir = csv_path.parent
        np.save(output_dir / "x_data.npy", x_data)
        np.save(output_dir / "y_data.npy", y_data)
        
        logger.info(f"✓ Сохранены x_data.npy ({x_data.shape}) и y_data.npy ({y_data.shape})")
    except Exception as e:
        logger.exception(f"Ошибка обработки: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()