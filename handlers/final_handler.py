"""
Финальный обработчик для подготовки выходных numpy-массивов.

Извлекает матрицу признаков и целевую переменную для обучения моделей.
"""

import numpy as np
import pandas as pd
from .base_handler import Handler


class FinalHandler(Handler):
    """
    Финальный обработчик, подготавливающий выходные массивы для машинного обучения.
    
    Собирает все обработанные признаки и целевую переменную в numpy-массивы.
    """
    
    def __init__(self) -> None:
        """Инициализация с пустыми выходными массивами."""
        super().__init__()
        self.x_data = None
        self.y_data = None

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечь матрицу признаков и вектор целевой переменной из обработанного DataFrame.
        
        Аргументы:
            df: Полностью обработанный DataFrame со всеми признаками
            
        Возвращает:
            Исходный DataFrame (без изменений)
        """
        numeric_cols = ["age", "experience_years"]
        city_cols = [col for col in df.columns if col.startswith("city_")]
        feature_cols = numeric_cols + city_cols
        
        self.x_data = df[feature_cols].fillna(0).astype(float).values
        self.y_data = df["salary_num"].values.astype(float)
        return df
    
    def get_outputs(self) -> tuple:
        """
        Получить обработанные выходные массивы.
        
        Возвращает:
            Кортеж из (x_data, y_data) numpy-массивов
            
        Вызывает:
            RuntimeError: Если process() не был вызван ранее
        """
        if self.x_data is None or self.y_data is None:
            raise RuntimeError("Сначала обработайте данные через process()")
        return self.x_data, self.y_data