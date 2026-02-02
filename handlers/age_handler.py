"""
Обработчик возраста.

Извлекает возраст из строк вида "Мужчина , 42 года , родился 6 октября 1976".
"""

import re
import pandas as pd
from .base_handler import Handler


class AgeHandler(Handler):
    """
    Обработчик для извлечения возраста из текста профиля.
    
    Парсит строки, содержащие кириллические символы и неразрывные пробелы.
    """
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечь значения возраста из столбца 'Пол, возраст'.
        
        Аргументы:
            df: DataFrame с сырыми строками возраста
            
        Возвращает:
            DataFrame с новым столбцом 'age' (пропуски заполнены медианой)
        """
        def parse_age(val) -> int | None:
            if pd.isna(val):
                return None
            match = re.search(r"(\d+)\s*[гл]", str(val))
            if match:
                return int(match.group(1))
            return None
        
        df["age"] = df["Пол, возраст"].apply(parse_age)
        df["age"] = df["age"].fillna(df["age"].median())
        return df