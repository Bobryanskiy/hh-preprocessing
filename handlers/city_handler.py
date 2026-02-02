"""
Обработчик города.

Извлекает название города из сложных строк и выполняет one-hot кодирование
для 10 самых популярных городов, остальные группирует в 'Other'.
"""

import re
import pandas as pd
from .base_handler import Handler


class CityHandler(Handler):
    """
    Обработчик для извлечения и кодирования названий городов.
    
    Извлекает название города из комплексных строк и выполняет
    one-hot кодирование для наиболее частых значений.
    """
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечь названия городов и выполнить one-hot кодирование.
        
        Аргументы:
            df: DataFrame с сырыми строками городов
            
        Возвращает:
            DataFrame с one-hot закодированными столбцами городов (префикс 'city_')
        """
        def extract_city(val) -> str:
            if pd.isna(val):
                return "Unknown"
            parts = str(val).split(",")
            if parts:
                city = parts[0].strip()
                city = re.sub(r"[^а-яА-ЯёЁ\s-]", "", city).strip()
                return city if city else "Unknown"
            return "Unknown"
        
        df["city"] = df["Город"].apply(extract_city)
        top_cities = df["city"].value_counts().nlargest(10).index.tolist()
        df["city"] = df["city"].apply(lambda x: x if x in top_cities else "Other")
        df = pd.get_dummies(df, columns=["city"], prefix="city", drop_first=True)
        return df