"""
Обработчик города.

Извлекает название города и нормализует англоязычные варианты (Moscow → Москва).
"""

import re
import pandas as pd
from .base_handler import Handler


class CityHandler(Handler):
    """
    Обработчик для извлечения и кодирования названий городов.
    """
    
    def __init__(self):
        super().__init__()
        # Только англоязычные варианты → русские названия
        self.city_map = {
            "moscow": "Москва",
            "saint petersburg": "Санкт-Петербург",
            "spb": "Санкт-Петербург",
        }
    
    def _normalize_city(self, city: str) -> str:
        """Нормализовать англоязычные названия, остальные оставить как есть."""
        if pd.isna(city):
            return "Unknown"
        city_clean = city.strip().lower()
        return self.city_map.get(city_clean, city.strip())
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечь названия городов и выполнить one-hot кодирование.
        """
        def extract_city(val) -> str:
            if pd.isna(val):
                return "Unknown"
            parts = str(val).split(",")
            if parts:
                city = parts[0].strip()
                city = re.sub(r"[^а-яА-ЯёЁa-zA-Z\s-]", "", city).strip()
                return self._normalize_city(city) if city else "Unknown"
            return "Unknown"
        
        df["city"] = df["Город"].apply(extract_city)
        top_cities = df["city"].value_counts().nlargest(10).index.tolist()
        df["city"] = df["city"].apply(lambda x: x if x in top_cities else "Other")
        df = pd.get_dummies(df, columns=["city"], prefix="city", drop_first=True)
        return df