"""
Обработчик опыта работы.

Извлекает продолжительность опыта работы из подробного текстового поля,
содержащего полную историю трудоустройства.
"""

import re
import pandas as pd
from .base_handler import Handler


class ExperienceHandler(Handler):
    """
    Обработчик для извлечения общего стажа работы в годах.
    
    Парсит строки в формате "Опыт работы X лет Y месяцев".
    """
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечь продолжительность опыта работы в годах из текстового поля.
        
        Аргументы:
            df: DataFrame с сырым текстом опыта работы
            
        Возвращает:
            DataFrame с новым столбцом 'experience_years' типа float
        """
        def parse_experience(val) -> float:
            if pd.isna(val):
                return 0.0
            text = str(val).replace("\xa0", " ")
            
            # Формат: "Опыт работы X лет Y месяцев"
            match = re.search(r"Опыт работы\s+(\d+)\s+лет?\s+(\d+)\s+месяц", text)
            if match:
                return int(match.group(1)) + int(match.group(2)) / 12.0
            
            # Формат: "Опыт работы X лет"
            match = re.search(r"Опыт работы\s+(\d+)\s+лет?", text)
            if match:
                return float(match.group(1))
            
            return 0.0
        
        df["experience_years"] = df["Опыт (двойное нажатие для полной версии)"].apply(
            parse_experience
        )
        return df