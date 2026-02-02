"""
Обработчик зарплаты.

Извлекает числовые значения зарплаты из строк вида "60 000 руб." или "от 100 000 руб."
Корректно обрабатывает неразрывные пробелы (\xa0) и различные форматы.
"""

import re
import pandas as pd
from typing import Optional
from .base_handler import Handler


class SalaryHandler(Handler):
    """
    Обработчик для извлечения и нормализации значений зарплаты.
    
    Парсит строки с зарплатой, удаляет неразрывные пробелы и преобразует
    в числовые значения в рублях.
    """
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечь числовые значения зарплаты из столбца 'ЗП'.
        
        Аргументы:
            df: DataFrame с сырыми строками зарплат
            
        Возвращает:
            DataFrame с новым столбцом 'salary_num', содержащим float-значения
        """
        def parse_salary(val) -> Optional[float]:
            if pd.isna(val) or not isinstance(val, str):
                return None
            # Находим первую числовую последовательность (игнорируем "от", "до")
            match = re.search(r"(\d[\d\s\xa0]*)", val)
            if match:
                clean = re.sub(r"[\s\xa0]", "", match.group(1))
                if clean.isdigit():
                    return float(clean)
            return None
        
        df["salary_num"] = df["ЗП"].apply(parse_salary)
        df = df.dropna(subset=["salary_num"])
        return df