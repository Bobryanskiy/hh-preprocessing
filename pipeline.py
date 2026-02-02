"""
Пайплайн обработки данных с использованием паттерна "Цепочка ответственности".

Координация последовательности обработчиков для преобразования сырых CSV-данных
в чистые numpy-массивы, готовые для машинного обучения.
"""

import pandas as pd
import numpy as np
from handlers.salary_handler import SalaryHandler
from handlers.age_handler import AgeHandler
from handlers.experience_handler import ExperienceHandler
from handlers.city_handler import CityHandler
from handlers.final_handler import FinalHandler


class DataPipeline:
    """
    Основной класс пайплайна, управляющий обработкой данных.
    
    Собирает и запускает цепочку обработчиков для трансформации данных.
    """
    
    def __init__(self) -> None:
        """Инициализация пайплайна с построением цепочки обработчиков."""
        self.final_handler = FinalHandler()
        self.first_handler = SalaryHandler()
        (self.first_handler
         .set_next(AgeHandler())
         .set_next(ExperienceHandler())
         .set_next(CityHandler())
         .set_next(self.final_handler))
    
    def process(self, csv_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Выполнить полную обработку данных через пайплайн.
        
        Аргументы:
            csv_path: Путь к входному CSV-файлу
            
        Возвращает:
            Кортеж из (x_data, y_data) numpy-массивов
            
        Вызывает:
            FileNotFoundError: Если CSV-файл не найден
            pd.errors.ParserError: При ошибке парсинга CSV
        """
        df = pd.read_csv(csv_path)
        df = self.first_handler.process(df)
        return self.final_handler.get_outputs()