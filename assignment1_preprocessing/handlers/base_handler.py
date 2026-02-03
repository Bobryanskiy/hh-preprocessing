"""
Базовый абстрактный класс для паттерна "Цепочка ответственности".

Определяет интерфейс для всех обработчиков данных в пайплайне.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional


class Handler(ABC):
    """
    Абстрактный базовый класс обработчика в цепочке ответственности.
    
    Каждый конкретный обработчик должен реализовать метод handle()
    и может делегировать обработку следующему обработчику в цепочке.
    """
    
    def __init__(self) -> None:
        """Инициализация обработчика без следующего звена."""
        self._next_handler: Optional["Handler"] = None

    def set_next(self, handler: "Handler") -> "Handler":
        """
        Установить следующий обработчик в цепочке.
        
        Аргументы:
            handler: Следующий обработчик для делегирования
            
        Возвращает:
            Следующий обработчик (для цепочки вызовов)
        """
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обработать данные в DataFrame.
        
        Аргументы:
            df: Входной DataFrame для обработки
            
        Возвращает:
            Обработанный DataFrame
            
        Вызывает:
            NotImplementedError: Должен быть реализован в дочерних классах
        """
        pass

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обработать данные текущим обработчиком и передать дальше по цепочке.
        
        Аргументы:
            df: DataFrame для обработки
            
        Возвращает:
            Полностью обработанный DataFrame после прохождения всей цепочки
        """
        df = self.handle(df)
        if self._next_handler:
            return self._next_handler.process(df)
        return df