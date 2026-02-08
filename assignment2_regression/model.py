"""
Модуль линейной регрессии.

Реализует линейную регрессию с сохранением и загрузкой весов.
Использует нормальное уравнение для обучения без итераций.
"""

import numpy as np
from pathlib import Path


class LinearRegressionModel:
    """Класс линейной регрессии с ручным управлением весами."""
    
    def __init__(self) -> None:
        """Инициализация модели с пустыми весами."""
        self.weights = None
        self.bias = 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Обучить модель методом наименьших квадратов.
        
        Использует нормальное уравнение: θ = (X^T X)^(-1) X^T y
        
        Аргументы:
            X: Матрица признаков (n_samples, n_features)
            y: Вектор целевой переменной (n_samples,)
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.bias = theta[0]
        self.weights = theta[1:]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказать зарплаты по признакам.
        
        Аргументы:
            X: Матрица признаков (n_samples, n_features)
            
        Возвращает:
            Вектор предсказанных зарплат (n_samples,)
            
        Вызывает:
            RuntimeError: Если модель не обучена
        """
        if self.weights is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод fit().")
        return X.dot(self.weights) + self.bias
    
    def save(self, resources_dir: str | Path) -> None:
        """
        Сохранить веса модели в папку resources/.
        
        Аргументы:
            resources_dir: Путь к папке для сохранения весов
        """
        resources_dir = Path(resources_dir)
        resources_dir.mkdir(exist_ok=True)
        np.save(resources_dir / "weights.npy", self.weights)
        np.save(resources_dir / "bias.npy", np.array([self.bias]))
    
    def load(self, resources_dir: str | Path) -> None:
        """
        Загрузить веса модели из папки resources/.
        
        Аргументы:
            resources_dir: Путь к папке с весами
            
        Вызывает:
            FileNotFoundError: Если файлы весов не найдены
        """
        resources_dir = Path(resources_dir)
        weights_path = resources_dir / "weights.npy"
        bias_path = resources_dir / "bias.npy"
        
        if not weights_path.exists() or not bias_path.exists():
            raise FileNotFoundError(
                f"Файлы весов не найдены в {resources_dir}. "
                "Сначала обучите модель: python train.py"
            )
        
        self.weights = np.load(weights_path)
        self.bias = np.load(bias_path)[0]