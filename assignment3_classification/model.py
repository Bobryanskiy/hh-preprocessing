"""
Модуль классификации уровней разработчиков (junior/middle/senior).

Строгая фильтрация ТОЛЬКО настоящих разработчиков (программистов).
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib


class DeveloperLevelClassifier:
    """Классификатор уровня разработчика."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.classes_ = ["junior", "middle", "senior"]
    
    def _is_developer(self, title: str) -> bool:
        """
        СТРОГАЯ фильтрация ТОЛЬКО настоящих разработчиков.
        
        Принцип: должность ДОЛЖНА содержать ключевые слова разработки
        И НЕ ДОЛЖНА содержать слова не-разработчиков.
        """
        if pd.isna(title):
            return False
        
        title_lower = str(title).lower().strip()
        
        # Обязательные ключевые слова разработки
        dev_keywords = [
            "программист", "разработчик", "прогер", "разраб",
            "frontend", "front-end", "front end",
            "backend", "back-end", "back end",
            "fullstack", "full-stack", "full stack",
            "web-программист", "веб-программист",
            "1с", "1 с", "1с:", "1 с:",
            "java", "python", "c#", "c++", "c/c++", "javascript", "js",
            "typescript", "ts", "go", "golang", "rust", "ruby", "php",
            "flutter", "react", "vue", "angular", "django", "flask",
            "spring", "dotnet", ".net", "kotlin", "swift", "scala"
        ]
        
        # Запрещённые слова (не-разработчики)
        non_dev_keywords = [
            "администратор", "админ", "сисадмин", "системный администратор",
            "инженер", "техник", "монтажник", "электрик", "механик",
            "менеджер", "руководитель", "директор", "начальник",
            "аналитик", "бизнес-аналитик", "системный аналитик",
            "тестировщик", "qa", "автотест", "ручное тестирование",
            "дизайнер", "верстальщик", "маркетолог", "контент",
            "продаж", "поддержка", "консультант", "оператор",
            "архитектор", "девопс", "администрирование", "сопровождение"
        ]
        
        # Проверка: должно быть ключевое слово разработки
        has_dev = any(kw in title_lower for kw in dev_keywords)
        
        # Проверка: НЕ должно быть запрещённых слов (кроме особых случаев)
        has_non_dev = any(kw in title_lower for kw in non_dev_keywords)
        
        # Особый случай: "инженер-программист" разрешён
        is_programmer_engineer = "инженер-программист" in title_lower
        
        # Логика фильтрации
        if not has_dev:
            return False  # Нет ключевых слов разработки → не разработчик
        
        if has_non_dev and not is_programmer_engineer:
            return False  # Есть запрещённые слова → не разработчик
        
        return True
    
    def _extract_level(self, title: str) -> str | None:
        """Извлечь уровень ТОЛЬКО по ключевым словам (без опыта!)."""
        if pd.isna(title):
            return None
        
        title_lower = str(title).lower()
        
        # Junior
        if any(kw in title_lower for kw in [
            "junior", "младший", "стажер", "стажёр", "trainee", "intern", "начинающий"
        ]):
            return "junior"
        
        # Senior
        if any(kw in title_lower for kw in [
            "senior", "lead", "главный", "техлид", "архитектор", "ведущий"
        ]):
            return "senior"
        
        # Middle — только если явно указано
        if any(kw in title_lower for kw in ["middle", "миддл", "мидл"]):
            return "middle"
        
        # Без явного указания уровня — НЕ размечаем
        return None
    
    def label_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Разметить уровень разработчика для каждого резюме."""
        # Строгая фильтрация ТОЛЬКО разработчиков
        df_dev = df[df["Ищет работу на должность:"].apply(self._is_developer)].copy()
        
        if len(df_dev) == 0:
            raise ValueError("Не найдено резюме настоящих разработчиков. Проверьте фильтрацию.")
        
        # Извлечение опыта (для признаков, НЕ для разметки уровня!)
        def parse_experience(val) -> float:
            if pd.isna(val):
                return 0.0
            text = str(val).replace("\xa0", " ")
            match = re.search(r"Опыт работы\s+(\d+)\s+лет?\s+(\d+)\s+месяц", text)
            if match:
                return int(match.group(1)) + int(match.group(2)) / 12.0
            match = re.search(r"Опыт работы\s+(\d+)\s+лет?", text)
            if match:
                return float(match.group(1))
            return 0.0
        
        df_dev["experience_years"] = df_dev["Опыт (двойное нажатие для полной версии)"].apply(
            parse_experience
        )
        
        # Разметка уровня ТОЛЬКО по ключевым словам в должности
        df_dev["level"] = df_dev["Ищет работу на должность:"].apply(self._extract_level)
        
        # Убираем резюме без явного уровня (чтобы не добавлять шум)
        df_dev = df_dev[df_dev["level"].notna()]
        
        if len(df_dev) == 0:
            raise ValueError("Нет резюме с явным указанием уровня (junior/middle/senior).")
        
        return df_dev[["level", "experience_years", "Город", "ЗП"]]
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовить признаки и целевую переменную."""
        def parse_salary(val) -> float:
            if pd.isna(val):
                return 0.0
            match = re.search(r"(\d[\d\s\xa0]*)", str(val))
            if match:
                clean = re.sub(r"[\s\xa0]", "", match.group(1))
                if clean.isdigit():
                    return float(clean)
            return 0.0
        
        def extract_city(val) -> str:
            if pd.isna(val):
                return "Unknown"
            parts = str(val).split(",")
            if parts:
                city = parts[0].strip()
                city = re.sub(r"[^а-яА-ЯёЁ\s-]", "", city).strip()
                return city if city else "Unknown"
            return "Unknown"
        
        df["salary_num"] = df["ЗП"].apply(parse_salary)
        df["city"] = df["Город"].apply(extract_city)
        
        # Пайплайн предобработки
        num_features = ["experience_years", "salary_num"]
        cat_features = ["city"]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ]
        )
        
        X = preprocessor.fit_transform(df)
        y = df["level"].values
        
        self.preprocessor = preprocessor
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Обучить классификатор."""
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y
        )
        weights_dict = dict(zip(classes, class_weights))
        
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            class_weight=weights_dict,
            random_state=42
        )
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказать уровень."""
        if self.model is None:
            raise RuntimeError("Модель не обучена")
        return self.model.predict(X)
    
    def save(self, path: str | Path) -> None:
        """Сохранить модель."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "preprocessor": self.preprocessor
        }, path)
    
    def load(self, path: str | Path) -> None:
        """Загрузить модель."""
        data = joblib.load(path)
        self.model = data["model"]
        self.preprocessor = data["preprocessor"]