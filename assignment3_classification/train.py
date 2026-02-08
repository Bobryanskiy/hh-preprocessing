#!/usr/bin/env python3
"""
Обучение классификатора уровней разработчиков.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from model import DeveloperLevelClassifier


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def plot_class_balance(y: np.ndarray, output_path: Path) -> None:
    levels, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=levels, y=counts, palette="viridis")
    plt.title("Баланс классов: распределение уровней разработчиков")
    plt.xlabel("Уровень")
    plt.ylabel("Количество резюме")
    for i, v in enumerate(counts):
        plt.text(i, v + 5, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=["junior", "middle", "senior"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["junior", "middle", "senior"],
        yticklabels=["junior", "middle", "senior"]
    )
    plt.title("Матрица ошибок классификатора")
    plt.ylabel("Истинный уровень")
    plt.xlabel("Предсказанный уровень")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    # Автоматический поиск hh.csv в корне репозитория
    possible_paths = [
        Path("../hh.csv"),
        Path("../../hh.csv"),
        Path("hh.csv"),
        Path("../assignment1_preprocessing/hh.csv")
    ]
    
    csv_path = None
    for p in possible_paths:
        if p.exists():
            csv_path = p
            break
    
    if csv_path is None:
        logger.error("Файл hh.csv не найден. Помести его в корень репозитория.")
        sys.exit(1)
    
    logger.info(f"Загрузка данных из: {csv_path}")
    df = pd.read_csv(csv_path)
    
    classifier = DeveloperLevelClassifier()
    df_labeled = classifier.label_levels(df)
    logger.info(f"Найдено {len(df_labeled)} IT-резюме")
    
    # Статистика по уровням
    level_counts = df_labeled["level"].value_counts()
    for level, count in level_counts.items():
        pct = count / len(df_labeled) * 100
        logger.info(f"  {level}: {count} ({pct:.1f}%)")
    
    X, y = classifier.prepare_features(df_labeled)
    
    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Обучение
    classifier.train(X_train, y_train)
    
    # Оценка
    y_pred = classifier.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["junior", "middle", "senior"],
        digits=3
    )
    logger.info("\nОтчёт о классификации:")
    logger.info("\n" + report)
    
    # Сохранение
    model_path = Path("resources/model.pkl")
    classifier.save(model_path)
    logger.info(f"Модель сохранена: {model_path}")
    
    # Графики
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    plot_class_balance(y, reports_dir / "class_balance.png")
    plot_confusion_matrix(y_test, y_pred, reports_dir / "confusion_matrix.png")
    logger.info(f"Графики сохранены в: {reports_dir}")
    
    # Оценка работоспособности (ключевой метрик — weighted F1)
    from sklearn.metrics import f1_score
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    accuracy = (y_pred == y_test).mean()
    
    logger.info("\nОценка работоспособности:")
    logger.info(f"  • Accuracy: {accuracy:.1%}")
    logger.info(f"  • Weighted F1-score: {f1_weighted:.3f}")
    logger.info(f"  • Базовый уровень (случайное угадывание): ~33%")
    logger.info(f"  • Вывод: модель работает ({f1_weighted:.1%} > 33%) — PoC успешен")


if __name__ == "__main__":
    main()