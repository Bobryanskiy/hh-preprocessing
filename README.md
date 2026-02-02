# HH.ru Data Preprocessing Pipeline

Пайплайн обработки данных с сайта hh.ru с использованием паттерна "Цепочка ответственности" (Chain of Responsibility).

## Задача
Преобразование сырых CSV-данных резюме в числовые массивы `.npy`, готовые для машинного обучения.

## Структура проекта
hh-preprocessing/
  app.py
  pipeline.py
  requirements.txt
  README.md
  .gitignore
  handlers/
    __init__.py
    base_handler.py
    salary_handler.py
    age_handler.py
    experience_handler.py
    city_handler.py
    final_handler.py

## Использование
python app.py path/to/hh.csv

На выходе создаются файлы:
- x_data.npy — матрица признаков (возраст, опыт, города)
- y_data.npy — вектор целевой переменной (зарплаты в рублях)

## Паттерн проектирования
Реализован паттерн **Цепочка ответственности**:
- Каждый обработчик отвечает за одну задачу
- Обработчики связываются через `set_next()`
- Данные последовательно проходят через всю цепочку

## Требования
- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
