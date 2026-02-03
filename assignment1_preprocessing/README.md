# HH.ru Data Preprocessing Pipeline

Пайплайн обработки данных с сайта hh.ru с использованием паттерна "Цепочка ответственности" (Chain of Responsibility).

## Задача
Преобразование сырых CSV-данных резюме в числовые массивы `.npy`, готовые для машинного обучения.

## Структура проекта
hh-preprocessing/\
├── app.py&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;# Точка входа приложения\
├── pipeline.py&emsp;&emsp;&emsp;&ensp;# Сборка цепочки обработчиков\
├── requirements.txt&emsp;# Зависимости проекта\
├── README.md&emsp;&emsp;&ensp;# Документация\
├── .gitignore&emsp;&emsp;&emsp;&ensp;# Исключения для системы контроля версий\
└── handlers/&emsp;&emsp;&emsp;&ensp;# Модуль обработчиков данных\
&emsp;&emsp;├── __init__.py\
&emsp;&emsp;├── base_handler.py&emsp;&emsp;&emsp;&ensp;# Абстрактный базовый класс\
&emsp;&emsp;├── salary_handler.py&emsp;&emsp;&emsp;# Парсинг зарплаты\
&emsp;&emsp;├── age_handler.py&emsp;&emsp;&emsp;&emsp;# Извлечение возраста\
&emsp;&emsp;├── experience_handler.py&emsp;# Парсинг опыта работы\
&emsp;&emsp;├── city_handler.py&emsp;&emsp;&emsp;&emsp;# Обработка города (one-hot кодирование)\
&emsp;&emsp;└── final_handler.py&emsp;&emsp;&emsp;&ensp;# Формирование финальных массивов

## Установка
```bash
pip install -r requirements.txt
```

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
