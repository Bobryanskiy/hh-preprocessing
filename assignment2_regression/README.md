# Предсказание зарплат по данным hh.ru (Линейная регрессия)

Модель линейной регрессии для предсказания зарплат на основе обработанных данных резюме с hh.ru.

## Структура проекта
assignment2_regression/
├── app.py          # Предсказание: python app.py x_data.npy
├── train.py        # Обучение модели (выполняется 1 раз)
├── model.py        # Реализация линейной регрессии
├── requirements.txt
├── README.md
├── .gitignore
└── resources/      # Сохранённые веса модели
    ├── weights.npy
    └── bias.npy

## Установка
```bash
pip install -r requirements.txt