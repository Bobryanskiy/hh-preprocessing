# HH.ru Machine Learning Homework

Домашние задания по машинному обучению на данных с сайта hh.ru.

## 📋 Описание проекта

Проект включает три связанных задания по обработке и анализу данных резюме с портала hh.ru:

1. **Парсинг и анализ данных** — преобразование сырых CSV-данных в числовые массивы с использованием паттерна проектирования «Цепочка ответственности»
2. **Линейная регрессия** — предсказание зарплат на основе обработанных данных
3. **Классификация уровней разработчиков** — автоматическое определение уровня (junior/middle/senior) с построением графиков и отчётов

## 📁 Структура проекта

hh-preprocessing/\
&emsp;├── README.md&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Этот файл\
&emsp;├── .gitignore&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Исключения для Git\
&emsp;├── hh.csv&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Исходный датасет (должен быть здесь)\
&emsp;│\
&emsp;├── assignment1_preprocessing/&emsp;&emsp;&emsp;# Задание №1\
&emsp;│&emsp;&emsp;├── app.py&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Точка входа\
&emsp;│&emsp;&emsp;├── pipeline.py&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Сборка цепочки обработчиков\
&emsp;│&emsp;&emsp;├── requirements.txt&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Зависимости\
&emsp;│&emsp;&emsp;├── README.md&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Документация задания\
&emsp;│&emsp;&emsp;└── handlers/&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Обработчики данных\
&emsp;│&emsp;&emsp;&emsp;&emsp;├── __init__.py\
&emsp;│&emsp;&emsp;&emsp;&emsp;├── base_handler.py\
&emsp;│&emsp;&emsp;&emsp;&emsp;├── salary_handler.py\
&emsp;│&emsp;&emsp;&emsp;&emsp;├── age_handler.py\
&emsp;│&emsp;&emsp;&emsp;&emsp;├── experience_handler.py\
&emsp;│&emsp;&emsp;&emsp;&emsp;├── city_handler.py\
&emsp;│&emsp;&emsp;&emsp;&emsp;└── final_handler.py\
&emsp;│\
&emsp;├── assignment2_regression/&emsp;&emsp;&emsp;&emsp;&emsp;# Задание №2\
&emsp;│&emsp;&emsp;├── app.py&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Точка входа (предсказание)\
&emsp;│&emsp;&emsp;├── train.py&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Обучение модели\
&emsp;│&emsp;&emsp;├── model.py&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Реализация регрессии\
&emsp;│&emsp;&emsp;├── evaluate.py&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Оценка качества\
&emsp;│&emsp;&emsp;├── requirements.txt&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Зависимости\
&emsp;│&emsp;&emsp;├── README.md&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Документация задания\
&emsp;│&emsp;&emsp;└── resources/&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Сохранённые веса модели\
&emsp;│&emsp;&emsp;&emsp;&emsp;├── weights.npy\
&emsp;│&emsp;&emsp;&emsp;&emsp;└── bias.npy\
&emsp;│\
&emsp;└── assignment3_classification/&emsp;&emsp;&emsp;&emsp;# Задание №3\
&emsp;&emsp;&emsp;&emsp;├── train.py&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Обучение + отчёты\
&emsp;&emsp;&emsp;&emsp;├── model.py&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Классификатор\
&emsp;&emsp;&emsp;&emsp;├── requirements.txt&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Зависимости\
&emsp;&emsp;&emsp;&emsp;├── README.md&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Документация задания\
&emsp;&emsp;&emsp;&emsp;├── resources/&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Сохранённая модель\
&emsp;&emsp;&emsp;&emsp;│&emsp;&emsp;└── model.pkl\
&emsp;&emsp;&emsp;&emsp;└── reports/&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# Графики результатов\
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── class_balance.png\
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;└── confusion_matrix.png\

## 🖥️ Системные требования

- **Python**: 3.8 или выше
- **Оперативная память**: 4 ГБ (рекомендуется 8 ГБ)
- **Дисковое пространство**: 500 МБ
- **ОС**: Windows 10/11, macOS, Linux

## 📦 Установка зависимостей

### Общие зависимости для всех заданий:
pip install pandas numpy scikit-learn matplotlib seaborn joblib

### Или по отдельности для каждого задания:

**Задание №1:**
cd assignment1_preprocessing
pip install -r requirements.txt

**Задание №2:**
cd assignment2_regression
pip install -r requirements.txt

**Задание №3:**
cd assignment3_classification
pip install -r requirements.txt

## 🚀 Запуск проекта

### Подготовка данных

Поместите файл hh.csv в корневую папку проекта:

hh-preprocessing/\
&emsp;├── hh.csv&emsp;&emsp;&emsp;&emsp;← сюда\
&emsp;├── assignment1_preprocessing/\
&emsp;├── assignment2_regression/\
&emsp;└── assignment3_classification/\

---

### Задание №1: Парсинг и анализ данных

**Цель:** Преобразование сырых CSV-данных в числовые массивы .npy с использованием паттерна «Цепочка ответственности».

**Запуск:**
cd assignment1_preprocessing
python app.py

**Результат:**
- x_data.npy — матрица признаков (возраст, опыт, города)
- y_data.npy — вектор целевой переменной (зарплаты в рублях)

**Точка входа:** assignment1_preprocessing/app.py

---

### Задание №2: Линейная регрессия

**Цель:** Предсказание зарплат на основе обработанных данных из задания №1.

**Обучение модели:**
cd assignment2_regression
python train.py

**Предсказание зарплат:**
python app.py

**Результат:**
- Веса модели сохранены в resources/weights.npy и resources/bias.npy
- Вывод: список зарплат в рублях, по одной на строку

**Точка входа:** assignment2_regression/app.py

---

### Задание №3: Классификация уровней разработчиков

**Цель:** Автоматическое определение уровня разработчика (junior/middle/senior) с построением графиков и отчётов.

**Запуск:**
cd assignment3_classification
python train.py

**Результат:**
- Модель сохранена в resources/model.pkl
- График баланса классов: reports/class_balance.png
- Матрица ошибок: reports/confusion_matrix.png
- Отчёт о классификации выведен в консоль (precision, recall, F1-score)

**Точка входа:** assignment3_classification/train.py

---

## 🧪 Тестирование

Для проверки корректности работы всех заданий выполните:

# Проверка задания №1
cd assignment1_preprocessing
python app.py

# Проверка задания №2
cd ../assignment2_regression
python train.py
python app.py

# Проверка задания №3
cd ../assignment3_classification
python train.py

---

## 📊 Ожидаемые результаты

### Задание №1
- Успешная обработка ~67 000 резюме
- Создание файлов x_data.npy (66 945 × 12) и y_data.npy (66 945,)

### Задание №2
- Метрики качества после фильтрации выбросов:
  - R²: ~0.208 (20.8% объяснённой дисперсии)
  - RMSE: ~62 451 руб. (средняя ошибка предсказания)

### Задание №3
- Количество отфильтрованных IT-разработчиков: ~1 800
- Метрики качества классификации:
  - Accuracy: ~94.5%
  - Weighted F1-score: ~0.944

---

## 📚 Зависимости проекта

### Общие зависимости:
- pandas >= 1.5.0 — работа с табличными данными
- numpy >= 1.21.0 — числовые вычисления
- scikit-learn >= 1.0.0 — машинное обучение
- matplotlib >= 3.5.0 — построение графиков
- seaborn >= 0.11.0 — визуализация данных
- joblib >= 1.0.0 — сохранение моделей

### Специфичные зависимости:
- **Задание №1**: только pandas, numpy
- **Задание №2**: только numpy
- **Задание №3**: все перечисленные выше

---

## 📝 Документация по заданиям

### Задание №1: Парсинг и анализ данных
- **Паттерн проектирования**: Цепочка ответственности (Chain of Responsibility)
- **Обработчики**:
  - SalaryHandler — парсинг зарплаты
  - AgeHandler — извлечение возраста
  - ExperienceHandler — парсинг опыта работы
  - CityHandler — обработка города (one-hot encoding)
  - FinalHandler — формирование финальных массивов
- **Выходные данные**: x_data.npy, y_data.npy

### Задание №2: Линейная регрессия
- **Алгоритм**: Нормальное уравнение (без итераций)
- **Признаки**: возраст, опыт, города (one-hot)
- **Целевая переменная**: зарплата в рублях
- **Фильтрация**: удаление выбросов (< 15 000 руб., > 1 000 000 руб.)
- **Выходные данные**: веса в resources/, предсказания в консоли

### Задание №3: Классификация уровней
- **Алгоритм**: Random Forest с балансировкой классов
- **Фильтрация**: только настоящие разработчики (строгая фильтрация)
- **Разметка**: по ключевым словам в должности (без опыта)
- **Метрики**: precision, recall, F1-score по классам
- **Выходные данные**: модель в resources/, графики в reports/

---

## 🔧 Настройка окружения (опционально)

### Создание виртуального окружения:

**Windows:**
python -m venv venv
venv\Scripts\activate

**Linux/macOS:**
python3 -m venv venv
source venv/bin/activate

### Установка всех зависимостей сразу:
pip install pandas numpy scikit-learn matplotlib seaborn joblib