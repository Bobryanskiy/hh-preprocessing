import numpy as np

x = np.load('x_data.npy')
y = np.load('y_data.npy')

print(f"x_data shape: {x.shape}")
print(f"y_data shape: {y.shape}")
print(f"\nПервые 5 зарплат: {y[:5]}")
print(f"Средняя зарплата: {y.mean():.0f} руб.")
print(f"Мин/Макс: {y.min():.0f} / {y.max():.0f} руб.")