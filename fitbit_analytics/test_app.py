"""
Модуль тестирования.
"""
import os

import numpy as np
import pandas as pd

from data_loader import build_dataset
from metrics import personal_goal, regularity_index
from patterns import detect_anomalous_days, sleep_activity_correlation

# Вычисляем абсолютный путь к папке data (независимо от того, откуда запущен скрипт)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def test_regularity_index():
    """
    Тест 1: Проверка индекса регулярности активности.
    """
    # Сценарий 1: идеально ровные шаги -> регулярность 1.0
    df_flat = pd.DataFrame({"TotalSteps": [5000] * 10})
    assert np.isclose(regularity_index(df_flat), 1.0), (
        "Ровные шаги должны давать индекс 1.0"
    )

    # Сценарий 2: хаотичный разброс шагов -> низкая регулярность
    df_chaos = pd.DataFrame({"TotalSteps": [100, 10000, 100, 10000, 100]})
    assert regularity_index(df_chaos) < 0.5, (
        "Хаотичные шаги должны давать низкий индекс"
    )
    print("[OK] Тест регулярности пройден")


def test_personal_goal():
    """
    Тест 2: Проверка персональной цели по шагам.
    """
    # Сценарий 1: очень высокая активность -> цель упирается в потолок
    df_high = pd.DataFrame({"TotalSteps": [50000] * 10})
    assert personal_goal(df_high) == 20000, (
        "Верхний порог цели должен быть 20000"
    )

    # Сценарий 2: очень низкая активность -> цель упирается в пол
    df_low = pd.DataFrame({"TotalSteps": [100] * 10})
    assert personal_goal(df_low) == 5000, (
        "Нижний порог цели должен быть 5000"
    )
    print("[OK] Тест персональной цели пройден")


def test_anomalies():
    """
    Тест 3: Проверка поиска аномальных дней (выбросов).
    """
    data = {
        "ActivityDate": pd.date_range("2023-01-01", periods=10),
        # Явный выброс (15000) на фоне 5000 — алгоритм должен его найти
        "TotalSteps": [5000, 5000, 5000, 5000, 15000,
                       5000, 5000, 5000, 5000, 5000]
    }
    df = pd.DataFrame(data)
    anom = detect_anomalous_days(df)

    assert len(anom) == 1, "Должен быть найден ровно 1 аномальный день"
    assert anom["TotalSteps"].iloc[0] == 15000, (
        "Аномалией должен быть день с 15000 шагов"
    )
    print("[OK] Тест детекции аномалий пройден")


def test_sleep_correlation():
    """
    Тест 4: Проверка корреляции сна и активности следующего дня.
    """
    # Данные, где сон и шаги растут синхронно (корреляция = 1.0)
    data = {
        "ActivityDate": pd.date_range("2023-01-01", periods=5),
        "TotalSteps": [1000, 2000, 3000, 4000, 5000],
        "SleepHours": [1.0, 2.0, 3.0, 4.0, 5.0]
    }
    df = pd.DataFrame(data)
    res = sleep_activity_correlation(df)

    # Из-за сдвига на 1 день (shift) первые 4 пары коррелируют идеально
    assert np.isclose(res["corr"], 1.0), (
        "Корреляция должна быть прямой (1.0)"
    )
    print("[OK] Тест корреляции сна пройден")


def test_data_loading():
    """
    Тест 5 (Интеграционный): Проверка загрузки реальных CSV-файлов.
    """
    # Проверяем наличие папки по абсолютному пути
    if not os.path.exists(DATA_DIR):
        print("[INFO] Интеграционный тест: папка data не найдена. Пропускаем.")
        return

    # Проверяем, что пайплайн загрузки отрабатывает без исключений
    data = build_dataset(DATA_DIR)
    assert "daily" in data, "Словарь должен содержать ключ 'daily'"
    assert len(data["daily"]) > 0, "Данные не должны быть пустыми"
    assert "SleepHours" in data["daily"].columns, "Должна быть колонка сна"
    print("[OK] Интеграционный тест загрузки данных пройден")


if __name__ == "__main__":
    # Последовательный запуск всех тестов без pytest
    print("--- Запуск тестов ---")
    test_regularity_index()
    test_personal_goal()
    test_anomalies()
    test_sleep_correlation()
    test_data_loading()
    print("--- Все тесты успешно пройдены! ---")
