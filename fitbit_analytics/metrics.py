"""
Модуль расчёта метрик для одного пользователя (df_user).
"""
import numpy as np
import pandas as pd


def daily_summary(df_user: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет 7-дневное скользящее среднее (MA7) для шагов и калорий.
    Возвращает:
        pd.DataFrame — копию с колонками Steps_MA7 и Calories_MA7.
    """
    df = df_user.sort_values("ActivityDate").copy()
    # min_periods=1 — считаем среднее даже в первые дни,
    # когда в окне меньше 7 точек (иначе первые 6 дней были бы NaN)
    df["Steps_MA7"] = df["TotalSteps"].rolling(7, min_periods=1).mean()
    df["Calories_MA7"] = df["Calories"].rolling(7, min_periods=1).mean()
    return df


def regularity_index(df_user: pd.DataFrame) -> float:
    """
    Индекс регулярности (0..1): 1 - CV. Чем ближе к 1, тем стабильнее.
    Математика:
        CV = std / mean — коэффициент вариации, нормированный
        показатель разброса. Регулярность = 1 - CV, ограниченная [0, 1].
    """
    steps = df_user["TotalSteps"].astype(float)
    mean = steps.mean()
    if mean == 0 or np.isnan(mean):
        return 0.0
    cv = steps.std(ddof=0) / mean
    return float(np.clip(1 - cv, 0, 1))


def intensity_zone_minutes(df_user: pd.DataFrame) -> dict:
    """
    Суммарные минуты по зонам интенсивности (устойчиво к пропускам).
    """
    cols = {
        "very_active": "VeryActiveMinutes",
        "fairly_active": "FairlyActiveMinutes",
        "lightly_active": "LightlyActiveMinutes",
        "sedentary": "SedentaryMinutes",
    }
    return {k: int(df_user[v].sum()) if v in df_user else 0
            for k, v in cols.items()}


def active_days_streak(df_user: pd.DataFrame, step_goal: int = 10_000) -> int:
    """
    Максимальная серия дней подряд с шагами >= step_goal
    """
    flags = (df_user["TotalSteps"] >= step_goal).astype(int).tolist()
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def personal_goal(df_user: pd.DataFrame, quantile: float = 0.75) -> int:
    """
    Персональная цель по шагам.
    """
    if df_user.empty:
        return 8_000
    q = df_user["TotalSteps"].quantile(quantile)
    return int(np.clip(q, 5_000, 20_000))


def week_over_week(df_user: pd.DataFrame) -> dict:
    """
    Сравнение последних 7 дней с предыдущими 7 днями.
    """
    df = df_user.sort_values("ActivityDate")
    if df.empty:
        return {}
    last_date = df["ActivityDate"].max()

    cur_week = df[df["ActivityDate"] > last_date - pd.Timedelta(days=7)]
    prev_week = df[(df["ActivityDate"] <= last_date - pd.Timedelta(days=7)) &
                   (df["ActivityDate"] > last_date - pd.Timedelta(days=14))]

    def _mean(week, col):
        """Безопасное среднее: 0.0, если нет данных или колонки."""
        return float(week[col].mean()) if not week.empty and col in week else 0.0

    return {
        "steps_current": _mean(cur_week, "TotalSteps"),
        "steps_previous": _mean(prev_week, "TotalSteps"),
        "calories_current": _mean(cur_week, "Calories"),
        "calories_previous": _mean(prev_week, "Calories"),
        "sleep_current": _mean(cur_week, "SleepHours"),
        "sleep_previous": _mean(prev_week, "SleepHours"),
    }
