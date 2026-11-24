"""
Модуль поиска паттернов (на срезе одного пользователя).
"""
import numpy as np
import pandas as pd


def activity_heatmap(hourly_steps: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Матрица 7×24 (день недели × час) со средними шагами.
    """
    df = hourly_steps[hourly_steps["Id"] == user_id].copy()
    if df.empty:
        return pd.DataFrame()

    # dayofweek: 0 = понедельник, 6 = воскресенье
    df["DayOfWeek"] = df["ActivityHour"].dt.dayofweek
    pivot = df.pivot_table(
        index="DayOfWeek", columns="Hour", values="StepTotal", aggfunc="mean"
    )
    # reindex(range(7)) гарантирует все 7 строк, даже если какого-то дня нет
    return pivot.reindex(range(7)).fillna(0)


def detect_anomalous_days(df_user: pd.DataFrame, k: float = 2.0) -> pd.DataFrame:
    """
    Поиск аномальных дней через robust z-score (медиана и MAD).
    """
    df = df_user.copy()
    med = df["TotalSteps"].median()
    # MAD = median(|x - median|); 1.4826 — нормировка к std для нормальных данных
    mad = (df["TotalSteps"] - med).abs().median() or 1.0  # защита от нуля
    df["z_robust"] = (df["TotalSteps"] - med) / (1.4826 * mad)
    df["is_anomaly"] = df["z_robust"].abs() > k
    return df[df["is_anomaly"]][["ActivityDate", "TotalSteps", "z_robust"]]


def sleep_activity_correlation(df_user: pd.DataFrame) -> dict:
    """
    Корреляция сна и активности следующего дня (shift(-1).
    """
    df = df_user.sort_values("ActivityDate").copy()
    df["Steps_NextDay"] = df["TotalSteps"].shift(-1)
    sub = df.dropna(subset=["SleepHours", "Steps_NextDay"])
    if len(sub) < 3:
        return {"corr": np.nan, "n": len(sub)}
    corr = float(sub["SleepHours"].corr(sub["Steps_NextDay"]))
    return {"corr": corr, "n": len(sub)}
