"""
Модуль поиска паттернов.

Роль в архитектуре:
    Ищет скрытые закономерности в данных пользователя, которые
    не видны в простых дневных сводках:
        1) суточные паттерны (тепловая карта);
        2) аномальные дни (выбросы вверх/вниз);
        3) влияние сна на активность следующего дня.

    Все функции работают с «срезом» одного пользователя, чтобы
    паттерны были ЛИЧНЫМИ, а не популяционными.
"""
import numpy as np
import pandas as pd


def activity_heatmap(hourly_steps: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Матрица «день недели × час суток» со средними шагами.

    Зачем нужна:
        Визуальный «отпечаток» режима: например, провал в рабочие
        часы и пик вечером, или активные выходные. Используется
        для heatmap в visualization.plot_heatmap.

    Параметры:
        hourly_steps : pd.DataFrame — почасовые шаги (после preprocess_hourly).
        user_id      : int          — Id пользователя.

    Возвращает:
        pd.DataFrame 7×24 (index = dayofweek 0..6, columns = hours 0..23).
        Пустой DataFrame, если у пользователя нет почасовых данных.
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
    Поиск аномальных дней по шагам.

    Зачем нужна:
        Резкие выбросы (12 000 вместо обычных 4 000) или провалы
        (300 шагов) могут быть ошибкой трекера, болезнью или особым
        событием. Это полезный сигнал пользователю.

    Метод:
        Robust z-score: z = (x - median) / (1.4826 * MAD).
        Порог |z| > k (по умолчанию 2).
        Почему медиана/MAD, а не mean/std: обычные mean и std
        сильно зависят от самих выбросов — один «супер-день» может
        «замаскировать» аномалию. Медиана и MAD устойчивы.

    Параметры:
        df_user : pd.DataFrame — записи одного пользователя.
        k       : float        — порог (2.0 = умеренная строгость).

    Возвращает:
        pd.DataFrame c [ActivityDate, TotalSteps, z_robust]
        только по аномальным дням.
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
    Корреляция сна и активности СЛЕДУЮЩЕГО дня.

    Зачем нужна:
        Одна из прикладных задач из ИДЗ — «анализ образа жизни».
        Пользователю полезно знать: «когда вы спите больше,
        назавтра вы активнее?» (или наоборот).

    Логика:
        Сдвигаем шаги на день назад: Steps_NextDay = TotalSteps.shift(-1).
        Гипотеза: сон влияет на активность завтра, а не наоборот.

    Параметры:
        df_user : pd.DataFrame — записи одного пользователя.

    Возвращает:
        dict:
            corr — коэффициент Пирсона (nan, если пар < 3),
            n    — число валидных пар (сон, шаги следующего дня).
    """
    df = df_user.sort_values("ActivityDate").copy()
    df["Steps_NextDay"] = df["TotalSteps"].shift(-1)
    sub = df.dropna(subset=["SleepHours", "Steps_NextDay"])
    if len(sub) < 3:
        return {"corr": np.nan, "n": len(sub)}
    corr = float(sub["SleepHours"].corr(sub["Steps_NextDay"]))
    return {"corr": corr, "n": len(sub)}
