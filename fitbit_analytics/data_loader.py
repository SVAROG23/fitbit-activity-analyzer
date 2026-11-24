"""
Модуль загрузки и предобработки данных FitBit.
Читает 5 CSV, чистит, приводит типы и объединяет таблицы.
"""
import os
import warnings
import pandas as pd
import numpy as np

# На всякий случай гасим предупреждения о формате дат,
# если в каких-то версиях pandas они всё же появятся.
warnings.filterwarnings(
    "ignore",
    message="Could not infer format",
    category=UserWarning,
)


def load_csv(data_dir: str, filename: str) -> pd.DataFrame:
    """
    Читает CSV; если файла нет — понятная ошибка с подсказкой.
    """
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        available = os.listdir(data_dir) if os.path.isdir(data_dir) else []
        raise FileNotFoundError(
            f"Файл не найден: {path}\n"
            f"Скачайте датасет с Kaggle "
            f"(https://www.kaggle.com/datasets/arashnic/fitbit) "
            f"и положите CSV в '{data_dir}'.\n"
            f"Сейчас в каталоге: {available or 'каталог пуст или не существует'}"
        )
    return pd.read_csv(path)


def preprocess_daily_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает дневную активность: даты, пропуски, выбросы, дубли.
    """
    df = df.copy()

    # 1) Даты: errors="coerce" превращает мусор в NaT, дальше отбрасываем
    df["ActivityDate"] = pd.to_datetime(
        df["ActivityDate"], errors="coerce", format="mixed"
    )
    df = df.dropna(subset=["ActivityDate"])

    # 2) Числовые колонки: приводим к numeric, пропуски считаем нулём
    numeric_cols = ["TotalSteps", "TotalDistance", "Calories",
                    "VeryActiveMinutes", "FairlyActiveMinutes",
                    "LightlyActiveMinutes", "SedentaryMinutes"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 3) Физиологический фильтр: шаги > 60 000 в сутки — почти наверняка
    #    ошибка синхронизации трекера, такие записи искажают средние
    df = df[(df["TotalSteps"] >= 0) & (df["TotalSteps"] <= 60_000)]

    # 4) Дубликаты по ключу (пользователь, день) — оставляем первую запись
    df = df.drop_duplicates(subset=["Id", "ActivityDate"])

    # 5) Сортировка важна для rolling(7) и shift(-1) в других модулях
    return df.sort_values(["Id", "ActivityDate"]).reset_index(drop=True)


def preprocess_hourly(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Чистит почасовые данные, добавляет часы и день недели.
    Возвращает:
        pd.DataFrame — таблица с корректным временем и производными полями.
    """
    df = df.copy()
    # format="mixed" — в колонке встречаются разные форматы дат
    df[time_col] = pd.to_datetime(
        df[time_col], errors="coerce", format="mixed"
    )
    df = df.dropna(subset=[time_col])
    df["Hour"] = df[time_col].dt.hour
    df["Weekday"] = df[time_col].dt.day_name()
    return df


def preprocess_sleep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка данных о сне и агрегация до дневного уровня.
    Возвращает:
        pd.DataFrame с колонками [Id, ActivityDate, TotalMinutesAsleep, SleepHours].
        Имя ActivityDate выбрано специально, чтобы потом мержить с daily.
    """
    df = df.copy()
    if "date" not in df.columns:
        # Совсем пустая/несовместимая таблица — возвращаем каркас
        return pd.DataFrame(
            columns=["Id", "ActivityDate", "TotalMinutesAsleep", "SleepHours"]
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df = df.dropna(subset=["date"])

    # Безопасный выбор колонки-источника (без df.get со скаляром)
    if "value" in df.columns:
        src = df["value"]
    elif "TotalMinutesAsleep" in df.columns:
        src = df["TotalMinutesAsleep"]
    else:
        # Нет ни одной знакомой колонки — возвращаем пустой каркас
        return pd.DataFrame(
            columns=["Id", "ActivityDate", "TotalMinutesAsleep", "SleepHours"]
        )

    df["TotalMinutesAsleep"] = pd.to_numeric(src, errors="coerce")

    # Агрегация до одного значения на пользователя и день
    daily_sleep = (
        df.groupby(["Id", "date"], as_index=False)["TotalMinutesAsleep"]
          .sum()
          .rename(columns={"date": "ActivityDate"})  # единый ключ для merge
    )
    daily_sleep["SleepHours"] = daily_sleep["TotalMinutesAsleep"] / 60.0

    # Физиологический фильтр: > 16 ч сна — ошибка трекера или двойной учёт
    daily_sleep = daily_sleep[
        (daily_sleep["SleepHours"] >= 0) & (daily_sleep["SleepHours"] <= 16)
    ]
    return daily_sleep


def preprocess_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Чистит данные веса и ИМТ, убирает таймзону.
    Возвращает:
        pd.DataFrame с колонками [Id, ActivityDate, WeightKg, BMI].
        При отсутствии колонки Date — пустой каркас.
    """
    df = df.copy()
    if "Date" not in df.columns:
        return pd.DataFrame(columns=["Id", "ActivityDate", "WeightKg", "BMI"])

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce", utc=True, format="mixed"
    )
    df["Date"] = df["Date"].dt.tz_localize(None).dt.normalize()
    df = df.rename(columns={"Date": "ActivityDate"})

    for col in ["WeightKg", "BMI"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["ActivityDate"])


def build_dataset(data_dir: str) -> dict:
    """
    Собирает все данные в единый словарь, объединяя таблицы.
    Возвращает:
        dict с ключами:
            "daily"              — дневная активность,
            "hourly_steps"       — почасовые шаги,
            "hourly_intensities" — почасовые интенсивности,
            "sleep"              — дневной сон.
    """
    daily = preprocess_daily_activity(
        load_csv(data_dir, "dailyActivity_merged.csv"))
    hourly_steps = preprocess_hourly(
        load_csv(data_dir, "hourlySteps_merged.csv"), "ActivityHour")
    hourly_int = preprocess_hourly(
        load_csv(data_dir, "hourlyIntensities_merged.csv"), "ActivityHour")

    # Сон и вес — опциональные источники: если файла нет, не падаем,
    # а возвращаем пустые DataFrame с нужными колонками
    try:
        sleep = preprocess_sleep(load_csv(data_dir, "minuteSleep_merged.csv"))
    except FileNotFoundError:
        sleep = pd.DataFrame(columns=["Id", "ActivityDate", "SleepHours"])

    try:
        weight = preprocess_weight(
            load_csv(data_dir, "weightLogInfo_merged.csv"))
    except FileNotFoundError:
        weight = pd.DataFrame(
            columns=["Id", "ActivityDate", "WeightKg", "BMI"])

    # LEFT JOIN: сохраняем все дни активности, даже если нет данных о сне/весе
    merged = daily.merge(sleep, on=["Id", "ActivityDate"], how="left")
    merged = merged.merge(
        weight[["Id", "ActivityDate", "WeightKg", "BMI"]],
        on=["Id", "ActivityDate"], how="left")

    return {
        "daily": merged,
        "hourly_steps": hourly_steps,
        "hourly_intensities": hourly_int,
        "sleep": sleep,
    }
