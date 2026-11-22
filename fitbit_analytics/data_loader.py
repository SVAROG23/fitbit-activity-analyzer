"""
Модуль загрузки и предобработки данных FitBit Fitness Tracker Data.

Назначение:
    Превратить «сырые» CSV из датасета mturkfitbit_export_3.12.16-4.11.16
    в согласованные pandas.DataFrame с корректными типами, единым форматом
    дат и без явных аномалий.

Ключевые решения:
    * Все даты приводим к datetime через pd.to_datetime(..., format="mixed"),
      чтобы корректно обрабатывать два встречающихся формата
      (например, "4/12/2016 12:00:00 AM" и "2016-04-12 00:00:00")
      и не получать UserWarning в pandas 2.x/3.x.
    * Числовые колонки приводим к numeric и заполняем 0 там, где пропуск
      означает «активности не было» (шаги, минуты, калории).
    * Удаляем дубликаты по (Id, ActivityDate) — в датасете встречаются повторы.
    * Отсекаем физиологически невозможные значения (шаги > 60 000,
      сон > 16 ч) — это ошибки трекера, они искажают статистику.
    * Сон и вес — опциональные источники: при отсутствии файла/колонки
      возвращаем пустой DataFrame с нужными колонками, а не падаем.
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
    Читает CSV из каталога data_dir с понятной ошибкой, если файла нет.

    Параметры:
        data_dir : str — каталог с CSV-файлами датасета.
        filename : str — имя файла (например, "dailyActivity_merged.csv").

    Возвращает:
        pd.DataFrame — прочитанная таблица «как есть», без преобразований.
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
    Очистка основной таблицы дневной активности.

    Шаги:
        1) ActivityDate -> datetime (NaT отбрасываем).
        2) Числовые колонки -> numeric, пропуски -> 0.
        3) Фильтр выбросов по шагам: [0; 60 000].
        4) Удаление дубликатов по (Id, ActivityDate).
        5) Сортировка по (Id, ActivityDate) — нужна для rolling/shift.

    Параметры:
        df : pd.DataFrame — сырая dailyActivity_merged.csv.

    Возвращает:
        pd.DataFrame — очищенная дневная активность.
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
    Очистка почасовых таблиц (hourlySteps / hourlyIntensities).

    Добавляет две производные колонки:
        Hour    — час суток (0–23), нужен для pivot/heatmap.
        Weekday — название дня недели, удобно для подписей графиков.

    Параметры:
        df       : pd.DataFrame — сырая почасовая таблица.
        time_col : str          — имя колонки со временем (ActivityHour).

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

    Особенность датасета:
        minuteSleep_merged.csv в разных выгрузках содержит либо колонку
        'value' (код/минуты сна), либо 'TotalMinutesAsleep'. Здесь
        мы аккуратно выбираем источник по факту наличия колонки,
        не используя df.get(...) с fallback-скаляром, чтобы избежать
        несовместимости с pandas 3.x.

    Логика:
        * date -> datetime.
        * Суммируем минуты сна по (Id, date) — получаем дневной итог.
        * Переводим минуты в часы (SleepHours).
        * Отсекаем невозможные значения: сон ∈ [0; 16] ч.

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
    Очистка данных веса и ИМТ.

    Особенность:
        Дата в weightLogInfo_merged.csv приходит с таймзоной (UTC).
        Для мёржа с dailyActivity нужна «наивная» дата без времени,
        поэтому:
            * tz_localize(None) — убираем таймзону,
            * normalize() — оставляем только дату (00:00:00).

    Возвращает:
        pd.DataFrame с колонками [Id, ActivityDate, WeightKg, BMI, ...].
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
    Точка входа модуля: собирает все нужные таблицы и обогащает daily.

    Порядок:
        1) daily      — основная таблица.
        2) hourly_*   — почасовые шаги/интенсивности (для паттернов).
        3) sleep      — дневной сон (может отсутствовать — graceful).
        4) weight     — вес/ИМТ (может отсутствовать — graceful).
        5) merge daily + sleep + weight по (Id, ActivityDate).

    Возвращает:
        dict с ключами:
            "daily"              — обогащённая дневная активность,
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
