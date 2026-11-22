"""
Модуль расчёта метрик.

Роль в архитектуре:
    Здесь считаются все числовые показатели, которые потом идут
    в отчёт и на графики. Функции принимают «срез» одного пользователя
    (df_user), чтобы метрики были ПЕРСОНАЛЬНЫМИ, а не средними по популяции.

Почему так:
    Тема проекта — «персонализированный отчёт». Если считать средние
    по всем 30+ пользователям, пропадёт весь смысл рекомендаций.
"""
import numpy as np
import pandas as pd


def daily_summary(df_user: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет к дневным данным 7-дневное скользящее среднее (MA7).

    Зачем нужна:
        Сырые дневные шаги сильно «шумят» (погода, выходные, болезнь).
        Скользящее среднее за 7 дней сглаживает колебания и показывает
        реальный тренд. Именно MA7 используется в отчёте для сравнения
        «текущей недели» с «прошлой».

    Параметры:
        df_user : pd.DataFrame — записи одного пользователя.

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
    Индекс регулярности активности (0..1).

    Зачем нужна:
        Показать, насколько стабильно пользователь активен.
        Высокий индекс = ровный график, низкий = «качели» (то 12 000,
        то 300 шагов). Для здоровья важнее стабильность, чем рекорды.

    Математика:
        CV = std / mean — коэффициент вариации, нормированный
        показатель разброса. Регулярность = 1 - CV, ограниченная [0, 1].
        Почему не std напрямую: std зависит от масштаба
        (10 000 и 1 000 шагов несравнимы), а CV — безразмерный.

    Параметры:
        df_user : pd.DataFrame — записи одного пользователя.

    Возвращает:
        float в [0, 1]. При mean == 0 возвращает 0.0.
    """
    steps = df_user["TotalSteps"].astype(float)
    mean = steps.mean()
    if mean == 0 or np.isnan(mean):
        return 0.0
    cv = steps.std(ddof=0) / mean
    return float(np.clip(1 - cv, 0, 1))


def intensity_zone_minutes(df_user: pd.DataFrame) -> dict:
    """
    Суммарные минуты по зонам интенсивности.

    Зачем нужна:
        Показать баланс между «очень активными», «умеренными»,
        «лёгкими» и «сидячими» минутами. Если сидячих слишком много —
        в отчёт попадёт рекомендация делать перерывы.

    Параметры:
        df_user : pd.DataFrame — записи одного пользователя.

    Возвращает:
        dict с ключами very_active, fairly_active, lightly_active,
        sedentary. Отсутствующие колонки считаются нулём (устойчиво
        к неполным данным).
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
    Максимальная серия дней подряд с шагами >= step_goal.

    Зачем нужна:
        Профилактика перетренированности. Если серия ≥ 6 дней —
        в отчёт попадёт рекомендация запланировать день отдыха.
        Это одна из прикладных задач из ИДЗ (п. 3).

    Реализация:
        Простой линейный проход по флагам «день активный / нет».
        O(n) — эффективно даже на длинных историях.

    Параметры:
        df_user   : pd.DataFrame — записи одного пользователя.
        step_goal : int          — порог «активного дня» (по умолчанию 10 000).

    Возвращает:
        int — длина самой длинной серии подряд.
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

    Зачем нужна:
        Отказ от универсальных 10 000 шагов. Цель рассчитывается
        из ЛИЧНОЙ истории: 75-й перцентиль шагов = уровень, который
        пользователь уже достигал в ~25% дней. Это реалистично
        и мотивирует, а не пугает.

    Ограничения:
        clip в [5 000; 20 000] — защита от абсурдных целей
        (200 шагов у малоподвижного или 45 000 у марафонца).

    Параметры:
        df_user  : pd.DataFrame — записи одного пользователя.
        quantile : float        — квантиль (0.75 = «хороший» уровень).

    Возвращает:
        int — цель по шагам. 8 000 если истории нет.
    """
    if df_user.empty:
        return 8_000
    q = df_user["TotalSteps"].quantile(quantile)
    return int(np.clip(q, 5_000, 20_000))


def week_over_week(df_user: pd.DataFrame) -> dict:
    """
    Сравнение последних 7 дней с предыдущими 7 днями.

    Зачем нужна:
        Главный блок отчёта — «Динамика». Пользователю важно видеть
        не абсолютные числа, а изменение: «стало лучше / хуже / так же».
        Именно этот dict используется в report.build_week_report
        для расчёта процентов (+12.3%, -5.1%).

    Как определяется «неделя»:
        last_date = максимальная дата у пользователя.
        current   = (last_date - 7 дней; last_date]
        previous  = (last_date - 14 дней; last_date - 7 дней]
        Такой подход не привязан к календарным неделям — работает
        даже если данные начинаются в середине недели.

    Параметры:
        df_user : pd.DataFrame — записи одного пользователя.

    Возвращает:
        dict с текущими и предыдущими средними по шагам/калориям/сну.
        Пустые недели дают 0.0 — это сигнал «нет базы для сравнения».
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
