"""
Модуль генерации отчёта.
Добавлен перевод минут в часы для единообразия с графиком.
"""
from metrics import (
    daily_summary, regularity_index, intensity_zone_minutes,
    active_days_streak, personal_goal, week_over_week,
)

def build_week_report(df_user) -> str:
    df = daily_summary(df_user)
    w = week_over_week(df)
    reg = regularity_index(df)
    zones = intensity_zone_minutes(df)
    streak = active_days_streak(df)
    goal = personal_goal(df)

    # Проверяем, есть ли данные о сне
    has_sleep = ("SleepHours" in df.columns) and df["SleepHours"].notna().any()

    def delta(cur, prev):
        if prev == 0: return "нет базы"
        d = (cur - prev) / prev * 100
        return f"{'+' if d >= 0 else ''}{d:.1f}%"

    sleep_line = (
        f"Средний сон: {w['sleep_current']:.1f} ч (изм. {delta(w['sleep_current'], w['sleep_previous'])})"
        if has_sleep else
        "Средний сон: нет данных"
    )

    lines = [
        "=" * 60,
        "ПЕРСОНАЛИЗИРОВАННЫЙ НЕДЕЛЬНЫЙ ОТЧЁТ",
        "=" * 60,
        f"Период: {df['ActivityDate'].min().date()} — {df['ActivityDate'].max().date()}",
        f"Дней с данными: {len(df)}",
        "",
        "--- Динамика ---",
        f"Средние шаги (тек. неделя): {w['steps_current']:.0f} (изм. {delta(w['steps_current'], w['steps_previous'])})",
        f"Средние калории: {w['calories_current']:.0f} (изм. {delta(w['calories_current'], w['calories_previous'])})",
        sleep_line,
        "",
        "--- Метрики ---",
        f"Индекс регулярности: {reg:.2f} (0–1)",
        f"Макс. серия активных дней: {streak}",
        f"Персональная цель по шагам: {goal:,}",
        "",
        "--- Время в зонах интенсивности ---",
        # Переводим минуты в часы
        f"Очень активные: {zones['very_active'] / 60:.1f} ч",
        f"Умеренные: {zones['fairly_active'] / 60:.1f} ч",
        f"Лёгкая активность: {zones['lightly_active'] / 60:.1f} ч",
        f"Сидячие: {zones['sedentary'] / 60:.1f} ч",
        "",
        "--- Рекомендации ---",
    ]

    # Правила
    if w["steps_current"] < w["steps_previous"] * 0.9:
        lines.append("• Активность снизилась — попробуйте добавить прогулку.")
    if streak >= 6:
        lines.append("• Серия активных дней ≥ 6 — запланируйте день отдыха.")
    if has_sleep and 0 < w["sleep_current"] < 7:
        lines.append("• Средний сон ниже 7 ч — сон влияет на активность.")
    if reg < 0.5:
        lines.append("• Низкая регулярность — график важнее интенсивности.")
    if zones["sedentary"] > 8 * 60:
        lines.append("• Много сидячих часов — делайте перерывы каждый час.")
    if not lines[-1].startswith("•"):
        lines.append("• Хороший баланс активности — продолжайте в том же духе.")

    lines.append("=" * 60)
    return "\n".join(lines)
