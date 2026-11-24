"""
Главный модуль (точка входа).
"""
import argparse
import os
import pandas as pd

from data_loader import build_dataset
from metrics import daily_summary
from patterns import (
    activity_heatmap, detect_anomalous_days, sleep_activity_correlation,
)
from report import build_week_report
from visualization import (
    plot_steps_dynamics, plot_zones, plot_heatmap,
)


def pick_user_interactive(daily: pd.DataFrame, user_id: int | None) -> int:
    """
    Выбор пользователя. Если user_id не задан, показываем список.
    """
    if user_id is not None:
        if user_id not in daily["Id"].values:
            raise ValueError(f"Пользователь с Id={user_id} не найден в базе.")
        return user_id

    # Сортируем по числу дней (самые полные профили сверху)
    user_stats = daily.groupby("Id").size().sort_values(ascending=False)

    print("\n--- Доступные пользователи ---")
    for i, (uid, count) in enumerate(user_stats.items(), 1):
        print(f"[{i:2}] Id: {uid} | Дней с данными: {count}")

    print("\nВыберите номер пользователя (или Enter для самого активного):")
    choice = input("> ").strip()

    if not choice:
        return int(user_stats.index[0])

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(user_stats):
            return int(user_stats.index[idx])
        print("Неверный номер. Выбран пользователь по умолчанию.")
        return int(user_stats.index[0])
    except ValueError:
        print("Введено не число. Выбран пользователь по умолчанию.")
        return int(user_stats.index[0])


def run(data_dir: str, user_id: int | None, out_base_dir: str) -> None:
    """
    Главный пайплайн.
    Создаем личную папку для каждого клиента (user_<id>),
    чтобы графики и отчеты не смешивались.
    """
    # 1. Загрузка данных
    print("[1/5] Загрузка и предобработка данных...")
    data = build_dataset(data_dir)
    daily = data["daily"]

    # 2. Выбор пользователя
    uid = pick_user_interactive(daily, user_id)
    df_user = daily[daily["Id"] == uid].copy()
    print(f"      Выбран Пользователь Id={uid}, дней: {len(df_user)}")

    # 3. Создаем личную папку для выгрузки (например, output/user_1503960366)
    #    Это решает проблему, когда файлы разных клиентов лежат в одной куче.
    out_dir = os.path.join(out_base_dir, f"user_{uid}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"      Все результаты будут сохранены в: {os.path.abspath(out_dir)}")

    # 4. Расчет метрик
    print("[2/5] Расчёт метрик и паттернов...")
    df_summary = daily_summary(df_user)
    anom = detect_anomalous_days(df_user)
    corr = sleep_activity_correlation(df_user)
    heat = activity_heatmap(data["hourly_steps"], uid)

    # 5. Текстовый отчет
    print("[3/5] Формирование отчёта...")
    report_text = build_week_report(df_user)
    
    # Сохраняем отчет в папку клиента
    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)  # дублируем в консоль

    if not anom.empty:
        print("\nАномальные дни:")
        print(anom.to_string(index=False))
    if corr["n"] >= 3:
        print(f"\nКорреляция сна и шагов следующего дня: r={corr['corr']:.2f}")

    # 6. Графики (онлайн + сохранение)
    print("\n[4/5] Построение графиков (закройте окно, чтобы продолжить)...")
    # Передаем пути внутрь папки клиента
    plot_steps_dynamics(df_summary, os.path.join(out_dir, "1_steps_dynamics.png"))
    plot_zones(df_user, os.path.join(out_dir, "2_zones.png"))
    plot_heatmap(heat, os.path.join(out_dir, "3_activity_heatmap.png"))

    # 7. Финал
    print(f"\n[5/5] Готово. Отчёт и графики для Id={uid} сохранены в:")
    print(f"      {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    # Вычисляем абсолютный путь к папке, где лежит сам файл main.py
    # Это гарантирует, что папка data будет найдена рядом с кодом,
    # даже если пользователь запускает скрипт из другой директории.
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(base_dir, "data")
    default_out_path = os.path.join(base_dir, "output")

    parser = argparse.ArgumentParser(description="Анализ фитнес-активности FitBit")
    parser.add_argument("--data", default=default_data_path, help="Каталог с CSV")
    parser.add_argument("--user", type=int, default=None, help="Id пользователя")
    parser.add_argument("--out", default=default_out_path, help="Каталог для выгрузки")
    
    args = parser.parse_args()
    run(args.data, args.user, args.out)
