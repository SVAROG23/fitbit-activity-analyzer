"""
Модуль визуализации.
"""
import os
import matplotlib.pyplot as plt


def plot_steps_dynamics(df_user, path: str):
    """График динамики шагов + 7-дневное скользящее среднее."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df_user["ActivityDate"], df_user["TotalSteps"], marker="o", label="Шаги")
    ax.plot(df_user["ActivityDate"], df_user["Steps_MA7"], linewidth=2, color="orange", label="MA7")
    ax.set_title("Динамика шагов")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Шаги")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    
    # Сохраняем файл ДО показа окна
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"      [Сохранено] {os.path.basename(path)}")
    
    plt.show() # Показываем окно
    plt.close(fig)


def plot_zones(df_user, path: str):
    """
    Столбчатая диаграмма суммарных ЧАСОВ по зонам интенсивности.
    Перевод в часы делает график адекватным (сидячие часы не «съедают» активные).
    """
    cols = ["VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes"]
    labels = ["Очень активные", "Умеренные", "Лёгкие", "Сидячие"]
    
    # Переводим минуты в часы
    values_hours = [df_user[c].sum() / 60.0 if c in df_user else 0.0 for c in cols]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values_hours, color=["#e74c3c", "#f39c12", "#2ecc71", "#95a5a6"])
    
    # Подписи значений над столбцами
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f} ч", ha="center", va="bottom")

    ax.set_title("Время в зонах интенсивности (часы)")
    ax.set_ylabel("Часы")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"      [Сохранено] {os.path.basename(path)}")
    
    plt.show()
    plt.close(fig)


def plot_heatmap(heatmap_df, path: str):
    """Тепловая карта «день недели × час суток»."""
    if heatmap_df.empty:
        return

    days = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(heatmap_df.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels([days[i] for i in heatmap_df.index])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_xlabel("Час суток")
    ax.set_title("Тепловая карта активности")
    fig.colorbar(im, ax=ax, label="Шаги")
    fig.tight_layout()
    
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"      [Сохранено] {os.path.basename(path)}")
    
    plt.show()
    plt.close(fig)
