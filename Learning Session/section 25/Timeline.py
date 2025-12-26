import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import matplotlib.cm as cm

# =============================
#       Data definition
# =============================
tasks = [
    "Application Process",
    "Unitec Offer Letter",
    "Visa Process",
    "Landing",
    "First Part-time Jobs",
    "Second Part-time Jobs",
    "Third Part-time Jobs",
    "Graduation",
    "Post-study Work Visa",
    "Job Seeking",
    "Full-time Job",

]

start_dates = [
    "2023-02-07",
    "2023-04-02",
    "2023-05-25",
    "2023-07-13",
    "2023-07-24",
    "2024-02-07",
    "2024-05-16",
    "2025-02-01",
    "2025-04-08",
    "2025-04-17",
    "2025-06-23",

]

end_dates = [
    "2023-04-02",
    "2023-05-25",
    "2023-07-13",
    "2023-07-24",
    "2024-05-07",
    "2025-01-10",
    "2025-11-09",
    "2025-04-08",
    "2025-04-17",
    "2025-06-23",
    "2025-11-17" , # Full-time ongoing

]

# Convert dates to datetime objects
start = [datetime.strptime(d, "%Y-%m-%d") for d in start_dates]
end = [datetime.strptime(d, "%Y-%m-%d") for d in end_dates]

# Duration in days
durations = [(end[i] - start[i]).days for i in range(len(tasks))]

# =============================
#         Soft Mint Green + Liquid Glass
# =============================
MINT = "#69D2A7"
MINT_GLASS = "#69D2A766"   # Slight transparency
EDGE = "#3E8C6C"           # Soft darker green edge

# =============================
#         Plot setup
# =============================
fig, ax = plt.subplots(figsize=(12, 6))   # 横向更宽，更科研

y_positions = np.arange(len(tasks))

# =============================
#         Color palette for each bar
# =============================

# Use a colormap (e.g., tab20 or Set3) to get N distinct colors
cmap = cm.get_cmap('tab20', len(tasks))
bar_colors = [cmap(i) for i in range(len(tasks))]

# =============================
#         Draw Gantt bars (each with a different color)
# =============================
for i in range(len(tasks)):
    ax.barh(
        y_positions[i],
        durations[i],
        left=start[i],
        height=0.5,
        color=bar_colors[i],      # different color for each bar
        edgecolor=EDGE,
        linewidth=0.75
    )
    # 在每个条框的y轴位置画一条横向虚线，帮助对齐label和条框
    ax.axhline(
        y=y_positions[i],
        xmin=0, xmax=1,
        color=bar_colors[i],
        linestyle='--',
        linewidth=0.8,
        alpha=0.5,
        zorder=0
    )

# =============================
#         Formatting
# =============================
ax.set_yticks(y_positions)
ax.set_yticklabels(tasks, fontsize=9, fontfamily="Avenir")

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # 2 months spacing

plt.xticks(fontsize=8, fontfamily="Avenir")

ax.invert_yaxis()   # top-down style like research papers

# Lighter academic grid
plt.grid(axis='x', linestyle='--', color="#BBBBBB", alpha=0.4)

# Background: very light paper color
fig.patch.set_facecolor("#F7F9F7")
ax.set_facecolor("#F1F6F4")

# =============================
#              Title
# =============================
plt.title(
    "NZ Timeline: From Master's Application to Full-time Job",
    fontsize=14,
    fontweight="bold",
    fontfamily="Avenir",
    pad=12
)

plt.tight_layout()
plt.show()
plt.savefig('nz_timeline__chart.png')
