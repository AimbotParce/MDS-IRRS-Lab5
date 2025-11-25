import os
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

log_folder = "data/logs"
log_files = [f for f in os.listdir(log_folder) if f.endswith(".h5") or f.endswith(".hdf5")]
sorted_log_files = sorted(log_files, key=lambda f: float(os.path.splitext(f)[0].split("-")[-1]))

teleport_log_files = filter(lambda f: "teleport" in f, sorted_log_files)
stay_log_files = filter(lambda f: "stay" in f, sorted_log_files)

data = []
for strategy_log_files, label in [(stay_log_files, "always-stay"), (teleport_log_files, "always-teleport")]:
    for log_file in strategy_log_files:
        damping = float(os.path.splitext(log_file)[0].split("-")[-1])

        with h5py.File(os.path.join(log_folder, log_file), "r") as log_fd:
            out_degrees = log_fd["out_degrees_index"][:]
            dead_end_rows = tf.where(tf.equal(out_degrees, 0))[:, 0]

            pagerank_steps = log_fd["pagerank_steps"]

            final_values = pagerank_steps[-1, :]
            average_dead_end_value = tf.reduce_mean(tf.gather(final_values, dead_end_rows)).numpy()
            data.append((label, damping, average_dead_end_value))

import pandas as pd

df = pd.DataFrame(data, columns=["Strategy", "Damping", "AvgDeadEndPageRankValue"])


sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Average PageRank Values of Dead Ends for Different Strategies and Damping Factors")
ax.set_xlabel("Damping Factor")
ax.set_ylabel("Average PageRank Value of Dead Ends")

sns.lineplot(
    data=df,
    x="Damping",
    y="AvgDeadEndPageRankValue",
    hue="Strategy",
    style="Strategy",
    markers=True,
    dashes=False,
    ax=ax,
)

# Save the plot
fig_path = "images/avg_dead_end_pagerank_values_vs_damping.png"
fig.savefig(fig_path)
print(f"Saved plot to {fig_path}")

plt.show()
