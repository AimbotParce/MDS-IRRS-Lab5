import os
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

log_folder = "data/logs"
log_files = [f for f in os.listdir(log_folder) if f.endswith(".h5") or f.endswith(".hdf5")]
raw_log_files = filter(lambda f: "teleport" in f, log_files)


sorted_log_files = sorted(raw_log_files, key=lambda f: float(os.path.splitext(f)[0].split("-")[-1]))

data = []

for log_file in sorted_log_files:
    damping = float(os.path.splitext(log_file)[0].split("-")[-1])

    with h5py.File(os.path.join(log_folder, log_file), "r") as log_fd:
        pagerank_steps = log_fd["pagerank_steps"]

        for i in range(min(5, pagerank_steps.shape[1])):  # Plot first 5 nodes
            data.append((damping, i, pagerank_steps[-1, i]))

import pandas as pd

df = pd.DataFrame(data, columns=["Damping", "Node", "PageRankValue"])

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Final PageRank Values for First 5 Nodes and Damping Factors")
ax.set_xlabel("Damping")
ax.set_ylabel("Final PageRank Value")

sns.lineplot(
    data=df,
    x="Damping",
    y="PageRankValue",
    hue="Node",
    style="Node",
    markers=True,
    dashes=False,
    ax=ax,
)

# Save the plot
fig_path = "images/final_pagerank_values_vs_damping.png"
fig.savefig(fig_path)
print(f"Saved plot to {fig_path}")

plt.show()
