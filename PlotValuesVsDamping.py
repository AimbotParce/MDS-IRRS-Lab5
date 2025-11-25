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
            for iteration in range(pagerank_steps.shape[0]):
                data.append((damping, i, iteration, pagerank_steps[iteration, i]))

import pandas as pd

df = pd.DataFrame(data, columns=["Damping", "Node", "Iteration", "PageRankValue"])


sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("PageRank Values Over Iterations for First 5 Nodes and Damping Factors")
ax.set_xlabel("Iteration")
ax.set_ylabel("PageRank Value")

sns.lineplot(
    data=df,
    x="Iteration",
    y="PageRankValue",
    hue="Damping",
    style="Node",
    markers=True,
    dashes=False,
    ax=ax,
)

# Save the plot
fig_path = "images/pagerank_values_vs_damping_vs_iteration.png"
fig.savefig(fig_path)
print(f"Saved plot to {fig_path}")

plt.show()
