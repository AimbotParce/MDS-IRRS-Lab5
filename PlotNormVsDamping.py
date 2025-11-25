import os
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

log_folder = "data/logs"
log_files = [f for f in os.listdir(log_folder) if f.endswith(".h5") or f.endswith(".hdf5")]
raw_log_files = filter(lambda f: "raw" in f, log_files)


sorted_log_files = sorted(raw_log_files, key=lambda f: float(os.path.splitext(f)[0].split("-")[-1]))

norms = []
dampings = []
for log_file in sorted_log_files:
    damping = float(os.path.splitext(log_file)[0].split("-")[-1])
    dampings.append(damping)

    with h5py.File(os.path.join(log_folder, log_file), "r") as log_fd:
        pagerank_steps = log_fd["pagerank_steps"]
        norms.append(tf.reduce_sum(pagerank_steps[-1, :]).numpy())

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=dampings, y=norms, marker="o", ax=ax)
ax.set_ylim(0, 1.1)
ax.set_title("PageRank Vector 1-Norm Over Damping Factors")
ax.set_xlabel("Damping Factor")
ax.set_ylabel("1-Norm of PageRank Vector")

# Save the plot
fig_path = "images/pagerank_norms_vs_damping.png"
fig.savefig(fig_path)
print(f"Saved plot to {fig_path}")

plt.show()
