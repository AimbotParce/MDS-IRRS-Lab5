import os
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf


def plot_pagerank_steps(log_file: str, output_folder: Optional[str] = None, show: bool = False):
    with h5py.File(log_file, "r") as log_fd:
        pagerank_steps = log_fd["pagerank_steps"]
        norms = np.zeros(pagerank_steps.shape[0])
        for i in range(pagerank_steps.shape[0]):
            norms[i] = tf.reduce_sum(pagerank_steps[i, :])

        sns.set(style="whitegrid")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=range(pagerank_steps.shape[0]), y=norms, marker="o", ax=ax1)
        ax1.set_ylim(0, 1.1)
        ax1.set_title("PageRank Vector 1-Norm Over Iterations")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("1-Norm of PageRank Vector")

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for i in range(min(5, pagerank_steps.shape[1])):  # Plot first 5 nodes
            sns.lineplot(
                x=range(pagerank_steps.shape[0]),
                y=pagerank_steps[:, i],
                marker="o",
                label=f"Node {i}",
                ax=ax2,
            )
        ax2.set_title("PageRank Values Over Iterations for First 5 Nodes")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("PageRank Value")

    if output_folder:
        # Get the name from the log file for naming the plots
        file_name = os.path.splitext(os.path.basename(log_file))[0]
        fig1_path = os.path.join(output_folder, file_name + "_norms.png")
        fig2_path = os.path.join(output_folder, file_name + "_values.png")
        fig1.savefig(fig1_path)
        fig2.savefig(fig2_path)
        print(f"Saved plots to {fig1_path} and {fig2_path}")

    if show:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot PageRank steps from log file.")
    parser.add_argument("log_file", type=str, help="Path to the PageRank log HDF5 file.")
    parser.add_argument("--output", type=str, default="images", help="Output folder for the plots.")
    parser.add_argument("--show", action="store_true", help="Show the plots interactively.")
    args = parser.parse_args()
    plot_pagerank_steps(args.log_file, args.output, args.show)
