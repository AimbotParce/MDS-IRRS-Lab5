import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf


def plot_pagerank_steps(log_file: str):
    with h5py.File(log_file, "r") as log_fd:
        pagerank_steps = log_fd["pagerank_steps"]
        norms = np.zeros(pagerank_steps.shape[0])
        for i in range(pagerank_steps.shape[0]):
            norms[i] = tf.reduce_sum(pagerank_steps[i, :])

        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=range(pagerank_steps.shape[0]), y=norms, marker="o", ax=ax)
        ax.set_ylim(0, 1.1)
        ax.set_title("PageRank Vector 1-Norm Over Iterations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("1-Norm of PageRank Vector")

        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(min(5, pagerank_steps.shape[1])):  # Plot first 5 nodes
            sns.lineplot(
                x=range(pagerank_steps.shape[0]),
                y=pagerank_steps[:, i],
                marker="o",
                label=f"Node {i}",
                ax=ax,
            )
        ax.set_title("PageRank Values Over Iterations for First 5 Nodes")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("PageRank Value")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot PageRank steps from log file.")
    parser.add_argument("log_file", type=str, help="Path to the PageRank log HDF5 file.")
    args = parser.parse_args()
    plot_pagerank_steps(args.log_file)
