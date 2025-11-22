from typing import Optional

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf


def plot_pagerank_steps(log_file: str, *, max_nodes: int = 50, out_file: Optional[str] = None):
    with h5py.File(log_file, "r") as log_fd:
        pagerank_steps = log_fd["pagerank_steps"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"PageRank Values of the first {max_nodes} Nodes")
        ax.set_xlabel("Node Index")
        ax.set_ylabel("PageRank Value")
        ax.set_ylim(0, np.max(pagerank_steps[:, :max_nodes]) * 1.1)
        sns.barplot(
            x=np.arange(max_nodes),
            y=pagerank_steps[0, :max_nodes],
            ax=ax,
        )

        def update(step: int):
            ax.clear()
            ax.set_title(f"PageRank Values of the first {max_nodes} Nodes")
            ax.set_xlabel("Node Index")
            ax.set_ylabel("PageRank Value")
            ax.set_ylim(0, np.max(pagerank_steps[:, :max_nodes]) * 1.1)
            barplot = sns.barplot(
                x=np.arange(max_nodes),
                y=pagerank_steps[step, :max_nodes],
                ax=ax,
            )
            return barplot

        ani = animation.FuncAnimation(
            fig=fig,
            func=update,
            frames=pagerank_steps.shape[0],
            repeat=False,
            interval=1,
        )

        if out_file:
            print("Saving animation to", out_file)
            ani.save(out_file, writer="pillow", dpi=300)

        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot PageRank steps from log file.")
    parser.add_argument("log_file", type=str, help="Path to the PageRank log HDF5 file.")
    parser.add_argument(
        "--max-nodes",
        type=int,
        help="Maximum number of nodes to display in the animation.",
        default=50,
    )
    parser.add_argument("--output", type=str, help="Output file to save the animation.", default=None)
    args = parser.parse_args()
    plot_pagerank_steps(args.log_file, max_nodes=args.max_nodes, out_file=args.output)
