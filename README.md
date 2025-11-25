# Using PageRank to compute the importance of Airports

This repository contains a Python implementation of the PageRank algorithm applied to a dataset of airports and their connections. The goal is to compute the importance of each airport based on the network of flights between them.

## Dataset

The dataset used in this project is a pair of CSV files named `data/airports.csv` and `data/routes.csv`, which contain information about various airports and their connections. It was taken from [OpenFlights](https://openflights.org/data).

## Installation

To run the code, you need to have Python installed on your machine. You can install the required packages using uv:

```bash
uv sync
```

which you can get following their [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## Usage

To compute the PageRank of the airports, run the following command:

```bash
uv run python PageRank.py --airports data/airports.csv --routes data/routes.csv --output data/out/pagerank_results.csv
```

You can control the parameters of the PageRank algorithm using the following optional arguments:

- `--damping`: Damping factor (default: 0.85)
- `--max-iter`: Maximum number of iterations (default: 100)
- `--tolerance`: Tolerance for convergence (default: 1.0e-6)
- `--dead-end-strategy`: Strategy for handling dead-end nodes (default: None, options: 'always-teleport', 'always-stay')

And finally, you can specify a log file to record every step of the simulation for debugging and plotting purposes:
- `--log-file`: Path to the log file, which must be ended in .hdf5 (default: None)

## Output

You can visualize the results using the provided plotting scripts, for example, to visualize the evolution of the 1-norm
of the PageRank vector over iterations, as well as the evolution of the actual PageRank values for a few airports, run:

```bash
uv run PlotConvergence.py data/logs/your-log-file.hdf5 --show --output images/
```

