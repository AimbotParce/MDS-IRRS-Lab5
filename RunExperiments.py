import subprocess
import sys

lambdas = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
strategies = [None, "always-teleport", "always-stay"]
strategy_names = {None: "raw", "always-teleport": "teleport", "always-stay": "stay"}

airports = "data/airports.txt"
routes = "data/routes.txt"
tolerance = 1.0e-6
max_iter = 100
out_file = "data/out/{strategy}-{damping:.2f}.csv"
log_file = "data/logs/{strategy}-{damping:.2f}.hdf5"

for strategy in strategies:
    for damping in lambdas:
        args = [
            "--airports",
            airports,
            "--routes",
            routes,
            "--damping",
            str(damping),
            "--tolerance",
            str(tolerance),
            "--max-iter",
            str(max_iter),
            "--output",
            out_file.format(strategy=strategy_names[strategy], damping=damping),
            "--log-steps",
            log_file.format(strategy=strategy_names[strategy], damping=damping),
        ]
        if strategy is not None:
            args += ["--dead-end-strategy", strategy]

        subprocess.run([sys.executable, "PageRank.py"] + args)
