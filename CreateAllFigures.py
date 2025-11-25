import os
import subprocess
import sys

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot PageRank steps from log file.")
    parser.add_argument("logs_folder", type=str, help="Path to the PageRank logs folder with HDF5 files.")
    parser.add_argument("--output", type=str, default="images", help="Output folder for the plots.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for file_name in os.listdir(args.logs_folder):
        if file_name.endswith(".h5") or file_name.endswith(".hdf5"):
            log_file_path = os.path.join(args.logs_folder, file_name)
            subprocess.run([sys.executable, "PlotConvergence.py", log_file_path, "--output", args.output])
