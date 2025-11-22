#!/usr/bin/python

import csv
import logging
import os
import time
from collections import defaultdict
from typing import List, Literal, Mapping, Optional, Sequence, cast

import h5py
import tensorflow as tf
from pydantic import BaseModel


class Airport(BaseModel):
    id: int
    name: str
    city: str
    country: str
    iata_code: Optional[str]
    icao_code: str
    latitude: float
    longitude: float
    altitude: int
    timezone: float  # Most are int, but for some reason some are fractional
    dst: Literal["E", "A", "S", "O", "Z", "N", "U"]  # Daylight savings time


# 2B,410,AER,2965,ASF,2966,,0,CR2
class Route(BaseModel):
    airline: Optional[str]
    airline_id: Optional[int]
    source_airport: Optional[str]
    source_airport_id: Optional[int]
    destination_airport: Optional[str]
    destination_airport_id: Optional[int]
    codeshare: Optional[Literal["Y"]]
    stops: Optional[int]
    equipment: Optional[str]


def read_airports(file: os.PathLike) -> List[Airport]:
    with open(file, "r", encoding="utf-8") as fd:
        airports: List[Airport] = []
        for row in csv.reader(fd, delimiter=",", quotechar='"'):
            airport = Airport(
                id=int(row[0]),
                name=row[1],
                city=row[2],
                country=row[3],
                iata_code=row[4],
                icao_code=row[5],
                latitude=float(row[6]),
                longitude=float(row[7]),
                altitude=int(row[8]),
                timezone=float(row[9]),
                dst=cast(Literal["E", "A", "S", "O", "Z", "N", "U"], row[10]),
            )
            airports.append(airport)
    return airports


def nullify(value: str):
    r"""Converts '\N' to None, otherwise returns the original value."""
    # This is needed because the dataset uses '\N' to represent null values.
    return None if value == r"\N" else value


def read_routes(file: os.PathLike) -> List[Route]:
    with open(file, "r", encoding="utf-8") as fd:
        routes: List[Route] = []
        for row in csv.reader(fd, delimiter=",", quotechar='"'):
            row = [nullify(value) for value in row]
            routes.append(
                Route(
                    airline=row[0],
                    airline_id=int(row[1]) if row[1] else None,
                    source_airport=row[2],
                    source_airport_id=int(row[3]) if row[3] else None,
                    destination_airport=row[4],
                    destination_airport_id=int(row[5]) if row[5] else None,
                    codeshare=cast(Literal["Y"], row[6]) if row[6] else None,
                    stops=int(row[7]) if row[7] else None,
                    equipment=row[8],
                )
            )
    return routes


def compute_adjacency_matrix(airports: List[Airport], routes: List[Route]) -> tf.sparse.SparseTensor:
    """
    Computes the adjacency matrix for the given airports and routes.
    Each entry (i, j) in the matrix represents the number of routes from airport i to airport j.
    Routes with source or destination airports not in the airports list are ignored.
    Args:
        airports (List[Airport]): List of Airport objects.
        routes (List[Route]): List of Route objects.
    Returns:
        tf.sparse.SparseTensor: Sparse adjacency matrix of shape (num_airports, num_airports).
    """

    airport_id_to_index = {airport.id: index for index, airport in enumerate(airports)}
    num_airports = len(airports)

    edge_counts = defaultdict(int)
    for route in routes:
        if route.source_airport_id is None or route.destination_airport_id is None:
            continue
        source_index = airport_id_to_index.get(route.source_airport_id)
        dest_index = airport_id_to_index.get(route.destination_airport_id)
        if source_index is not None and dest_index is not None:
            edge_counts[(source_index, dest_index)] += 1

    edge_indices = tf.constant(list(edge_counts.keys()), dtype=tf.int64)
    edge_weights = tf.constant(list(edge_counts.values()), dtype=tf.float32)

    adjacency_matrix = tf.sparse.SparseTensor(
        indices=edge_indices, values=edge_weights, dense_shape=[num_airports, num_airports]
    )
    return adjacency_matrix


def compute_pagerank(
    airports: List[Airport],
    routes: List[Route],
    *,
    damping: float = 0.85,
    tolerance: float = 1.0e-6,
    max_iter: int = 100,
    dead_end_strategy: Literal["always-teleport", None] = None,
    log_steps: Optional[str] = None,
) -> Sequence[float]:
    assert 0 < damping < 1, "Damping factor must be between 0 and 1."
    assert tolerance > 0, "Tolerance must be positive."
    assert max_iter > 0, "Maximum iterations must be positive."
    assert dead_end_strategy in (None, "always-teleport"), "Invalid dead-end strategy."
    assert log_steps == None or os.path.splitext(log_steps)[1] == ".hdf5", "Log steps file must be a HDF5 file."

    if log_steps:
        log_fd = h5py.File(log_steps, "w")

    # We'll attempt to simulate the power method.
    adjacency_matrix = compute_adjacency_matrix(airports, routes)
    num_nodes = int(adjacency_matrix.dense_shape[0])  # type: ignore

    # Transpose the adjacency matrix to get the correct direction for PageRank
    out_degrees = tf.sparse.reduce_sum(adjacency_matrix, axis=1, keepdims=True)  # Column vector of out-degrees
    transition_matrix = adjacency_matrix / out_degrees  # type: ignore

    if dead_end_strategy != None:
        dead_end_rows = tf.where(tf.equal(out_degrees, 0))[:, 0]

    transpose_transition_matrix = tf.sparse.transpose(transition_matrix)

    norm_warning_issued = False
    y = tf.ones(num_nodes, dtype=tf.float32) / tf.cast(num_nodes, tf.float32)
    y = tf.expand_dims(y, axis=1)  # Make it a column vector

    if log_steps:
        log_fd.create_dataset("adjacency_matrix_indices", data=adjacency_matrix.indices)
        log_fd.create_dataset("adjacency_matrix_values", data=adjacency_matrix.values)
        log_fd.create_dataset("adjacency_matrix_shape", data=adjacency_matrix.dense_shape)
        log_fd.create_dataset("out_degrees_index", data=out_degrees)
        log_fd.create_dataset("transpose_transition_matrix_indices", data=transpose_transition_matrix.indices)
        log_fd.create_dataset("transpose_transition_matrix_values", data=transpose_transition_matrix.values)
        log_fd.create_dataset("transpose_transition_matrix_shape", data=transpose_transition_matrix.dense_shape)
        log_pagerank = log_fd.create_dataset(
            "pagerank_steps", (0, num_nodes), maxshape=(None, num_nodes), dtype="float32"
        )
        log_pagerank.resize(1, axis=0)
        log_pagerank[0, :] = tf.squeeze(y, axis=1)

    for _ in range(max_iter):
        next_y = tf.sparse.sparse_dense_matmul(transpose_transition_matrix, y) * damping + (1 - damping) / num_nodes
        if dead_end_strategy == "always-teleport":
            # Sum the probability mass from all dead-end nodes at this iteration
            dead_ends_probability_mass = tf.reduce_sum(tf.gather(y, dead_end_rows))
            # Now add this mass uniformly to all nodes
            next_y += dead_ends_probability_mass * damping / num_nodes

        if tf.reduce_max(tf.abs(next_y - y)) < tolerance:
            break  # Break if the maximum change is below the tolerance
        y = next_y

        if log_steps:
            log_pagerank.resize(log_pagerank.shape[0] + 1, axis=0)
            log_pagerank[log_pagerank.shape[0] - 1, :] = tf.squeeze(y, axis=1)

        norm = tf.reduce_sum(y)  # Compute the 1-norm to check stability
        if not norm_warning_issued and tf.abs(norm - 1.0) > 1.0e-6:
            logger.warning(f"PageRank vector 1-norm is deviating from 1: {norm.numpy()}")
            norm_warning_issued = True
    return tf.squeeze(y, axis=1).numpy()


logger = logging.getLogger(__name__)


def main(
    airports_file: os.PathLike,
    routes_file: os.PathLike,
    damping: float,
    tolerance: float,
    max_iter: int,
    dead_end_strategy: Literal["always-teleport", None],
    log_steps: str | None,
    output: str | None,
) -> None:
    logger.info("Reading data from files...")
    airports = read_airports(airports_file)
    routes = read_routes(routes_file)
    logger.info(f"Number of airports read: {len(airports)}")
    logger.info(f"Number of routes read: {len(routes)}")

    # logger.info("Filtering airports with no IATA code...")
    # airports = [airport for airport in airports if airport.iata_code]
    # logger.info(f"Number of airports after filtering: {len(airports)}")
    # I won't filter out airports with no IATA code, because I changed the way we handle routes to use airport IDs.

    logger.info("Computing PageRank...")
    time_start = time.time()
    pagerank_values = compute_pagerank(
        airports,
        routes,
        damping=damping,
        tolerance=tolerance,
        max_iter=max_iter,
        dead_end_strategy=dead_end_strategy,
        log_steps=log_steps,
    )
    time_end = time.time()
    logger.info(f"Finished computing PageRank in {time_end - time_start} seconds.")
    logger.info(f"Sorting airports by PageRank...")
    airport_ranks = sorted(zip(airports, pagerank_values), key=lambda pair: pair[1], reverse=True)
    if output:
        logger.info(f"Writing PageRank results to {output}...")
        with open(output, "w", encoding="utf-8") as fd:
            writer = csv.writer(fd, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
            for airport, score in airport_ranks:
                writer.writerow([score, airport.name, airport.id, airport.iata_code])
    logger.info("Top 10 PageRank Results:")
    for rank, (airport, score) in enumerate(airport_ranks[:10], start=1):
        logger.info(f"{rank}: {score:.6f} - {airport.name} ({airport.iata_code})")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Compute PageRank for Airports based on Routes")
    parser.add_argument("--airports", type=str, default="airports.txt", help="Path to airports data file")
    parser.add_argument("--routes", type=str, default="routes.txt", help="Path to routes data file")
    parser.add_argument("--damping", type=float, default=0.85, help="Damping factor for PageRank")
    parser.add_argument("--tolerance", type=float, default=1.0e-6, help="Tolerance for convergence")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum number of iterations for PageRank")
    parser.add_argument("--dead-end-strategy", type=str, default=None, help="Strategy for handling dead-end nodes")
    parser.add_argument("--log-steps", type=str, default=None, help="Log steps to specified file")
    parser.add_argument("--output", type=str, default=None, help="Output file for PageRank results")
    args = parser.parse_args()
    main(
        args.airports,
        args.routes,
        args.damping,
        args.tolerance,
        args.max_iter,
        args.dead_end_strategy,
        args.log_steps,
        args.output,
    )
