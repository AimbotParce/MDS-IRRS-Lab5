#!/usr/bin/python

import csv
import logging
import os
import time
from collections import defaultdict
from typing import List, Literal, Mapping, Optional, Sequence, cast

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
) -> Sequence[float]:
    assert 0 < damping < 1, "Damping factor must be between 0 and 1."
    assert tolerance > 0, "Tolerance must be positive."
    assert max_iter > 0, "Maximum iterations must be positive."

    # We'll attempt to simulate the power method.
    adjacency_matrix = compute_adjacency_matrix(airports, routes)
    num_nodes = int(adjacency_matrix.dense_shape[0])  # type: ignore

    # Transpose the adjacency matrix to get the correct direction for PageRank
    transition_matrix = adjacency_matrix / tf.sparse.reduce_sum(adjacency_matrix, axis=1, keepdims=True)  # type: ignore
    transition_matrix = tf.sparse.transpose(transition_matrix)

    def step(y_prev, adj):
        y_prev = tf.expand_dims(y_prev, axis=1)
        y_next = tf.sparse.sparse_dense_matmul(adj, y_prev) * damping + (1 - damping) / num_nodes
        y_next = tf.squeeze(y_next, axis=1)
        # y_next = y_next / tf.reduce_sum(y_next) # It should already sum to 1
        return y_next

    y = tf.ones(num_nodes, dtype=tf.float32) / tf.cast(num_nodes, tf.float32)
    for _ in range(max_iter):
        next_y = step(y, transition_matrix)
        if tf.reduce_max(tf.abs(next_y - y)) < tolerance:
            # Break if the maximum change is below the tolerance
            break
        y = next_y
    return y.numpy()


logger = logging.getLogger(__name__)


def main(
    airports_file: os.PathLike,
    routes_file: os.PathLike,
    damping: float,
    tolerance: float,
    max_iter: int,
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
    pagerank_values = compute_pagerank(airports, routes, damping=damping, tolerance=tolerance, max_iter=max_iter)
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
    parser.add_argument("--output", type=str, default=None, help="Output file for PageRank results")
    args = parser.parse_args()
    main(args.airports, args.routes, args.damping, args.tolerance, args.max_iter, args.output)
