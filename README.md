# MimicGraph

Master's thesis algorithm based on RoarGraph. The project is split into two crates `mimicgraph-core` and `mimicgraph-cli`. The core crate implements the algorithm, which is then used by the CLI crate to build and evaluate datasets.

## Abstract 
> When the size and dimensionality of a dataset grow too large, we reach for Approximate Nearest Neighbors Search (ANNS) to answer queries that would otherwise be prohibitively slow to answer exactly. In recent years, some subproblems in the field have gained increased focus, such as Filtered ANNS, where we impose strict requirements on the search result, or out-of-distribution (OOD) queries that represent data of a different type to what is in the dataset. The current ANNS landscape is dominated by graph-based indices that do an expensive index construction operation upfront in order to save time when serving user queries. RoarGraph is such an algorithm; it builds a bipartite graph of the corpus and queries from the ground truth, projecting it into a final index. We take RoarGraph and drastically shorten its construction time by estimating the ground truth rather than computing it exactly. Then we more than make up the lost performance by adding more edges to the bipartite graph. The result is MimicGraph, an ANNS algorithm based on imitating another ANNS algorithm and scaling it up, while keeping build times short. This framing leads us to mimicking the filtered ANNS algorithm FilteredVamana, showing how MimicGraph can adapt to different workloads. Our results show generally better performance than RoarGraph and the state-of-the-art HNSW, all while building up to 10$\times$ faster than any of them. To ease its use, we present a method for fast parameter tuning based on dataset characteristics.

## Usage

In the mimicgraph directory, build with native optimizations

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

The binary has a handful of subcommands
```
$ ./target/release/mimicgraph
Usage: mimicgraph [OPTIONS] <COMMAND>

Commands:
  eval          Build index artifacts and evaluate recall
  build         Build index artifacts
  ground-truth  Compute ground truth for given queries
  inspect       Inspect artifact files written by mimicgraph-cli
  help          Print this message or the help of the given subcommand(s)
```

To build and evaluate against HNSW (and/or RoarGraph) use the `eval` command with a `--datafile`
```sh
./target/release/mimicgraph eval --datafile llama-128-ip.hdf5
```
The datafile must be an HDF5 file with datasets "points" and "query_points". If using filtered mode, there must additionally be two groups "labels and "query_labels" with datasets "data", "indices", and "indptr", representing CSR matrices.

