# Algorithm

This example considers the DBSCAN algorithm [1]. DBSCAN algorithm computes
clusters based on two parameters, `minPts` (number of neighbors required to be
considered a core point) and `eps` (radius).

An example of an application requiring a solution to such problem is computing
halos in a cosmology simulation. Here, points correspond to points in the
universe, `eps` is called linking length, `minPts = 2` and clusters are called
halos, or Friends-of-Friends (FoF).

A straightforward approach to compute halos, for example, would have one
compute the connectivity graph explicitly through ArborX spatial query search,
and then run a connected components algorithm on that graph. The implemented
approach is an improvement over straightforward approach. Instead of
constructing the graph explicitly, it uses a combination of the callback
mechanism in ArborX with a union-find routine to construct clusters in a single
tree query. A general DBSCAN algorithm is implemented in a similar fashion,
with a main distinction being that the number of neighbors is pre-computed.
Thus, it is expected that the `minPts > 2` algorithm is twice slower compared
to `minPts = 2` case. More details about the implemented algorithms can be
found in [2].

[1] Ester, Kriegel, Sander, Xu. "A density-based algorithm for discovering
clusters in large spatial databases with noise". In Proceedings of the Second
International Conference on Knowledge Discovery and Data Mining, pp. 226-231.
1996.

[2] Prokopenko, Lebrun-Grandie, Arndt. "Fast tree-based algorithms for DBSCAN
on GPUs." arXiv preprint arXiv:2103.05162.

# Input

The example controls its input through command-line options:
- `--binary`
  Indicator whether the data provided through `--filename` option is text or
  binary. If the data is binary, it is expected that number of points is an
  4-byte integer, and each coordinate is a 4-byte floating point number.
- `--core-min-size`
  `minPts` parameter of the DBSCAN algorithm
- `--eps`
  `eps` parameter of the DBSCAN algorithm
- `--cluster-min-size`
- `--filename`
  The data is expected to be provided as an argument to the `--filename`
  option.
- `--impl`
  Switch between two algorithms described in [2]: `fdbscan` (FDBSCAN) and
  `fdbscan-densebox` (FDBSCAN-DenseBox).
- `--verify`
  Internal check switch to verify clusters. This options is significantly more
  expensive, as it explicitly computes the graph. This may also mean that it
  may run out of memory on GPU even if the DBSCAN algorithm itself does not.

## Data file format

 For an `d`-dimensional data of size `n`, the structure of the file is `[n, d,
 p_{1,1}, ..., p_{1,d}, p_{2,1}, ..., p_{2,d}, ...]`, where `p_i = (p_{i,1},
 ..., p_{i,d})` is the `i`-th point in the dataset. In the binary format, all
 fields are 4 bytes, with size and dimension being `int`, and coordinates being
 `float`.

# Output

The example produces clusters in CSR (compressed sparse row) format
consisting of two arrays `(cluster_indices, cluster_offset)`, with indices for
a cluster `i` being entries
`cluster_indices(cluster_offset(i):cluster_offset(i+1)`.

# Running the example with the HACC data

The example is often used in ArborX to make sure no performance regressions
were introduced into the codebase. The experiments are usually run with the
data from a HACC cosmology simulation, obtained from a single run. A physically
meaningful value of `eps` is derived from the simulation parameters. The two
data files commonly used are:
- 37M problem

  `eps = 0.042` (computed as `0.168 * 256/1024`)
- 497M problem

  `eps = 0.014` (computed as `0.168 * 250/3072`)

For example, to run the smaller 37M HACC problem in Friends-of-Friends mode
(`minPts = 2`), use the following command:
```shell
./ArborX_Benchmark_DBSCAN.exe --eps 0.042 --binary --filename hacc_37M.arborx --core-min-size 2 --verbose
```
which would produce an output similar to this:
```text
ArborX version    : 1.3 (dev)
ArborX hash       : 170bae34
Kokkos version    : 3.7.0
algorithm         : dbscan
eps               : 0.042000
cluster min size  : 1
implementation    : fdbscan
verify            : false
minpts            : 2
filename          : hacc_37M.arborx [binary, max_pts = -1]
samples           : -1
print timers      : true
Reading in "hacc_37M.arborx" in binary mode...done
Read in 36902719 3D points
-- construction     :      0.025
-- query+cluster    :      0.186
-- postprocess      :      0.018
total time          :      0.229

#clusters       : 2763469
#cluster points : 23609948 [63.98%]
#noise   points : 13292771 [36.02%]
```
The last three lines (number of clusters and cluster points) can be used for
validation, as any difference in them would indicate an error.
