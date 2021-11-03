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
  option. The data is in the format `[number of points, X-coordinates,
  Y-coordinates, Z-coordinates]`.
- `--impl`
  Switch between two algorithms described in [2]: `fdbscan` (FDBSCAN) and
  `fdbscan-densebox` (FDBSCAN-DenseBox).
- `--verify`
  Internal check switch to verify clusters. This options is significantly more
  expected, as it explicitly computes the graph. This may also mean that it
  will run out of memory on GPU even if the DBSCAN algorithm itself does not.

# Output

The example produces clusters in CSR (compressed sparse storage) format
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
