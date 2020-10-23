# Algorithm

This example considers the DBSCAN algorithm [1]. DBSCAN algorithm computes
clusters based on two parameters, `min_pts` (number of neighbors required to be
considered a core point) and `eps` (radius).

An example of an application requiring a solution to such problem is computing
halos in a cosmology simulation. Here, points correspond to points in the
universe, `eps` is called linking length, `min_pts = 1` and clusters are called
halos, or Friends-of-Friends (FoF).

A straightforward approach to compute halos, for example, would have one
compute the connectivity graph explicitly through ArborX spatial query search,
and then run a CC algorithm, such as ECL-CC ([2]), on that graph. The
implemented approach is an improvement over straightforward approach. Instead
of constructing the graph explicitly, it uses a combination of the callback
mechanism in ArborX with the `compute1` routine of the CC algorithm in [2] to
construct CC in a single tree query. A general DBSCAN algorithm is implemented
in a similar fashion, with a main distinction being that the number of
neighbors is pre-computed. Thus, it is expected that the `min_pts > 1`
algorithm is twice slower compared to `min_pts = 1` case.

[1] Ester, Kriegel, Sander, Xu. "A density-based algorithm for discovering
clusters in large spatial databases with noise". In Proceedings of the Second
International Conference on Knowledge Discovery and Data Mining, pp. 226-231.
1996.

[2] Jaiganesh, Burtscher. "A high-performance connected
components implementation for GPUs." In Proceedings of the 27th International
Symposium on High-Performance Parallel and Distributed Computing, pp. 92-104.
2018.

# Input

The example conrols its input through command-line options:
- `--filename`
  The data is expected to be provided as an argument to the `--filename`
  option. The data is in the format `[number of points, X-coordinates,
  Y-coordinates, Z-coordinates]`.
- `--binary`
  Indicator whether the data provided through `--filename` option is text or
  binary. If the data is binary, it is expected that number of points is an
  4-byte integer, and each coordinate is a 4-byte floating point number.
- `--core-min-size`
  `min_pts` parameter of the DBSCAN algorithm
- `--eps`
  `eps` parameter of the DBSCAN algorithm
- `--cluster-min-size`
  Further post-processing to filter out clusters under certain size
- `--verify`
  Internal check switch to verify clusters. This options is significantly more
  expected, as it explicitly computes the graph. This may also mean that it
  will run out of memory on GPU even if the DBSCAN algorithm itself does not.

# Output

The example produces clusters in CSR (compressed sparse storage) format
consisting of two arrays `(cluster_indices, cluster_offset)`, with indices for
a cluster `i` being entries
`cluster_indices(cluster_offset(i):cluster_offset(i+1)`. A simple
postprocessing step that calculates the sizes and centers of each cluster is
then performed.
