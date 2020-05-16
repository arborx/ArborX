# Algorithm

This example considers the following problem: find all connected components
(CCs) in a graph `G = (V, E)`, where `V` is a set of vertices corresponding to
geometric points `{P_i}`, and `E = {e_ij}` are edges, with `e_ij = 1` if
`dist(P_i, P_j) <= r`, and 0 otherwise. Here, `r` is a specified length.

An example of an application requiring a solution to such problem is computing
halos in an astrophysics simulation. Here, points correspond to points in the
universe, `r` is called linking length, and connected components are called
halos, or Friends-of-Friends (FoF).

A straightforward approach would have one compute `G` explicitly through ArborX
spatial query search, and then run a CC algorithm, such as ECL-CC ([1]), on
that graph. The implemented approach is an improvement over straightforward
approach. Instead of constructing the graph explicitly, it uses a combination
of the callback mechanism in ArborX with the `compute1` routine of the CC
algorithm in [1] to construct CC in a single tree query.

[1] Jaiganesh, Jayadharini, and Martin Burtscher. "A high-performance connected
components implementation for GPUs." In Proceedings of the 27th International
Symposium on High-Performance Parallel and Distributed Computing, pp. 92-104.
2018.

# Input

The example conrols its input through command-line options. The data is
expected to be provided as an argument to the `--filename` option. The data is
in the format `[number of points, X-coordinates, Y-coordinates,
Z-coordinates]`, and can be either text or binary (controlled by `--binary`
switch). If the data is binary, it is expected that number of points is an
4-byte integer, and each coordinate is a 4-byte floating point number.

# Output

The example produces halos in CSR (compressed sparse storage) format consisting
of two arrays `(halo_indices, halo_offset)`, with indices for a halo `i` being
entries `halo_indices(halo_offset(i):halo_offset(i+1)`. A simple postprocessing
step that calculates the sizes and centers of each halo is then performed.
