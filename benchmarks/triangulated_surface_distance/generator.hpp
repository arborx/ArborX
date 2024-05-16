/****************************************************************************
 * Copyright (c) 2024 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_HyperTriangle.hpp>

#include <Kokkos_MathematicalConstants.hpp>

#if KOKKOS_VERSION >= 40400
#include <Kokkos_KokkosArray.hpp>
using KokkosArray = Kokkos::Kokkos;
#else
template <class T, size_t N>
struct KokkosArray
{
  static constexpr auto size() { return N; }
  KOKKOS_FUNCTION constexpr T &operator[](int i) { return _data[i]; }
  KOKKOS_FUNCTION constexpr T const &operator[](int i) const
  {
    return _data[i];
  }

  T _data[N];
};
#endif

struct GeometryParams
{
  std::string type;
  float radius;
  int num_refinements;
};

auto icosahedron()
{
  auto a = Kokkos::numbers::phi_v<float>;
  auto b = 1.f;

  using Point = ArborX::ExperimentalHyperGeometry::Point<3>;

  std::vector<Point> vertices;

  vertices.push_back(Point{0, b, -a});
  vertices.push_back(Point{b, a, 0});
  vertices.push_back(Point{-b, a, 0});
  vertices.push_back(Point{0, b, a});
  vertices.push_back(Point{0, -b, a});
  vertices.push_back(Point{-a, 0, b});
  vertices.push_back(Point{0, -b, -a});
  vertices.push_back(Point{a, 0, -b});
  vertices.push_back(Point{a, 0, b});
  vertices.push_back(Point{-a, 0, -b});
  vertices.push_back(Point{b, -a, 0});
  vertices.push_back(Point{-b, -a, 0});

  std::vector<KokkosArray<int, 3>> triangles;

  triangles.push_back({2, 1, 0});
  triangles.push_back({1, 2, 3});
  triangles.push_back({5, 4, 3});
  triangles.push_back({4, 8, 3});
  triangles.push_back({7, 6, 0});
  triangles.push_back({6, 9, 0});
  triangles.push_back({11, 10, 4});
  triangles.push_back({10, 11, 6});
  triangles.push_back({9, 5, 2});
  triangles.push_back({5, 9, 11});
  triangles.push_back({8, 7, 1});
  triangles.push_back({7, 8, 10});
  triangles.push_back({2, 5, 3});
  triangles.push_back({8, 1, 3});
  triangles.push_back({9, 2, 0});
  triangles.push_back({1, 7, 0});
  triangles.push_back({11, 9, 6});
  triangles.push_back({7, 10, 6});
  triangles.push_back({5, 11, 4});
  triangles.push_back({10, 8, 4});

  return std::make_tuple(vertices, triangles);
}

auto plane()
{
  using Point = ArborX::ExperimentalHyperGeometry::Point<3>;

  std::vector<Point> vertices;

  vertices.push_back(Point{0, 0, 0});
  vertices.push_back(Point{1, 0, 0});
  vertices.push_back(Point{2, 0, 0});
  vertices.push_back(Point{3, 0, 0});
  vertices.push_back(Point{0, 1, 0});
  vertices.push_back(Point{1, 1, 0});
  vertices.push_back(Point{2, 1, 0});
  vertices.push_back(Point{3, 1, 0});
  vertices.push_back(Point{0, 2, 0});
  vertices.push_back(Point{1, 2, 0});
  vertices.push_back(Point{2, 2, 0});
  vertices.push_back(Point{3, 2, 0});
  vertices.push_back(Point{0, 3, 0});
  vertices.push_back(Point{1, 3, 0});
  vertices.push_back(Point{2, 3, 0});
  vertices.push_back(Point{3, 3, 0});

  std::vector<KokkosArray<int, 3>> triangles;

  triangles.push_back({0, 1, 4});
  triangles.push_back({1, 5, 4});
  triangles.push_back({1, 2, 5});
  triangles.push_back({2, 6, 5});
  triangles.push_back({2, 3, 6});
  triangles.push_back({3, 7, 6});
  triangles.push_back({4, 5, 8});
  triangles.push_back({5, 9, 8});
  triangles.push_back({5, 6, 9});
  triangles.push_back({6, 10, 9});
  triangles.push_back({6, 7, 10});
  triangles.push_back({7, 11, 10});
  triangles.push_back({8, 9, 12});
  triangles.push_back({9, 13, 12});
  triangles.push_back({9, 10, 13});
  triangles.push_back({10, 14, 13});
  triangles.push_back({10, 11, 14});
  triangles.push_back({11, 15, 14});

  return std::make_tuple(vertices, triangles);
}

void convertTriangles2EdgeForm(std::vector<KokkosArray<int, 2>> &edges,
                               std::vector<KokkosArray<int, 3>> &triangles)
{
  std::map<std::pair<int, int>, int> hash;

  edges.clear();
  for (int i = 0, eindex = 0; i < (int)triangles.size(); ++i)
  {
    int e[3];
    for (int j = 0; j < 3; ++j)
    {
      auto minmax_pair =
          std::minmax(triangles[i][j], triangles[i][(j + 1) % 3]);
      auto it = hash.find(minmax_pair);
      if (it == hash.end())
      {
        edges.push_back({minmax_pair.first, minmax_pair.second});
        hash[minmax_pair] = eindex;
        e[j] = eindex++;
      }
      else
      {
        e[j] = it->second;
      }
    }
    triangles[i] = {e[0], e[1], e[2]};
  }
}

template <typename ExecutionSpace, typename MemorySpace>
void convertTriangles2VertexForm(
    ExecutionSpace const &space,
    Kokkos::View<KokkosArray<int, 2> *, MemorySpace> const &edges,
    Kokkos::View<KokkosArray<int, 3> *, MemorySpace> &triangles)
{
  int const num_triangles = triangles.size();
  Kokkos::parallel_for(
      "Benchmark::to_vertex_form",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_triangles),
      KOKKOS_LAMBDA(int i) {
        auto const &e0 = edges(triangles(i)[0]);
        auto const &e1 = edges(triangles(i)[1]);

        KOKKOS_ASSERT(e0[0] == e1[0] || e0[0] == e1[1] || e0[1] == e1[0] ||
                      e0[1] == e1[1]);
        if (e0[0] == e1[0] || e0[1] == e1[0])
          triangles(i) = {e0[0], e0[1], e1[1]};
        else
          triangles(i) = {e0[0], e0[1], e1[0]};
      });
}

/* Subdivide every triangle into four
         /\              /\
        /  \            /  \
       /    \   --->   /____\
      /      \        /\    /\
     /        \      /  \  /  \
    /__________\    /____\/____\
*/
template <typename ExecutionSpace, typename MemorySpace>
void subdivide(ExecutionSpace const &space,
               Kokkos::View<ArborX::ExperimentalHyperGeometry::Point<3> *,
                            MemorySpace> &vertices,
               Kokkos::View<KokkosArray<int, 2> *, MemorySpace> &edges,
               Kokkos::View<KokkosArray<int, 3> *, MemorySpace> &triangles)
{
  using Point = ArborX::ExperimentalHyperGeometry::Point<3>;

  int const num_vertices = vertices.size();
  int const num_edges = edges.size();
  int const num_triangles = triangles.size();

  Kokkos::resize(space, vertices, vertices.size() + edges.size());

  // Each edge is split in two, and each triangle adds three internal edges
  Kokkos::View<KokkosArray<int, 2> *, MemorySpace> new_edges(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "Benchmark::edges"),
      2 * num_edges + 3 * num_triangles);
  Kokkos::parallel_for(
      "Benchmark::split_edges",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_edges),
      KOKKOS_LAMBDA(int i) {
        int v = edges(i)[0];
        int w = edges(i)[1];

        int new_vindex = num_vertices + i;
        vertices(new_vindex) = Point{(vertices(v)[0] + vertices(w)[0]) / 2,
                                     (vertices(v)[1] + vertices(w)[1]) / 2,
                                     (vertices(v)[2] + vertices(w)[2]) / 2};
        new_edges(2 * i + 0) = {v, new_vindex};
        new_edges(2 * i + 1) = {w, new_vindex};
      });

  Kokkos::View<KokkosArray<int, 3> *, MemorySpace> new_triangles(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "Benchmark::triangles"),
      4 * num_triangles);
  Kokkos::parallel_for(
      "Benchmark::split_triangles",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_triangles),
      KOKKOS_LAMBDA(int i) {
        int e[3] = {triangles(i)[0], triangles(i)[1], triangles(i)[2]};

        int new_edges_offset = 2 * num_edges + 3 * i;
        for (int j = 0; j < 3; ++j)
        {
          // Find indices of the new edges that share a vertex
          int e0 = 2 * e[j];
          int e1 = 2 * e[(j + 1) % 3];
          if (new_edges(e0)[0] == new_edges(e1 + 1)[0])
            ++e1;
          else if (new_edges(e0 + 1)[0] == new_edges(e1)[0])
            ++e0;
          else if (new_edges(e0 + 1)[0] == new_edges(e1 + 1)[0])
          {
            ++e0;
            ++e1;
          }
          assert(new_edges(e0)[0] == new_edges(e1)[0]);
          int enew = new_edges_offset + j;

          new_edges(enew) = {new_edges(e0)[1], new_edges(e1)[1]};
          new_triangles(4 * i + j) = {e0, e1, enew};
        }
        new_triangles[4 * i + 3] = {new_edges_offset + 0, new_edges_offset + 1,
                                    new_edges_offset + 2};
      });
  edges = new_edges;
  triangles = new_triangles;
}

template <typename ExecutionSpace, typename MemorySpace>
void projectVerticesToSphere(
    ExecutionSpace const &space,
    Kokkos::View<ArborX::ExperimentalHyperGeometry::Point<3> *, MemorySpace>
        &points,
    float radius)
{
  Kokkos::parallel_for(
      "Benchmark::project_to_surface",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, points.size()),
      KOKKOS_LAMBDA(int i) {
        auto &v = points(i);
        auto norm = Kokkos::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        v[0] *= radius / norm;
        v[1] *= radius / norm;
        v[2] *= radius / norm;
      });
}

template <typename ExecutionSpace, typename MemorySpace>
void rotateVertices(ExecutionSpace const &space,
                    Kokkos::View<ArborX::ExperimentalHyperGeometry::Point<3> *,
                                 MemorySpace> &points,
                    float angle)
{
  // Rotate points around (1, -1, 0) axis given angle (in degrees)
  using Point = ArborX::ExperimentalHyperGeometry::Point<3>;
  using Vector = ArborX::Details::Vector<3>;

  Point o{0, 0, 0};
  Vector k{1 / std::sqrt(2.f), -1 / std::sqrt(2.f), 0}; // normalized axis
  auto cos = Kokkos::cos(Kokkos::numbers::pi_v<float> / 180 * angle);
  auto sin = Kokkos::sin(Kokkos::numbers::pi_v<float> / 180 * angle);
  Kokkos::parallel_for(
      "Benchmark::project_to_surface",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, points.size()),
      KOKKOS_LAMBDA(int i) {
        auto &p = points(i);
        Vector v = p - o;
        auto kxv = k.cross(v);
        auto kv = k.dot(v);

        // Rodrigues rotation formula
        for (int d = 0; d < 3; ++d)
          p[d] = v[d] * cos + kxv[d] * sin + k[d] * kv * (1 - cos);
      });
}

template <typename... P, typename T>
auto vec2view(std::vector<T> const &in, std::string const &label = "")
{
  Kokkos::View<T *, P...> out(
      Kokkos::view_alloc(label, Kokkos::WithoutInitializing), in.size());
  Kokkos::deep_copy(out, Kokkos::View<T const *, Kokkos::HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>{
                             in.data(), in.size()});
  return out;
}

template <typename MemorySpace, typename ExecutionSpace>
auto buildTriangles(ExecutionSpace const &space, GeometryParams const &params)
{
  Kokkos::Profiling::ScopedRegion guard("Benchmark::build_triangles");

  auto [vertices_v, triangles_v] =
      (params.type == "ball" ? icosahedron() : plane());

  // Convert to edge form
  std::vector<KokkosArray<int, 2>> edges_v;
  convertTriangles2EdgeForm(edges_v, triangles_v);

  auto vertices = vec2view<MemorySpace>(vertices_v);
  auto edges = vec2view<MemorySpace>(edges_v);
  auto triangles = vec2view<MemorySpace>(triangles_v);

  for (int i = 1; i <= params.num_refinements; ++i)
    subdivide(space, vertices, edges, triangles);

  if (params.type == "ball")
    projectVerticesToSphere(space, vertices, params.radius);

  convertTriangles2VertexForm(space, edges, triangles);

  return std::make_pair(vertices, triangles);
}
