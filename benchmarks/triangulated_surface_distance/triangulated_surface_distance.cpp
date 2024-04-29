/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX.hpp>
#include <ArborXBenchmark_PointClouds.hpp>
#include <ArborX_HyperTriangle.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Array.hpp>
#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <string>

using Point = ArborX::ExperimentalHyperGeometry::Point<3>;
using Triangle = ArborX::ExperimentalHyperGeometry::Triangle<3>;

auto icosahedron()
{
  auto a = Kokkos::numbers::phi_v<float>;
  auto b = 1.f;

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

  std::vector<Kokkos::Array<int, 3>> triangles;

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

void convertTriangles2EdgeForm(std::vector<Kokkos::Array<int, 3>> &triangles,
                               std::vector<Kokkos::Array<int, 2>> &edges)
{
  std::map<std::pair<int, int>, int> hash;

  int const num_triangles = triangles.size();
  edges.clear();
  for (int i = 0, eindex = 0; i < num_triangles; ++i)
  {
    int v[3] = {triangles[i][0], triangles[i][1], triangles[i][2]};
    int e[3];
    for (int j = 0; j < 3; ++j)
    {
      int vmin = std::min(v[j], v[(j + 1) % 3]);
      int vmax = std::max(v[j], v[(j + 1) % 3]);

      auto it = hash.find(std::make_pair(vmin, vmax));
      if (it == hash.end())
      {
        edges.push_back({vmin, vmax});
        e[j] = eindex;
        hash[std::make_pair(vmin, vmax)] = eindex;
        ++eindex;
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
    Kokkos::View<Kokkos::Array<int, 3> *, MemorySpace> &triangles,
    Kokkos::View<Kokkos::Array<int, 2> *, MemorySpace> const &edges)
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
               Kokkos::View<Point *, MemorySpace> &vertices,
               Kokkos::View<Kokkos::Array<int, 2> *, MemorySpace> &edges,
               Kokkos::View<Kokkos::Array<int, 3> *, MemorySpace> &triangles)
{
  int const num_vertices = vertices.size();
  int const num_edges = edges.size();
  int const num_triangles = triangles.size();

  Kokkos::resize(space, vertices, vertices.size() + edges.size());

  // Each edge is split in two, and each triangle adds three internal edges
  Kokkos::View<Kokkos::Array<int, 2> *, MemorySpace> new_edges(
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

  Kokkos::View<Kokkos::Array<int, 3> *, MemorySpace> new_triangles(
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
void projectVerticesToSphere(ExecutionSpace const &space,
                             Kokkos::View<Point *, MemorySpace> &points,
                             float radius)
{
  Kokkos::parallel_for(
      "Benchmark::project_to_surface",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, points.size()),
      KOKKOS_LAMBDA(int i) {
        auto &v = points(i);
        auto norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        v[0] *= radius / norm;
        v[1] *= radius / norm;
        v[2] *= radius / norm;
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
auto buildTriangles(ExecutionSpace const &space, float radius,
                    int num_refinements)
{
  Kokkos::Profiling::ScopedRegion guard("Benchmark::build_triangles");

  auto [vertices_v, triangles_v] = icosahedron();

  // Convert to edge form
  std::vector<Kokkos::Array<int, 2>> edges_v;
  convertTriangles2EdgeForm(triangles_v, edges_v);

  auto vertices = vec2view<MemorySpace>(vertices_v);
  auto edges = vec2view<MemorySpace>(edges_v);
  auto triangles = vec2view<MemorySpace>(triangles_v);

  for (int i = 1; i <= num_refinements; ++i)
    subdivide(space, vertices, edges, triangles);

  projectVerticesToSphere(space, vertices, radius);

  convertTriangles2VertexForm(space, triangles, edges);

  return std::make_pair(vertices, triangles);
}

template <typename MemorySpace>
struct Triangles
{
  Kokkos::View<Point *, MemorySpace> _points;
  Kokkos::View<Kokkos::Array<int, 3> *, MemorySpace> _triangle_vertices;
};

template <typename MemorySpace>
class ArborX::AccessTraits<Triangles<MemorySpace>, ArborX::PrimitivesTag>
{
  using Self = Triangles<MemorySpace>;

public:
  using memory_space = MemorySpace;

  static KOKKOS_FUNCTION auto size(Self const &self)
  {
    return self._triangle_vertices.size();
  }
  static KOKKOS_FUNCTION auto get(Self const &self, int i)
  {
    auto const &vertices = self._triangle_vertices;
    return Triangle{self._points(vertices(i)[0]), self._points(vertices(i)[1]),
                    self._points(vertices(i)[2])};
  }
};

struct DistanceCallback
{
  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate,
                                  Value const &value,
                                  OutputFunctor const &out) const
  {
    using ArborX::Details::distance;
    out(distance(ArborX::getGeometry(predicate), value));
  }
};

template <typename Points, typename Triangles>
void writeVtk(std::string const &filename, Points const &vertices,
              Triangles const &triangles)
{
  int const num_vertices = vertices.size();
  int const num_elements = triangles.size();

  constexpr int DIM =
      ArborX::GeometryTraits::dimension_v<typename Points::value_type>;

  auto vertices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, vertices);
  auto triangles_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, triangles);

  std::ofstream out(filename);

  out << "# vtk DataFile Version 2.0\n";
  out << "Mesh example\n";
  out << "ASCII\n";
  out << "DATASET POLYDATA\n\n";
  out << "POINTS " << num_vertices << " float\n";
  for (int i = 0; i < num_vertices; ++i)
  {
    for (int d = 0; d < DIM; ++d)
      out << " " << vertices(i)[d];
    out << '\n';
  }

  constexpr int num_cell_vertices = 3;
  out << "\nPOLYGONS " << num_elements << " "
      << (num_elements * (1 + num_cell_vertices)) << '\n';
  for (int i = 0; i < num_elements; ++i)
  {
    out << num_cell_vertices;
    for (int j = 0; j < num_cell_vertices; ++j)
      out << " " << triangles(i)[j];
    out << '\n';
  }
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  std::cout << "ArborX version    : " << ArborX::version() << '\n';
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << '\n';
  std::cout << "Kokkos version    : " << ArborX::Details::KokkosExt::version()
            << '\n';

  namespace bpo = boost::program_options;

  bpo::options_description desc("Allowed options");
  int n;
  int num_refinements;
  float radius;
  std::string vtk_filename;
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "n", bpo::value<int>(&n)->default_value(1000), "number of points" )
      ( "radius", bpo::value<float>(&radius)->default_value(1.f), "sphere radius" )
      ( "refinements", bpo::value<int>(&num_refinements)->default_value(5), "number of icosahedron refinements" )
      ( "vtk-filename", bpo::value<std::string>(&vtk_filename), "filename to dump mesh to in VTK format" )
      ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") > 0)
  {
    std::cout << desc << '\n';
    return 1;
  }

  ExecutionSpace space;

  auto [vertices, triangles] =
      buildTriangles<MemorySpace>(space, radius, num_refinements);

  if (!vtk_filename.empty())
    writeVtk(vtk_filename, vertices, triangles);

  Kokkos::Profiling::pushRegion("Benchmark::build_points");
  Kokkos::View<Point *, MemorySpace> random_points(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "Benchmark::points"),
      n);
  ArborXBenchmark::generatePointCloud(
      ArborXBenchmark::PointCloudType::filled_box, std::cbrt(n), random_points);
  Kokkos::Profiling::popRegion();

  std::cout << "#triangles        : " << triangles.size() << '\n';
  std::cout << "#queries          : " << random_points.size() << '\n';

  ArborX::BVH<MemorySpace, Triangle> index(
      space, Triangles<MemorySpace>{vertices, triangles});

  Kokkos::View<int *, MemorySpace> offset("Benchmark::offsets", 0);
  Kokkos::View<float *, MemorySpace> distances("Benchmark::distances", 0);
  index.query(space, ArborX::Experimental::make_nearest(random_points, 1),
              DistanceCallback{}, distances, offset);

  return 0;
}
