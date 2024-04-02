/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX.hpp>
#include <ArborX_HyperTriangle.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <string>

#include <point_clouds.hpp>

using Point = ArborX::ExperimentalHyperGeometry::Point<3>;
using Triangle = ArborX::ExperimentalHyperGeometry::Triangle<3>;

void icosahedron(std::vector<Point> &vertices, std::vector<int> &triangles)
{
  auto a = Kokkos::numbers::phi_v<float>;
  auto b = 1.f;

  vertices.clear();

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

  triangles.clear();

  triangles.insert(triangles.end(), {2, 1, 0});
  triangles.insert(triangles.end(), {1, 2, 3});
  triangles.insert(triangles.end(), {5, 4, 3});
  triangles.insert(triangles.end(), {4, 8, 3});
  triangles.insert(triangles.end(), {7, 6, 0});
  triangles.insert(triangles.end(), {6, 9, 0});
  triangles.insert(triangles.end(), {11, 10, 4});
  triangles.insert(triangles.end(), {10, 11, 6});
  triangles.insert(triangles.end(), {9, 5, 2});
  triangles.insert(triangles.end(), {5, 9, 11});
  triangles.insert(triangles.end(), {8, 7, 1});
  triangles.insert(triangles.end(), {7, 8, 10});
  triangles.insert(triangles.end(), {2, 5, 3});
  triangles.insert(triangles.end(), {8, 1, 3});
  triangles.insert(triangles.end(), {9, 2, 0});
  triangles.insert(triangles.end(), {1, 7, 0});
  triangles.insert(triangles.end(), {11, 9, 6});
  triangles.insert(triangles.end(), {7, 10, 6});
  triangles.insert(triangles.end(), {5, 11, 4});
  triangles.insert(triangles.end(), {10, 8, 4});
}

void subdivide(std::vector<Point> &vertices, std::vector<int> &triangles)
{
  std::map<std::pair<int, int>, int> hash;

  int const num_triangles = triangles.size() / 3;

  int vindex = vertices.size();
  std::vector<int> new_triangles;
  for (int i = 0; i < num_triangles; ++i)
  {
    int v[3] = {triangles[3 * i + 0], triangles[3 * i + 1],
                triangles[3 * i + 2]};
    int vmid[3];

    for (int j = 0; j < 3; ++j)
    {
      int vmin = std::min(v[j], v[(j + 1) % 3]);
      int vmax = std::max(v[j], v[(j + 1) % 3]);

      auto it = hash.find(std::make_pair(vmin, vmax));
      if (it != hash.end())
      {
        vmid[j] = it->second;
      }
      else
      {
        vmid[j] = vindex;
        hash[std::make_pair(vmin, vmax)] = vindex;
        ++vindex;

        vertices.push_back(Point{(vertices[vmin][0] + vertices[vmax][0]) / 2,
                                 (vertices[vmin][1] + vertices[vmax][1]) / 2,
                                 (vertices[vmin][2] + vertices[vmax][2]) / 2});
      }
    }

    new_triangles.insert(new_triangles.end(), {v[0], vmid[0], vmid[2]});
    new_triangles.insert(new_triangles.end(), {v[1], vmid[0], vmid[1]});
    new_triangles.insert(new_triangles.end(), {v[2], vmid[2], vmid[1]});
    new_triangles.insert(new_triangles.end(), {vmid[0], vmid[1], vmid[2]});
  }
  triangles = new_triangles;
}

void projectVerticesToSphere(std::vector<Point> &vertices, float radius)
{
  for (auto &v : vertices)
  {
    auto norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] *= radius / norm;
    v[1] *= radius / norm;
    v[2] *= radius / norm;
  }
}

void buildTriangles(float radius, int num_refinements,
                    std::vector<Point> &vertices, std::vector<int> &triangles)
{
  Kokkos::Profiling::ScopedRegion guard("Benchmark::build_triangles");

  icosahedron(vertices, triangles);
  for (int i = 1; i <= num_refinements; ++i)
    subdivide(vertices, triangles);

  projectVerticesToSphere(vertices, radius);
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

template <typename MemorySpace>
struct Triangles
{
  Kokkos::View<Point *, MemorySpace> _points;
  Kokkos::View<int *, MemorySpace> _triangle_vertices;
};

template <typename MemorySpace>
class ArborX::AccessTraits<Triangles<MemorySpace>, ArborX::PrimitivesTag>
{
  using Self = Triangles<MemorySpace>;

public:
  using memory_space = MemorySpace;

  static KOKKOS_FUNCTION auto size(Self const &self)
  {
    return self._triangle_vertices.size() / 3;
  }
  static KOKKOS_FUNCTION auto get(Self const &self, int i)
  {
    auto const &vertices = self._triangle_vertices;
    return Triangle{self._points(vertices(3 * i + 0)),
                    self._points(vertices(3 * i + 1)),
                    self._points(vertices(3 * i + 2))};
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
  int const num_elements = triangles.size() / 3;

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

  int const num_cell_vertices = 3;
  out << "\nPOLYGONS " << num_elements << " "
      << (num_elements * (1 + num_cell_vertices)) << '\n';
  for (int i = 0; i < num_elements; ++i)
  {
    out << num_cell_vertices;
    for (int j = 0; j < num_cell_vertices; ++j)
      out << " " << triangles(i * num_cell_vertices + j);
    out << '\n';
  }
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard;

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

  std::vector<Point> vertices_v;
  std::vector<int> triangles_v;
  buildTriangles(radius, num_refinements, vertices_v, triangles_v);

  auto vertices = vec2view<MemorySpace>(vertices_v);
  auto triangles = vec2view<MemorySpace>(triangles_v);

  if (!vtk_filename.empty())
    writeVtk(vtk_filename, vertices, triangles);

  ExecutionSpace space;

  Kokkos::Profiling::pushRegion("Benchmark::build_points");
  Kokkos::View<Point *, MemorySpace> random_points(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "Benchmark::points"),
      n);
  generatePointCloud(PointCloudType::filled_box, std::cbrt(n), random_points);
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
