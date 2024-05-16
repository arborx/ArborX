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

#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <Kokkos_Timer.hpp>

#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "generator.hpp"

using Point = ArborX::ExperimentalHyperGeometry::Point<3>;
using Triangle = ArborX::ExperimentalHyperGeometry::Triangle<3>;

template <typename MemorySpace>
struct Triangles
{
  Kokkos::View<Point *, MemorySpace> _points;
  Kokkos::View<KokkosArray<int, 3> *, MemorySpace> _triangle_vertices;
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

template <typename T>
std::string vec2string(std::vector<T> const &s, std::string const &delim = ", ")
{
  assert(s.size() > 1);

  std::ostringstream ss;
  std::copy(s.begin(), s.end(),
            std::ostream_iterator<std::string>{ss, delim.c_str()});
  auto delimited_items = ss.str().erase(ss.str().length() - delim.size());
  return "(" + delimited_items + ")";
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

  std::vector<std::string> allowed_geometries = {"ball", "plane"};

  bpo::options_description desc("Allowed options");
  int n;
  std::string vtk_filename;
  GeometryParams params;
  float angle;
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "angle", bpo::value<float>(&angle)->default_value(0), "Angle (degrees)" )
      ( "geometry", bpo::value<std::string>(&params.type)->default_value("ball"), ("geometry " + vec2string(allowed_geometries, " | ")).c_str() )
      ( "n", bpo::value<int>(&n)->default_value(-1), "number of query points" )
      ( "radius", bpo::value<float>(&params.radius)->default_value(1.f), "sphere radius" )
      ( "refinements", bpo::value<int>(&params.num_refinements)->default_value(5), "number of icosahedron refinements" )
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

  auto found = [](auto const &v, auto x) {
    return std::find(v.begin(), v.end(), x) != v.end();
  };
  if (!found(allowed_geometries, params.type))
  {
    std::cerr << "Geometry must be one of " << vec2string(allowed_geometries)
              << "\n";
    return 2;
  }

  ExecutionSpace space;

  Kokkos::Profiling::pushRegion("Benchmark::build_points");
  auto [vertices, triangles] = buildTriangles<MemorySpace>(space, params);
  Kokkos::Profiling::popRegion();

  if (n == -1)
    n = vertices.size();

  Kokkos::Profiling::pushRegion("Benchmark::build_points");
  Kokkos::View<Point *, MemorySpace> random_points(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "Benchmark::points"),
      n);
  ArborXBenchmark::generatePointCloud(
      space, ArborXBenchmark::PointCloudType::filled_box, std::cbrt(n),
      random_points);
  Kokkos::Profiling::popRegion();

  if (angle != 0)
  {
    rotateVertices(space, vertices, angle);
    rotateVertices(space, random_points, angle);
  }

  if (!vtk_filename.empty())
    writeVtk(vtk_filename, vertices, triangles);

  std::cout << "geometry          : " << params.type << '\n';
  std::cout << "#triangles        : " << triangles.size() << '\n';
  std::cout << "#queries          : " << random_points.size() << '\n';

  Kokkos::Timer timer;

  Kokkos::fence();
  ArborX::BVH<MemorySpace, Triangle> index(
      space, Triangles<MemorySpace>{vertices, triangles});
  Kokkos::fence();
  auto construction_time = timer.seconds();

  Kokkos::fence();
  timer.reset();
  Kokkos::View<int *, MemorySpace> offset("Benchmark::offsets", 0);
  Kokkos::View<float *, MemorySpace> distances("Benchmark::distances", 0);
  index.query(space, ArborX::Experimental::make_nearest(random_points, 1),
              DistanceCallback{}, distances, offset);
  Kokkos::fence();
  auto query_time = timer.seconds();

  printf("-- construction   : %5.3f\n", construction_time);
  printf("-- query          : %5.3f\n", query_time);
  printf("-- rate           : %5.3f\n", n / (1'000'000 * query_time));

  return 0;
}
