/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_LinearBVH.hpp>
#include <detail/ArborX_TreeVisualization.hpp>
#include <kokkos_ext/ArborX_KokkosExtAccessibilityTraits.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

// FIXME: this is a temporary place for loadPointCloud and writePointCloud
// Right now, only loadPointCloud is being used, and only in this example
template <typename Points>
void loadPointCloud(std::string const &filename, Points &random_points)
{
  static_assert(Kokkos::is_view_v<Points>);
  static_assert(Points::rank() == 1);
  static_assert(
      ArborX::GeometryTraits::dimension_v<typename Points::value_type> == 3);

  std::ifstream file(filename);
  if (file.is_open())
  {
    int size = -1;
    file >> size;
    ARBORX_ASSERT(size > 0);
    Kokkos::realloc(random_points, size);
    auto random_points_host = Kokkos::create_mirror_view(random_points);
    for (int i = 0; i < size; ++i)
      for (int j = 0; j < 3; ++j)
        file >> random_points(i)[j];
    Kokkos::deep_copy(random_points, random_points_host);
  }
  else
  {
    throw std::runtime_error("Cannot open file");
  }
}

template <typename Points>
void writePointCloud(Points random_points, std::string const &filename)
{
  static_assert(Kokkos::is_view_v<Points>);
  static_assert(Points::rank() == 1);
  static_assert(
      ArborX::GeometryTraits::dimension_v<typename Points::value_type> == 3);

  namespace KokkosExt = ArborX::Details::KokkosExt;
  static_assert(
      KokkosExt::is_accessible_from_host<decltype(random_points)>::value,
      "The View should be accessible on the Host");

  std::ofstream file(filename);
  if (file.is_open())
  {
    unsigned int const n = random_points.extent(0);
    for (unsigned int i = 0; i < n; ++i)
      file << random_points(i)[0] << " " << random_points(i)[1] << " "
           << random_points(i)[2] << "\n";
    file.close();
  }
}

template <typename View>
void printPointCloud(View points, std::ostream &os)
{
  auto const n = points.extent_int(0);
  for (int i = 0; i < n; ++i)
    os << "\\node[leaf] at (" << points(i)[0] << "," << points(i)[1]
       << ") {\\textbullet};\n";
}

void viz(std::string const &prefix, std::string const &infile, int n_neighbors)
{
  using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
  using DeviceType = ExecutionSpace::device_type;

  using Point = ArborX::Point<3>;
  using Box = ArborX::Box<3>;

  Kokkos::View<Point *, DeviceType> points("Example::points", 0);
  loadPointCloud(infile, points);

  Kokkos::View<Box *, DeviceType> boxes("Example::boxes", points.size());
  Kokkos::parallel_for(
      "Example::copy_points_to_boxes",
      Kokkos::RangePolicy<ExecutionSpace>(0, points.size()),
      KOKKOS_LAMBDA(int i) {
        boxes(i) = {points(i), points(i)};
      });

  ArborX::BoundingVolumeHierarchy bvh{
      ExecutionSpace{}, ArborX::Experimental::attach_indices(boxes)};

  using TreeVisualization = ArborX::Details::TreeVisualization;
  using TikZVisitor = typename TreeVisualization::TikZVisitor;
  using GraphvizVisitor = typename TreeVisualization::GraphvizVisitor;

  int const n_queries = bvh.size();
  if (n_neighbors < 0)
    n_neighbors = bvh.size();
  Kokkos::View<ArborX::Nearest<Point> *, DeviceType> queries("Example::queries",
                                                             n_queries);
  Kokkos::parallel_for(
      "Example::inititialize_queries",
      Kokkos::RangePolicy(ExecutionSpace{}, 0, n_queries),
      KOKKOS_LAMBDA(int i) {
        queries(i) = ArborX::nearest(points(i), n_neighbors);
      });

  auto performQueries = [&bvh, &queries](std::string const &p,
                                         std::string const &s) {
    std::ofstream fout;
    for (int i = 0; i < queries.extent_int(0); ++i)
    {
      std::string const fname = p + std::to_string(i) + s;
      fout.open(fname, std::fstream::out);
      TreeVisualization::visit(bvh, queries(i), GraphvizVisitor{fout});
      fout.close();
    }
  };

  std::fstream fout;

  // Print the point cloud
  fout.open(prefix + "points.tex", std::fstream::out);
  printPointCloud(points, fout);
  fout.close();

  // Print the bounding volume hierarchy
  fout.open(prefix + "bounding_volumes.tex", std::fstream::out);
  TreeVisualization::visitAllIterative(bvh, TikZVisitor{fout});
  fout.close();

  // Print the entire tree
  fout.open(prefix + "tree_all_nodes_and_edges.dot.m4", std::fstream::out);
  TreeVisualization::visitAllIterative(bvh, GraphvizVisitor{fout});
  fout.close();

  std::string const suffix = "_nearest_traversal.dot.m4";
  performQueries(prefix + "untouched_", suffix);

  // Shuffle the queries
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(queries.data(), queries.data() + queries.size(), g);
  performQueries(prefix + "shuffled_", suffix);

  // Sort them
  auto permute = ArborX::Details::computeSpaceFillingCurvePermutation(
      ExecutionSpace{},
      ArborX::Details::PredicateIndexables<decltype(queries)>{queries},
      ArborX::Experimental::Morton32{}, bvh.bounds());
  ArborX::Details::applyPermutation(ExecutionSpace{}, permute, queries);
  performQueries(prefix + "sorted_", suffix);
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  std::string prefix;
  std::string infile;
  int n_neighbors;
  boost::program_options::options_description desc("Allowed options");
  // clang-format off
    desc.add_options()
        ( "help", "produce help message" )
        ( "prefix", boost::program_options::value<std::string> (&prefix)->default_value("viz_"), "set prefix for output files" )
        ( "infile", boost::program_options::value<std::string> (&infile)->default_value("leaf_cloud.txt"), "set input point cloud file" )
        ( "neighbors", boost::program_options::value<int> (&n_neighbors)->default_value(5), "set the number of neighbors to search for (negative value means all)" )
    ;
  // clang-format on

  boost::program_options::variables_map vm;
  boost::program_options::store(
      boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  if (vm.count("help") > 0)
  {
    std::cout << desc << "\n";
    return 1;
  }

  viz(prefix, infile, n_neighbors);

  return 0;
}
