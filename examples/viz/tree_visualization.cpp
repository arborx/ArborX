/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_DetailsTreeVisualization.hpp>
#include <ArborX_LinearBVH.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <algorithm>
#include <fstream>
#include <random>

#include <point_clouds.hpp>

template <typename View>
void printPointCloud(View points, std::ostream &os)
{
  auto const n = points.extent_int(0);
  for (int i = 0; i < n; ++i)
    os << "\\node[leaf] at (" << points(i)[0] << "," << points(i)[1]
       << ") {\\textbullet};\n";
}

template <typename TreeType>
void viz(std::string const &prefix, std::string const &infile, int n_neighbors)
{
  using DeviceType = typename TreeType::device_type;
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::View<ArborX::Point *, DeviceType> points("points", 0);
  loadPointCloud(infile, points);

  TreeType bvh(points);

  using TreeVisualization =
      typename ArborX::Details::TreeVisualization<DeviceType>;
  using TikZVisitor = typename TreeVisualization::TikZVisitor;
  using GraphvizVisitor = typename TreeVisualization::GraphvizVisitor;

  int const n_queries = bvh.size();
  if (n_neighbors < 0)
    n_neighbors = bvh.size();
  Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType> queries("queries",
                                                                     n_queries);
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
                       KOKKOS_LAMBDA(int i) {
                         queries(i) = ArborX::nearest(points(i), n_neighbors);
                       });
  Kokkos::fence();

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
  auto permute =
      ArborX::Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(
          bvh.bounds(), queries);
  queries = ArborX::Details::BatchedQueries<DeviceType>::applyPermutation(
      permute, queries);
  performQueries(prefix + "sorted_", suffix);
}

// FIXME version of Kokkos in the CI/dev base image is too old and does not have
// scope guards so we define our own...
struct KokkosScopeGuard
{
  KokkosScopeGuard(Kokkos::InitArguments const &args)
  {
    Kokkos::initialize(args);
  }
  ~KokkosScopeGuard() { Kokkos::finalize(); }
};

int main(int argc, char *argv[])
{
  Kokkos::InitArguments args;
  args.disable_warnings = true;
  KokkosScopeGuard guard(args);

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

  if (vm.count("help"))
  {
    std::cout << desc << "\n";
    return 1;
  }

  using Tree = ArborX::BVH<Kokkos::Serial::device_type>;
  viz<Tree>(prefix, infile, n_neighbors);

  return 0;
}
