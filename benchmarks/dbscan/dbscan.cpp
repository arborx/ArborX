/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "dbscan.hpp"

#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// FIXME: ideally, this function would be next to `loadData` in
// dbscan_timpl.hpp. However, that file is used for explicit instantiation,
// which would result in multiple duplicate symbols. So it is kept here.
int getDataDimension(std::string const &filename, bool binary)
{
  std::ifstream input;
  if (!binary)
    input.open(filename);
  else
    input.open(filename, std::ifstream::binary);
  if (!input.good())
    throw std::runtime_error("Error reading file \"" + filename + "\"");

  int num_points;
  int dim;
  if (!binary)
  {
    input >> num_points;
    input >> dim;
  }
  else
  {
    input.read(reinterpret_cast<char *>(&num_points), sizeof(int));
    input.read(reinterpret_cast<char *>(&dim), sizeof(int));
  }
  input.close();

  return dim;
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

  std::cout << "ArborX version    : " << ArborX::version() << std::endl;
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;
  std::cout << "Kokkos version    : " << KokkosExt::version() << std::endl;

  namespace bpo = boost::program_options;

  ArborXBenchmark::Parameters params;
  int dim;

  std::vector<std::string> allowed_algorithms = {"dbscan", "hdbscan", "mst"};
  std::vector<std::string> allowed_dendrograms = {"boruvka", "union-find"};
  std::vector<std::string> allowed_impls = {"fdbscan", "fdbscan-densebox"};

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "algorithm", bpo::value<std::string>(&params.algorithm)->default_value("dbscan"), ("algorithm " + vec2string(allowed_algorithms, " | ")).c_str() )
      ( "binary", bpo::bool_switch(&params.binary), "binary file indicator")
      ( "cluster-min-size", bpo::value<int>(&params.cluster_min_size)->default_value(1), "minimum cluster size")
      ( "core-min-size", bpo::value<int>(&params.core_min_size)->default_value(2), "DBSCAN min_pts")
      ( "dendrogram", bpo::value<std::string>(&params.dendrogram)->default_value("boruvka"), ("dendrogram " + vec2string(allowed_dendrograms, " | ")).c_str() )
      ( "dimension", bpo::value<int>(&dim)->default_value(3), "dimension of points to generate" )
      ( "eps", bpo::value<float>(&params.eps), "DBSCAN eps" )
      ( "filename", bpo::value<std::string>(&params.filename), "filename containing data" )
      ( "impl", bpo::value<std::string>(&params.implementation)->default_value("fdbscan"), ("implementation " + vec2string(allowed_impls, " | ")).c_str() )
      ( "labels", bpo::value<std::string>(&params.filename_labels)->default_value(""), "clutering results output" )
      ( "max-num-points", bpo::value<int>(&params.max_num_points)->default_value(-1), "max number of points to read in")
      ( "n", bpo::value<int>(&params.n)->default_value(10), "number of points to generate" )
      ( "samples", bpo::value<int>(&params.num_samples)->default_value(-1), "number of samples" )
      ( "variable-density", bpo::bool_switch(&params.variable_density), "type of cluster density to generate" )
      ( "verbose", bpo::bool_switch(&params.verbose), "verbose")
      ( "verify", bpo::bool_switch(&params.verify), "verify connected components")
      ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") > 0)
  {
    std::cout << desc << '\n';
    std::cout << "[Generator Help]\n"
                 "If using generator, the recommended DBSCAN parameters are:\n"
                 "- core-min-size = 10\n"
                 "- eps = 60 (2D constant), 100 (2D variable), 200 (3D "
                 "constant), 400 (3D variable)"
              << std::endl;
    return 1;
  }

  auto found = [](auto const &v, auto x) {
    return std::find(v.begin(), v.end(), x) != v.end();
  };

  if (!found(allowed_impls, params.implementation))
  {
    std::cerr << "Implementation must be one of " << vec2string(allowed_impls)
              << "\n";
    return 2;
  }
  if (!found(allowed_algorithms, params.algorithm))
  {
    std::cerr << "Algorithm must be one of " << vec2string(allowed_algorithms)
              << "\n";
    return 3;
  }
  if (!found(allowed_dendrograms, params.dendrogram))
  {
    std::cerr << "Dendrogram must be one of " << vec2string(allowed_dendrograms)
              << "\n";
    return 4;
  }

  std::stringstream ss;
  ss << params.implementation;

  // Print out the runtime parameters
  printf("algorithm         : %s\n", params.algorithm.c_str());
  if (params.algorithm == "dbscan")
  {
    printf("eps               : %f\n", params.eps);
    printf("cluster min size  : %d\n", params.cluster_min_size);
    printf("implementation    : %s\n", ss.str().c_str());
    printf("verify            : %s\n", (params.verify ? "true" : "false"));
  }
  if (params.algorithm == "hdbscan")
  {
    printf("dendrogram        : %s\n", params.dendrogram.c_str());
  }
  printf("minpts            : %d\n", params.core_min_size);
  if (!params.filename.empty())
  {
    // Data is read in
    printf("filename          : %s [%s, max_pts = %d]\n",
           params.filename.c_str(), (params.binary ? "binary" : "text"),
           params.max_num_points);
    printf("samples           : %d\n", params.num_samples);
  }
  else
  {
    // Data is generated
    printf("generator         : n = %d, dim = %d, density = %s\n", params.n,
           dim, (params.variable_density ? "variable" : "constant"));
  }
  if (!params.filename_labels.empty())
    printf("filename [labels] : %s [binary]\n", params.filename_labels.c_str());
  printf("verbose           : %s\n", (params.verbose ? "true" : "false"));

  if (!params.filename.empty())
    dim = getDataDimension(params.filename, params.binary);

  using ArborXBenchmark::run;

  bool success;
  switch (dim)
  {
  case 2:
    success = run<2>(params);
    break;
  case 3:
    success = run<3>(params);
    break;
  case 4:
    success = run<4>(params);
    break;
  case 5:
    success = run<5>(params);
    break;
  case 6:
    success = run<6>(params);
    break;
  default:
    std::cerr << "Error: dimension " << dim << " not allowed\n" << std::endl;
    success = false;
  }

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
