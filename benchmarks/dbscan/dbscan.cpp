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

#include "dbscan.hpp"

#include <ArborX_DBSCAN.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <cstdlib>
#include <fstream>

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
  ARBORX_ASSERT(input.good());

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

namespace ArborX::DBSCAN
{
// This function is required for Boost program_options to be able to use the
// Implementation enum.
std::istream &operator>>(std::istream &in, Implementation &implementation)
{
  std::string impl_string;
  in >> impl_string;

  if (impl_string == "fdbscan")
    implementation = ArborX::DBSCAN::Implementation::FDBSCAN;
  else if (impl_string == "fdbscan-densebox")
    implementation = ArborX::DBSCAN::Implementation::FDBSCAN_DenseBox;
  else
    in.setstate(std::ios_base::failbit);

  return in;
}

// This function is required for Boost program_options to use Implementation
// enum as the default_value().
std::ostream &operator<<(std::ostream &out,
                         Implementation const &implementation)
{
  switch (implementation)
  {
  case ArborX::DBSCAN::Implementation::FDBSCAN:
    out << "fdbscan";
    break;
  case ArborX::DBSCAN::Implementation::FDBSCAN_DenseBox:
    out << "fdbscan-densebox";
    break;
  }
  return out;
}
} // namespace ArborX::DBSCAN

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  std::cout << "ArborX version    : " << ArborX::version() << std::endl;
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;
  std::cout << "Kokkos version    : " << KokkosExt::version() << std::endl;

  namespace bpo = boost::program_options;
  using ArborX::DBSCAN::Implementation;

  ArborXBenchmark::Parameters params;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "algorithm", bpo::value<std::string>(&params.algorithm)->default_value("dbscan"), "algorithm (dbscan | mst)" )
      ( "filename", bpo::value<std::string>(&params.filename), "filename containing data" )
      ( "binary", bpo::bool_switch(&params.binary)->default_value(false), "binary file indicator")
      ( "max-num-points", bpo::value<int>(&params.max_num_points)->default_value(-1), "max number of points to read in")
      ( "eps", bpo::value<float>(&params.eps), "DBSCAN eps" )
      ( "cluster-min-size", bpo::value<int>(&params.cluster_min_size)->default_value(1), "minimum cluster size")
      ( "core-min-size", bpo::value<int>(&params.core_min_size)->default_value(2), "DBSCAN min_pts")
      ( "verify", bpo::bool_switch(&params.verify)->default_value(false), "verify connected components")
      ( "samples", bpo::value<int>(&params.num_samples)->default_value(-1), "number of samples" )
      ( "labels", bpo::value<std::string>(&params.filename_labels)->default_value(""), "clutering results output" )
      ( "print-dbscan-timers", bpo::bool_switch(&params.print_dbscan_timers)->default_value(false), "print dbscan timers")
      ( "impl", bpo::value<Implementation>(&params.implementation)->default_value(Implementation::FDBSCAN), R"(implementation ("fdbscan" or "fdbscan-densebox"))")
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
  printf("minpts            : %d\n", params.core_min_size);
  printf("filename          : %s [%s, max_pts = %d]\n", params.filename.c_str(),
         (params.binary ? "binary" : "text"), params.max_num_points);
  if (!params.filename_labels.empty())
    printf("filename [labels] : %s [binary]\n", params.filename_labels.c_str());
  printf("samples           : %d\n", params.num_samples);
  printf("print timers      : %s\n",
         (params.print_dbscan_timers ? "true" : "false"));

  int dim = getDataDimension(params.filename, params.binary);

  using ArborXBenchmark::run;

  bool success;
  switch (dim)
  {
  case 3:
    success = run<3>(params);
    break;
#if KOKKOS_VERSION >= 30700
  case 2:
    success = run<2>(params);
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
#endif
  default:
    std::cerr << "Error: dimension " << dim << " not allowed\n" << std::endl;
    success = false;
  }

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
