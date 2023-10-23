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
#include <ArborX_InterpMovingLeastSquares.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <random>

using Point = ArborX::ExperimentalHyperGeometry::Point<2, double>;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = typename ExecutionSpace::memory_space;

// Randomly fills a 2 * half_edge box centered at 0
void filledBoxRandom(double half_edge,
                     Kokkos::View<Point *, MemorySpace> points)
{
  constexpr auto DIM = ArborX::GeometryTraits::dimension<Point>::value;
  int n = points.extent(0);
  auto points_host = Kokkos::create_mirror_view(points);

  std::uniform_real_distribution<double> dist(-half_edge, half_edge);
  std::random_device rd;
  std::default_random_engine gen(rd());
  auto random = [&dist, &gen]() { return dist(gen); };
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < DIM; ++d)
      points_host(i)[d] = random();

  Kokkos::deep_copy(points, points_host);
}

// Dumps the data in a file in a csv format
void dump_msl_data(std::string const &dump_filename,
                   Kokkos::View<Point *, MemorySpace> source_points,
                   Kokkos::View<Point *, MemorySpace> target_points,
                   Kokkos::View<double *, MemorySpace> source_values,
                   Kokkos::View<double *, MemorySpace> target_values,
                   Kokkos::View<double *, MemorySpace> approx_values)
{
  std::fstream dump_file_stream(dump_filename, std::ios::out | std::ios::trunc);
  if (!dump_file_stream)
    throw std::runtime_error("Unable to open/create file " + dump_filename);

  int const dim = ArborX::GeometryTraits::dimension_v<Point>;
  for (int i = 0; i < dim; i++)
    dump_file_stream << "source coord " << i << ';';
  dump_file_stream << "source value;";
  for (int i = 0; i < dim; i++)
    dump_file_stream << "target coord " << i << ';';
  dump_file_stream << "target value;approx value\n";

  auto source_points_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, source_points);
  auto target_points_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, target_points);
  auto source_values_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, source_values);
  auto target_values_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, target_values);
  auto approx_values_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, approx_values);

  int const num_points = source_points.extent(0);
  for (int i = 0; i < num_points; i++)
  {
    for (int j = 0; j < dim; j++)
      dump_file_stream << source_points_host(i)[j] << ';';
    dump_file_stream << source_values_host(i) << ';';
    for (int j = 0; j < dim; j++)
      dump_file_stream << target_points_host(i)[j] << ';';
    dump_file_stream << target_values_host(i) << ';' << approx_values_host(i)
                     << '\n';
  }

  dump_file_stream.close();
}

// Funtion to approximate
KOKKOS_INLINE_FUNCTION double functionToApproximate(Point const &p)
{
  auto const x = p[0];
  return (x > 0) ? 1 : 0;
}

void mls_example(std::size_t num_points, std::string const &dump_file)
{
  ExecutionSpace space{};

  // Generation of random points uniformely distributed in a [-1/2; 1/2] square.
  Kokkos::View<Point *, MemorySpace> source_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::source_points"),
      num_points);
  Kokkos::View<double *, MemorySpace> source_values(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::source_values"),
      num_points);
  Kokkos::View<Point *, MemorySpace> target_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::target_points"),
      num_points);
  Kokkos::View<double *, MemorySpace> target_values(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::target_values"),
      num_points);
  filledBoxRandom(0.5, source_points);
  filledBoxRandom(0.5, target_points);
  Kokkos::parallel_for(
      "Example::fill_views",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_points),
      KOKKOS_LAMBDA(int const i) {
        source_values(i) = functionToApproximate(source_points(i));
        target_values(i) = functionToApproximate(target_points(i));
      });

  ArborX::Interpolation::MovingLeastSquares<MemorySpace, double> mls(
      space, source_points, target_points);

  auto approx_values = mls.interpolate(space, source_values);

  double l2_error;
  Kokkos::parallel_reduce(
      "Example::l2_error",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_points),
      KOKKOS_LAMBDA(int const i, double &loc_error) {
        auto val = target_values(i) - approx_values(i);
        loc_error += val * val;
      },
      Kokkos::Sum<double>(l2_error));
  l2_error = Kokkos::sqrt(l2_error / num_points);

  std::cout << "L2 Error: " << l2_error << '\n';

  if (!dump_file.empty())
    dump_msl_data(dump_file, source_points, target_points, source_values,
                  target_values, approx_values);
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  std::size_t num_points;
  std::string dump_file;
  boost::program_options::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help", "show help message")
    ("points",
      boost::program_options::value<std::size_t>(&num_points)
        ->default_value(1000),
      "Sets the number of points in the [-1/2 ; 1/2] range")
    ("dump",
      boost::program_options::value<std::string>(&dump_file)
        ->default_value(""),
      "Dump file name (as csv format)");
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

  mls_example(num_points, dump_file);
  return 0;
}