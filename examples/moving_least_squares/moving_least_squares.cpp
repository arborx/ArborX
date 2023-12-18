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
#include <optional>

using Point = ArborX::ExperimentalHyperGeometry::Point<2, double>;
constexpr int DIM = ArborX::GeometryTraits::dimension<Point>::value;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = typename ExecutionSpace::memory_space;

// Dumps the data in a file in a csv format
void dump_mls_data(std::string const &dump_filename,
                   Kokkos::View<Point *, MemorySpace> const source_points,
                   Kokkos::View<Point *, MemorySpace> const target_points,
                   Kokkos::View<double *, MemorySpace> const source_values,
                   Kokkos::View<double *, MemorySpace> const target_values,
                   Kokkos::View<double *, MemorySpace> const approx_values)
{
  std::fstream source_fs("source-" + dump_filename,
                         std::ios::out | std::ios::trunc);
  if (!source_fs)
    throw std::runtime_error("Unable to open/create file source-" +
                             dump_filename);
  std::fstream target_fs("target-" + dump_filename,
                         std::ios::out | std::ios::trunc);
  if (!target_fs)
    throw std::runtime_error("Unable to open/create file target-" +
                             dump_filename);

  for (int i = 0; i < DIM; i++)
    source_fs << "source coord " << i << ';';
  source_fs << "source value\n";
  for (int i = 0; i < DIM; i++)
    target_fs << "target coord " << i << ';';
  target_fs << "target value;approx value\n";

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

  int const source_num_points = source_points.extent(0);
  int const target_num_points = target_points.extent(0);
  for (int i = 0; i < source_num_points; i++)
  {
    for (int j = 0; j < DIM; j++)
      source_fs << source_points_host(i)[j] << ';';
    source_fs << source_values_host(i) << '\n';
  }
  for (int i = 0; i < target_num_points; i++)
  {
    for (int j = 0; j < DIM; j++)
      target_fs << target_points_host(i)[j] << ';';
    target_fs << target_values_host(i) << ';' << approx_values_host(i) << '\n';
  }
}

// Evenly fills a [-1, 1] box centered at 0
void filledBoxEven(Kokkos::View<Point *, MemorySpace> points)
{
  int n = points.extent(0);
  auto points_host = Kokkos::create_mirror_view(points);
  int const side_length = Kokkos::pow(points.extent(0), 1. / DIM);

  for (int i = 0; i < n; ++i)
  {
    int j = i;
    for (int d = 0; d < DIM; ++d)
    {
      points_host(i)[d] = 2 * (j % side_length) * 1. / (side_length - 1) - 1.;
      j = j / side_length;
    }
  }

  Kokkos::deep_copy(points, points_host);
}

// Switches the set of points to the new base
void basisChange(ExecutionSpace space,
                 Kokkos::View<Point *, MemorySpace> points,
                 Kokkos::View<Point[DIM], MemorySpace> const basis)
{
  int n = points.extent(0);
  Kokkos::parallel_for(
      "Example::basis_change", Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
      KOKKOS_LAMBDA(int const i) {
        Point t = Point{};
        for (int j = 0; j < DIM; j++)
          for (int k = 0; k < DIM; k++)
            t[j] += points(i)[k] * basis(k)[j];
        points(i) = t;
      });
}

// Funtion to approximate
KOKKOS_INLINE_FUNCTION double functionToApproximate(Point const &p)
{
  auto const x = p[0];
  return (x > 0) ? 1 : 0;
}

void mls_example(int source_num_points, int target_num_points,
                 std::optional<int> num_neighbors, std::string const &dump_file)
{
  ExecutionSpace space{};

  // Generation of points
  Kokkos::View<Point *, MemorySpace> source_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::source_points"),
      source_num_points);
  Kokkos::View<double *, MemorySpace> source_values(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::source_values"),
      source_num_points);
  Kokkos::View<Point *, MemorySpace> target_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::target_points"),
      target_num_points);
  Kokkos::View<double *, MemorySpace> target_values(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::target_values"),
      target_num_points);
  Kokkos::View<double *, MemorySpace> approx_values(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::approx_values"),
      0);

  filledBoxEven(source_points);
  filledBoxEven(target_points);

  // Basis change
  Kokkos::View<Point[DIM], MemorySpace> target_basis("Example::target_basis");
  Kokkos::parallel_for(
      "Example::fill_basis", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1),
      KOKKOS_LAMBDA(int const) {
        auto const sqrt2 = Kokkos::sqrt(2.);
        target_basis(0) = {.5 * sqrt2 / 2, .5 * sqrt2 / 2};
        target_basis(1) = {.5 * -sqrt2 / 2, .5 * sqrt2 / 2};
      });
  basisChange(space, target_points, target_basis);

  Kokkos::parallel_for(
      "Example::fill_views",
      Kokkos::RangePolicy<ExecutionSpace>(
          space, 0, Kokkos::max(source_num_points, target_num_points)),
      KOKKOS_LAMBDA(int const i) {
        if (i < source_num_points)
          source_values(i) = functionToApproximate(source_points(i));
        if (i < target_num_points)
          target_values(i) = functionToApproximate(target_points(i));
      });

  ArborX::Interpolation::MovingLeastSquares<MemorySpace, double> mls(
      space, source_points, target_points, num_neighbors,
      ArborX::Interpolation::CRBF::Wendland<0>{},
      ArborX::Interpolation::PolynomialDegree<2>{});
  mls.interpolate(space, source_values, approx_values);

  double l2_error;
  Kokkos::parallel_reduce(
      "Example::l2_error",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, target_num_points),
      KOKKOS_LAMBDA(int const i, double &loc_error) {
        auto val = target_values(i) - approx_values(i);
        loc_error += val * val;
      },
      Kokkos::Sum<double>(l2_error));
  l2_error = Kokkos::sqrt(l2_error / target_num_points);

  std::cout << "L2 Error: " << l2_error << '\n';

  if (!dump_file.empty())
    dump_mls_data(dump_file, source_points, target_points, source_values,
                  target_values, approx_values);
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  std::size_t source_num_points;
  std::size_t target_num_points;
  std::string dump_file;
  std::size_t num_neighbors;
  boost::program_options::options_description desc("Allowed options");

  // clang-format off
  desc.add_options()
    ("help", "show help message")
    ("src",
      boost::program_options::value<std::size_t>(&source_num_points)
        ->default_value(Kokkos::pow(100, DIM)),
      "Sets the number of source points")
    ("tgt",
      boost::program_options::value<std::size_t>(&target_num_points)
        ->default_value(Kokkos::pow(100, DIM)),
      "Sets the number of target points")
    ("neighbors",
      boost::program_options::value<std::size_t>(&num_neighbors)
        ->default_value(0),
      "Sets the number of neighbors")
    ("dump",
      boost::program_options::value<std::string>(&dump_file)
        ->default_value(""),
      "Dumps evaluation and approximation in filename (in csv format)");
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

  mls_example(source_num_points, target_num_points,
              num_neighbors == 0 ? std::nullopt : std::optional(num_neighbors),
              dump_file);
  return 0;
}