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
#include <random>

using Point = ArborX::ExperimentalHyperGeometry::Point<2, double>;
constexpr int DIM = ArborX::GeometryTraits::dimension<Point>::value;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = typename ExecutionSpace::memory_space;

// Randomly fills a 2 * half_edge box centered at 0
void filledBoxRandom(double half_edge,
                     Kokkos::View<Point *, MemorySpace> points)
{
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

// Evenly fills a 2 * half_edge box centered at 0
void filledBoxEven(double half_edge, std::array<int, DIM> const line_len,
                   Kokkos::View<Point *, MemorySpace> points)
{
  int n = points.extent(0);
  auto points_host = Kokkos::create_mirror_view(points);

  for (int i = 0; i < n; ++i)
  {
    int j = i;
    for (int d = 0; d < DIM; ++d)
    {
      points_host(i)[d] =
          2 * (j % line_len[d]) * half_edge / (line_len[d] - 1) - half_edge;
      j = j / line_len[d];
    }
  }

  Kokkos::deep_copy(points, points_host);
}

// Switches the set of points to the new base
void basisChange(ExecutionSpace space,
                 Kokkos::View<Point *, MemorySpace> points,
                 Kokkos::View<Point[DIM], MemorySpace> const basis,
                 Point const offset)
{
  int n = points.extent(0);
  Kokkos::parallel_for(
      "Example::basis_change", Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
      KOKKOS_LAMBDA(int const i) {
        Point t = offset;
        for (int j = 0; j < DIM; j++)
          for (int k = 0; k < DIM; k++)
            t[j] += points(i)[k] * basis(k)[j];
        points(i) = t;
      });
}

// Dumps the data in a file in a csv format
void dump_msl_data(std::string const &dump_filename,
                   Kokkos::View<Point *, MemorySpace> const source_points,
                   Kokkos::View<Point *, MemorySpace> const target_points,
                   Kokkos::View<double *, MemorySpace> const source_values,
                   Kokkos::View<double *, MemorySpace> const target_values,
                   Kokkos::View<double *, MemorySpace> const approx_values)
{
  std::fstream dump_file_stream(dump_filename, std::ios::out | std::ios::trunc);
  if (!dump_file_stream)
    throw std::runtime_error("Unable to open/create file " + dump_filename);

  for (int i = 0; i < DIM; i++)
    dump_file_stream << "source coord " << i << ';';
  dump_file_stream << "source value;";
  for (int i = 0; i < DIM; i++)
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
    for (int j = 0; j < DIM; j++)
      dump_file_stream << source_points_host(i)[j] << ';';
    dump_file_stream << source_values_host(i) << ';';
    for (int j = 0; j < DIM; j++)
      dump_file_stream << target_points_host(i)[j] << ';';
    dump_file_stream << target_values_host(i) << ';' << approx_values_host(i)
                     << '\n';
  }
}

// Funtion to approximate
KOKKOS_INLINE_FUNCTION double functionToApproximate(Point const &p)
{
  auto const x = p[0];
  return (x > 0) ? 1 : 0;
}

void mls_example(std::size_t num_points, std::optional<int> num_neighbors,
                 std::string const &dump_file)
{
  ExecutionSpace space{};
  int const side_len = Kokkos::pow(num_points, 1. / DIM);

  // Basis change
  Kokkos::View<Point[DIM], MemorySpace> source_basis("Example::source_basis");
  Kokkos::View<Point[DIM], MemorySpace> target_basis("Example::target_basis");
  Kokkos::parallel_for(
      "Example::fill_basis", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1),
      KOKKOS_LAMBDA(int const) {
        for (int i = 0; i < DIM; i++)
        {
          source_basis(i) = Point{};
          target_basis(i) = Point{};
          for (int j = 0; j < DIM; j++)
          {
            source_basis(i)[j] = double(i == j);
            target_basis(i)[j] = double(i == j);
          }
        }
      });

  // Generation of points
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

  // Sets the meshes
  filledBoxEven(0.5, {side_len, side_len}, source_points);
  // filledBoxRandom(0.5, source_points);
  basisChange(space, source_points, source_basis, Point{});
  filledBoxEven(0.5, {side_len, side_len}, target_points);
  // filledBoxRandom(0.5, target_points);
  basisChange(space, target_points, target_basis, Point{});

  Kokkos::parallel_for(
      "Example::fill_views",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_points),
      KOKKOS_LAMBDA(int const i) {
        source_values(i) = functionToApproximate(source_points(i));
        target_values(i) = functionToApproximate(target_points(i));
      });

  ArborX::Interpolation::MovingLeastSquares<MemorySpace, double> mls(
      space, source_points, target_points, num_neighbors,
      ArborX::Interpolation::CRBF::Wendland<0>{},
      ArborX::Interpolation::PolynomialDegree<2>{});

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
  std::size_t num_neighbors;
  boost::program_options::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help", "show help message")
    ("points",
      boost::program_options::value<std::size_t>(&num_points)
        ->default_value(Kokkos::pow(100, DIM)),
      "Sets the number of points in the [-1/2 ; 1/2] range")
    ("neighbors",
      boost::program_options::value<std::size_t>(&num_neighbors)
        ->default_value(0),
      "Sets the number of neighbors")
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

  mls_example(num_points,
              num_neighbors == 0 ? std::nullopt : std::optional(num_neighbors),
              dump_file);
  return 0;
}