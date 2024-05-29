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

#ifndef ARBORX_BENCHMARK_POINT_CLOUDS_HPP
#define ARBORX_BENCHMARK_POINT_CLOUDS_HPP

#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
#include <ArborX_Exception.hpp>
#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <fstream>
#include <random>

namespace ArborXBenchmark
{

enum class PointCloudType
{
  filled_box,
  hollow_box,
  filled_sphere,
  hollow_sphere
};

inline PointCloudType to_point_cloud_enum(std::string const &str)
{
  if (str == "filled_box")
    return PointCloudType::filled_box;
  if (str == "hollow_box")
    return PointCloudType::hollow_box;
  if (str == "filled_sphere")
    return PointCloudType::filled_sphere;
  if (str == "hollow_sphere")
    return PointCloudType::hollow_sphere;
  throw std::runtime_error(str +
                           " doesn't correspond to any known PointCloudType!");
}

namespace Details
{

template <typename ExecutionSpace, typename Points>
void filledBoxCloud(ExecutionSpace const &exec, double const half_edge,
                    Points &random_points)
{
  using Point = typename Points::value_type;
  constexpr auto DIM = ArborX::GeometryTraits::dimension_v<Point>;
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<Point>;

  using GeneratorPool = Kokkos::Random_XorShift1024_Pool<ExecutionSpace>;
  using GeneratorType = typename GeneratorPool::generator_type;
  constexpr unsigned int batch_size = 8;

  GeneratorPool random_pool(0);
  unsigned int const n = random_points.extent(0);
  Kokkos::parallel_for(
      "ArborXBenchmark::filledBoxCloud::generate",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n / batch_size),
      KOKKOS_LAMBDA(int i) {
        auto generator = random_pool.get_state();
        auto random = [&generator, half_edge]() {
          return Kokkos::rand<GeneratorType, Coordinate>::draw(
              generator, -half_edge, half_edge);
        };

        auto begin = i * batch_size;
        auto end = Kokkos::min((i + 1) * batch_size, n);
        for (unsigned int k = begin; k < end; ++k)
          for (int d = 0; d < DIM; ++d)
            random_points(k)[d] = random();

        random_pool.free_state(generator);
      });
}

template <typename ExecutionSpace, typename Points>
void hollowBoxCloud(ExecutionSpace const &exec, double const half_edge,
                    Points &random_points)
{
  using Point = typename Points::value_type;
  constexpr auto DIM = ArborX::GeometryTraits::dimension_v<Point>;
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<Point>;

  using GeneratorPool = Kokkos::Random_XorShift1024_Pool<ExecutionSpace>;
  using GeneratorType = typename GeneratorPool::generator_type;
  constexpr unsigned int batch_size = 8;

  GeneratorPool random_pool(0);
  unsigned int const n = random_points.extent(0);
  // Points are cyclically placed on the faces of a box
  Kokkos::parallel_for(
      "ArborXBenchmark::hollowBoxCloud::generate",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n / batch_size),
      KOKKOS_LAMBDA(int i) {
        auto generator = random_pool.get_state();
        auto random = [&generator, half_edge]() {
          return Kokkos::rand<GeneratorType, Coordinate>::draw(
              generator, -half_edge, half_edge);
        };

        auto begin = i * batch_size;
        auto end = Kokkos::min((i + 1) * batch_size, n);
        for (unsigned int k = begin; k < end; ++k)
        {
          // For 3D, the order is
          // Indices: 0  1  2  3  4  5  6  7  8  9
          // Axes   : 0- 0+ 1- 1+ 2- 2+ 0- 0+ 1- 1+
          int axis = (k / 2) % DIM;
          for (int d = 0; d < DIM; ++d)
          {
            if (d != axis)
              random_points(k)[d] = random();
            else if (k % 2 == 0)
              random_points(k)[d] = -half_edge;
            else
              random_points(k)[d] = half_edge;
          }
        }

        random_pool.free_state(generator);
      });
}

template <typename ExecutionSpace, typename Points>
void filledSphereCloud(ExecutionSpace const &exec, double const radius,
                       Points &random_points)
{
  using Point = typename Points::value_type;
  constexpr auto DIM = ArborX::GeometryTraits::dimension_v<Point>;
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<Point>;

  using GeneratorPool = Kokkos::Random_XorShift1024_Pool<ExecutionSpace>;
  using GeneratorType = typename GeneratorPool::generator_type;
  constexpr unsigned int batch_size = 8;

  GeneratorPool random_pool(0);
  unsigned int const n = random_points.extent(0);
  Kokkos::parallel_for(
      "ArborXBenchmark::filledSphereCloud::generate",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n / batch_size),
      KOKKOS_LAMBDA(int i) {
        auto generator = random_pool.get_state();
        auto random = [&generator, radius]() {
          return Kokkos::rand<GeneratorType, Coordinate>::draw(generator,
                                                               -radius, radius);
        };

        auto begin = i * batch_size;
        auto end = Kokkos::min((i + 1) * batch_size, n);
        for (unsigned int k = begin; k < end; ++k)
        {
          do
          {
            Point p;
            Coordinate norm = 0;
            for (int d = 0; d < DIM; ++d)
            {
              p[d] = random();
              norm += p[d] * p[d];
            }
            norm = Kokkos::sqrt(norm);

            // Only accept points that are in the sphere
            if (norm <= radius)
            {
              random_points(k) = p;
              break;
            }
          } while (true);
        }

        random_pool.free_state(generator);
      });
}

template <typename ExecutionSpace, typename Points>
void hollowSphereCloud(ExecutionSpace const &exec, double const radius,
                       Points &random_points)
{
  using Point = typename Points::value_type;
  constexpr auto DIM = ArborX::GeometryTraits::dimension_v<Point>;
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<Point>;

  using GeneratorPool = Kokkos::Random_XorShift1024_Pool<ExecutionSpace>;
  using GeneratorType = typename GeneratorPool::generator_type;
  constexpr unsigned int batch_size = 8;

  GeneratorPool random_pool(0);
  unsigned int const n = random_points.extent(0);
  Kokkos::parallel_for(
      "ArborXBenchmark::hollowSphereCloud::generate",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n / batch_size),
      KOKKOS_LAMBDA(int i) {
        auto generator = random_pool.get_state();
        auto random = [&generator, radius]() {
          return Kokkos::rand<GeneratorType, Coordinate>::draw(generator,
                                                               -radius, radius);
        };

        auto begin = i * batch_size;
        auto end = Kokkos::min((i + 1) * batch_size, n);
        for (unsigned int k = begin; k < end; ++k)
        {
          Point p;
          Coordinate norm = 0;
          for (int d = 0; d < DIM; ++d)
          {
            p[d] = random();
            norm += p[d] * p[d];
          }
          norm = Kokkos::sqrt(norm);

          for (int d = 0; d < DIM; ++d)
            random_points(k)[d] = radius * p[d] / norm;
        }

        random_pool.free_state(generator);
      });
}

} // namespace Details

template <typename ExecutionSpace, typename Points>
void generatePointCloud(ExecutionSpace const &exec,
                        PointCloudType const point_cloud_type,
                        double const length, Points &random_points)
{
  static_assert(Kokkos::is_view_v<Points>);

  using Point = typename Points::value_type;

  using namespace ArborX::GeometryTraits;
  check_valid_geometry_traits(Point{});
  static_assert(is_point_v<Point>, "ArborX: View must contain point values");

  switch (point_cloud_type)
  {
  case PointCloudType::filled_box:
    Details::filledBoxCloud(exec, length, random_points);
    break;
  case PointCloudType::hollow_box:
    Details::hollowBoxCloud(exec, length, random_points);
    break;
  case PointCloudType::filled_sphere:
    Details::filledSphereCloud(exec, length, random_points);
    break;
  case PointCloudType::hollow_sphere:
    Details::hollowSphereCloud(exec, length, random_points);
    break;
  default:
    throw ArborX::SearchException("not implemented");
  }
}

} // namespace ArborXBenchmark

#endif
