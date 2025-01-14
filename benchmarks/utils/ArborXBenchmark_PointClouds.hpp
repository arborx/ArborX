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

#ifndef ARBORX_BENCHMARK_POINT_CLOUDS_HPP
#define ARBORX_BENCHMARK_POINT_CLOUDS_HPP

#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

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

template <typename Point>
class PointGenerationFunctor
{
private:
  static_assert(ArborX::GeometryTraits::is_point_v<Point>);

  static constexpr int DIM = ArborX::GeometryTraits::dimension_v<Point>;
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<Point>;

  PointCloudType _point_cloud_type;

public:
  PointGenerationFunctor(PointCloudType point_cloud_type)
      : _point_cloud_type(point_cloud_type)
  {}

  template <typename Generator>
  KOKKOS_FUNCTION auto operator()(int i, Generator &generator) const
  {
    switch (_point_cloud_type)
    {
    case PointCloudType::filled_box:
      return filledBoxPoint(generator);
    case PointCloudType::hollow_box:
      return hollowBoxPoint(i, generator);
    case PointCloudType::filled_sphere:
      return filledSpherePoint(generator);
    case PointCloudType::hollow_sphere:
      return hollowSpherePoint(generator);
    default:
      Kokkos::abort("ArborX: implementation bug");
    }
  }

private:
  template <typename Generator>
  KOKKOS_FUNCTION auto filledBoxPoint(Generator &generator) const
  {
    auto random = [&generator]() {
      return Kokkos::rand<Generator, Coordinate>::draw(generator, -1, 1);
    };

    Point p;
    for (int d = 0; d < DIM; ++d)
      p[d] = random();
    return p;
  }

  template <typename Generator>
  KOKKOS_FUNCTION auto hollowBoxPoint(int i, Generator &generator) const
  {
    auto random = [&generator]() {
      return Kokkos::rand<Generator, Coordinate>::draw(generator, -1, 1);
    };

    Point p;
    // For 3D, the order is
    // Indices: 0  1  2  3  4  5  6  7  8  9
    // Axes   : 0- 0+ 1- 1+ 2- 2+ 0- 0+ 1- 1+
    int axis = (i / 2) % DIM;
    for (int d = 0; d < DIM; ++d)
    {
      if (d != axis)
        p[d] = random();
      else if (i % 2 == 0)
        p[d] = -1;
      else
        p[d] = 1;
    }

    return p;
  }

  template <typename Generator>
  KOKKOS_FUNCTION auto filledSpherePoint(Generator &generator) const
  {
    auto random01 = [&generator]() {
      return Kokkos::rand<Generator, Coordinate>::draw(generator, 0, 1);
    };
    auto random_normal = [&generator]() {
      return generator.normal(); // draws double
    };

    Point p;
    Coordinate norm_squared;
    do
    {
      norm_squared = 0;
      for (int d = 0; d < DIM; ++d)
      {
        p[d] = random_normal();
        norm_squared += p[d] * p[d];
      }
    } while (norm_squared == 0);
    auto norm = Kokkos::sqrt(norm_squared);

    auto scaling = Kokkos::pow(random01(), 1 / (Coordinate)DIM) / norm;
    for (int d = 0; d < DIM; ++d)
      p[d] *= scaling;

    return p;
  }

  template <typename Generator>
  KOKKOS_FUNCTION auto hollowSpherePoint(Generator &generator) const
  {
    auto random_normal = [&generator]() {
      return generator.normal(); // draws double
    };

    Point p;
    Coordinate norm_squared;
    do
    {
      norm_squared = 0;
      for (int d = 0; d < DIM; ++d)
      {
        p[d] = random_normal();
        norm_squared += p[d] * p[d];
      }
    } while (norm_squared == 0);
    auto norm = Kokkos::sqrt(norm_squared);

    for (int d = 0; d < DIM; ++d)
      p[d] /= norm;

    return p;
  }
};

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

  KOKKOS_ASSERT(point_cloud_type == PointCloudType::filled_box ||
                point_cloud_type == PointCloudType::hollow_box ||
                point_cloud_type == PointCloudType::filled_sphere ||
                point_cloud_type == PointCloudType::hollow_sphere);

  static constexpr int DIM = ArborX::GeometryTraits::dimension_v<Point>;

  using GeneratorPool = Kokkos::Random_XorShift1024_Pool<ExecutionSpace>;
  constexpr unsigned int batch_size = 8;

  GeneratorPool random_pool(0);
  Details::PointGenerationFunctor<Point> functor(point_cloud_type);
  unsigned int const n = random_points.extent(0);
  Kokkos::parallel_for(
      "ArborXBenchmark::generatePointCloud::generate",
      Kokkos::RangePolicy(exec, 0, n / batch_size), KOKKOS_LAMBDA(int i) {
        auto generator = random_pool.get_state();

        auto begin = i * batch_size;
        auto end = Kokkos::min((i + 1) * batch_size, n);
        for (unsigned int k = begin; k < end; ++k)
        {
          random_points(k) = functor(k, generator);

          for (int d = 0; d < DIM; ++d)
            random_points(k)[d] *= length;
        }

        random_pool.free_state(generator);
      });
}

} // namespace ArborXBenchmark

#endif
