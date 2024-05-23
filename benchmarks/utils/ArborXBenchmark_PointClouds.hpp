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

template <class Point, typename... ViewProperties>
void filledBoxCloud(double const half_edge,
                    Kokkos::View<Point *, ViewProperties...> random_points)
{
  namespace KokkosExt = ArborX::Details::KokkosExt;
  static_assert(
      KokkosExt::is_accessible_from_host<decltype(random_points)>::value,
      "The View should be accessible on the Host");
  std::uniform_real_distribution<double> distribution(-half_edge, half_edge);
  constexpr int seed = 14;
  std::default_random_engine generator(seed);
  auto random = [&distribution, &generator]() {
    return distribution(generator);
  };
  unsigned int const n = random_points.extent(0);
  constexpr auto DIM = ArborX::GeometryTraits::dimension<Point>::value;
  for (unsigned int i = 0; i < n; ++i)
    for (int d = 0; d < DIM; ++d)
      random_points(i)[d] = random();
}

template <class Point, typename... ViewProperties>
void hollowBoxCloud(double const half_edge,
                    Kokkos::View<Point *, ViewProperties...> random_points)
{
  namespace KokkosExt = ArborX::Details::KokkosExt;
  static_assert(
      KokkosExt::is_accessible_from_host<decltype(random_points)>::value,
      "The View should be accessible on the Host");
  std::uniform_real_distribution<double> distribution(-half_edge, half_edge);
  constexpr int seed = 15;
  std::default_random_engine generator(seed);
  auto random = [&distribution, &generator]() {
    return distribution(generator);
  };
  unsigned int const n = random_points.extent(0);
  constexpr auto DIM = ArborX::GeometryTraits::dimension<Point>::value;
  // Points are cyclically placed on the faces of a box
  constexpr int num_faces = 2 * DIM;
  for (int axis = 0; axis < DIM; ++axis)
  {
    for (unsigned int i = 2 * axis; i < n; i += num_faces)
      for (int d = 0; d < DIM; ++d)
        random_points(i)[d] = (d != axis ? random() : -half_edge);

    for (unsigned int i = 2 * axis + 1; i < n; i += num_faces)
      for (int d = 0; d < DIM; ++d)
        random_points(i)[d] = (d != axis ? random() : +half_edge);
  }
}

template <class Point, typename... ViewProperties>
void filledSphereCloud(double const radius,
                       Kokkos::View<Point *, ViewProperties...> random_points)
{
  namespace KokkosExt = ArborX::Details::KokkosExt;
  static_assert(
      KokkosExt::is_accessible_from_host<decltype(random_points)>::value,
      "The View should be accessible on the Host");
  constexpr int seed = 16;
  std::default_random_engine generator(seed);

  std::uniform_real_distribution<double> distribution(-radius, radius);
  auto random = [&distribution, &generator]() {
    return distribution(generator);
  };

  unsigned int const n = random_points.extent(0);
  constexpr auto DIM = ArborX::GeometryTraits::dimension<Point>::value;
  for (unsigned int i = 0; i < n; ++i)
  {
    bool point_accepted = false;
    while (!point_accepted)
    {
      Point p;
      double norm = 0.f;
      for (int d = 0; d < DIM; ++d)
      {
        double v = random();
        p[d] = v;
        norm += v * v;
      }
      norm = std::sqrt(norm);

      // Only accept points that are in the sphere
      if (norm <= radius)
      {
        random_points(i) = p;
        point_accepted = true;
      }
    }
  }
}

template <class Point, typename... ViewProperties>
void hollowSphereCloud(double const radius,
                       Kokkos::View<Point *, ViewProperties...> random_points)
{
  namespace KokkosExt = ArborX::Details::KokkosExt;
  static_assert(
      KokkosExt::is_accessible_from_host<decltype(random_points)>::value,
      "The View should be accessible on the Host");
  constexpr int seed = 17;
  std::default_random_engine generator(seed);

  std::normal_distribution<float> distribution(0.f, 1.f);
  auto random = [&distribution, &generator]() {
    return distribution(generator);
  };

  unsigned int const n = random_points.extent(0);
  constexpr auto DIM = ArborX::GeometryTraits::dimension<Point>::value;
  for (unsigned int i = 0; i < n; ++i)
  {
    double v[DIM];
    double norm = 0.;
    for (int d = 0; d < DIM; ++d)
    {
      v[d] = random();
      norm += v[d] * v[d];
    }
    norm = std::sqrt(norm);

    for (int d = 0; d < DIM; ++d)
      random_points(i)[d] = radius * v[d] / norm;
  }
}

} // namespace Details

template <class Point, typename DeviceType>
void generatePointCloud(PointCloudType const point_cloud_type,
                        double const length,
                        Kokkos::View<Point *, DeviceType> random_points)
{
  using namespace ArborX::GeometryTraits;
  check_valid_geometry_traits(Point{});
  static_assert(is_point_v<Point>, "ArborX: View must contain point values");

  auto random_points_host = Kokkos::create_mirror_view(random_points);
  switch (point_cloud_type)
  {
  case PointCloudType::filled_box:
    Details::filledBoxCloud(length, random_points_host);
    break;
  case PointCloudType::hollow_box:
    Details::hollowBoxCloud(length, random_points_host);
    break;
  case PointCloudType::filled_sphere:
    Details::filledSphereCloud(length, random_points_host);
    break;
  case PointCloudType::hollow_sphere:
    Details::hollowSphereCloud(length, random_points_host);
    break;
  default:
    throw ArborX::SearchException("not implemented");
  }
  Kokkos::deep_copy(random_points, random_points_host);
}

} // namespace ArborXBenchmark

#endif
