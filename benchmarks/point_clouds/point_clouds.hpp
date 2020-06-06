/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_POINT_CLOUDS_HPP
#define ARBORX_POINT_CLOUDS_HPP

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <fstream>
#include <random>

enum class PointCloudType
{
  filled_box,
  hollow_box,
  filled_sphere,
  hollow_sphere
};

PointCloudType to_point_cloud_enum(std::string const &str)
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

template <typename Layout, typename DeviceType>
void filledBoxCloud(
    double const half_edge,
    Kokkos::View<ArborX::Point *, Layout, DeviceType> random_points)
{
  static_assert(
      Kokkos::Impl::MemorySpaceAccess<
          Kokkos::HostSpace, typename DeviceType::memory_space>::accessible,
      "The View should be accessible on the Host");
  std::uniform_real_distribution<double> distribution(-half_edge, half_edge);
  std::default_random_engine generator;
  auto random = [&distribution, &generator]() {
    return distribution(generator);
  };
  unsigned int const n = random_points.extent(0);
  for (unsigned int i = 0; i < n; ++i)
    random_points(i) = {{random(), random(), random()}};
}

template <typename Layout, typename DeviceType>
void hollowBoxCloud(
    double const half_edge,
    Kokkos::View<ArborX::Point *, Layout, DeviceType> random_points)
{
  static_assert(
      Kokkos::Impl::MemorySpaceAccess<
          Kokkos::HostSpace, typename DeviceType::memory_space>::accessible,
      "The View should be accessible on the Host");
  std::uniform_real_distribution<double> distribution(-half_edge, half_edge);
  std::default_random_engine generator;
  auto random = [&distribution, &generator]() {
    return distribution(generator);
  };
  unsigned int const n = random_points.extent(0);
  for (unsigned int i = 0; i < n; ++i)
  {
    unsigned int face = i % 6;
    switch (face)
    {
    case 0:
    {
      random_points(i) = {{-half_edge, random(), random()}};

      break;
    }
    case 1:
    {
      random_points(i) = {{half_edge, random(), random()}};

      break;
    }
    case 2:
    {
      random_points(i) = {{random(), -half_edge, random()}};

      break;
    }
    case 3:
    {
      random_points(i) = {{random(), half_edge, random()}};

      break;
    }
    case 4:
    {
      random_points(i) = {{random(), random(), -half_edge}};

      break;
    }
    case 5:
    {
      random_points(i) = {{random(), random(), half_edge}};

      break;
    }
    default:
    {
      throw std::runtime_error("Your compiler is broken");
    }
    }
  }
}

template <typename Layout, typename DeviceType>
void filledSphereCloud(
    double const radius,
    Kokkos::View<ArborX::Point *, Layout, DeviceType> random_points)
{
  static_assert(
      Kokkos::Impl::MemorySpaceAccess<
          Kokkos::HostSpace, typename DeviceType::memory_space>::accessible,
      "The View should be accessible on the Host");
  std::default_random_engine generator;

  std::uniform_real_distribution<double> distribution(-radius, radius);
  auto random = [&distribution, &generator]() {
    return distribution(generator);
  };

  unsigned int const n = random_points.extent(0);
  for (unsigned int i = 0; i < n; ++i)
  {
    bool point_accepted = false;
    while (!point_accepted)
    {
      double const x = random();
      double const y = random();
      double const z = random();

      // Only accept points that are in the sphere
      if (std::sqrt(x * x + y * y + z * z) <= radius)
      {
        random_points(i) = {{x, y, z}};
        point_accepted = true;
      }
    }
  }
}

template <typename Layout, typename DeviceType>
void hollowSphereCloud(
    double const radius,
    Kokkos::View<ArborX::Point *, Layout, DeviceType> random_points)
{
  static_assert(
      Kokkos::Impl::MemorySpaceAccess<
          Kokkos::HostSpace, typename DeviceType::memory_space>::accessible,
      "The View should be accessible on the Host");
  std::default_random_engine generator;

  std::uniform_real_distribution<double> distribution(-1., 1.);
  auto random = [&distribution, &generator]() {
    return distribution(generator);
  };

  unsigned int const n = random_points.extent(0);
  for (unsigned int i = 0; i < n; ++i)
  {
    double const x = random();
    double const y = random();
    double const z = random();
    double const norm = std::sqrt(x * x + y * y + z * z);

    random_points(i) = {
        {radius * x / norm, radius * y / norm, radius * z / norm}};
  }
}

template <typename DeviceType>
void generatePointCloud(PointCloudType const point_cloud_type,
                        double const length,
                        Kokkos::View<ArborX::Point *, DeviceType> random_points)
{
  auto random_points_host = Kokkos::create_mirror_view(random_points);
  switch (point_cloud_type)
  {
  case PointCloudType::filled_box:
    filledBoxCloud(length, random_points_host);
    break;
  case PointCloudType::hollow_box:
    hollowBoxCloud(length, random_points_host);
    break;
  case PointCloudType::filled_sphere:
    filledSphereCloud(length, random_points_host);
    break;
  case PointCloudType::hollow_sphere:
    hollowSphereCloud(length, random_points_host);
    break;
  default:
    throw ArborX::SearchException("not implemented");
  }
  Kokkos::deep_copy(random_points, random_points_host);
}

#endif
