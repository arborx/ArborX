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
#include <ArborX_Ray.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <iostream>
#include <random>

template <typename MemorySpace>
struct SpheresToBoxes
{
  Kokkos::View<ArborX::Sphere *, MemorySpace> _spheres;
};

template <typename MemorySpace>
struct ArborX::AccessTraits<SpheresToBoxes<MemorySpace>, ArborX::PrimitivesTag>
{
  using memory_space = MemorySpace;

  KOKKOS_FUNCTION static std::size_t
  size(SpheresToBoxes<MemorySpace> const &stob)
  {
    return stob._spheres.extent(0);
  }
  KOKKOS_FUNCTION static ArborX::Box
  get(SpheresToBoxes<MemorySpace> const &stobs, std::size_t const i)
  {
    auto const &sphere = stobs._spheres(i);
    auto const &c = sphere.centroid();
    auto const r = sphere.radius();
    return {{c[0] - r, c[1] - r, c[2] - r}, {c[0] + r, c[1] + r, c[2] + r}};
  }
};

template <typename MemorySpace>
struct Rays
{
  Kokkos::View<ArborX::Experimental::Ray *, MemorySpace> _rays;
};

template <typename MemorySpace>
struct ArborX::AccessTraits<Rays<MemorySpace>, ArborX::PredicatesTag>
{
  using memory_space = MemorySpace;

  KOKKOS_FUNCTION static std::size_t size(Rays<MemorySpace> const &rays)
  {
    return rays._rays.extent(0);
  }
  KOKKOS_FUNCTION static auto get(Rays<MemorySpace> const &rays, std::size_t i)
  {
    return attach(intersects(rays._rays(i)), (int)i);
  }
};

template <typename MemorySpace>
struct AccumRaySphereInterDist
{
  Kokkos::View<ArborX::Sphere *, MemorySpace> _spheres;
  Kokkos::View<float *, MemorySpace> _accumulator;

  template <typename Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate,
                                  int const primitive_index) const
  {
    auto const &ray = ArborX::getGeometry(predicate);
    auto const &sphere = _spheres(primitive_index);

    float const length = overlapDistance(ray, sphere);
    int const i = getData(predicate);

    Kokkos::atomic_add(&_accumulator(i), length);
  }
};

int main(int argc, char *argv[])
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  Kokkos::ScopeGuard guard(argc, argv);

  namespace bpo = boost::program_options;

  int num_spheres;
  int num_rays;
  float L;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ( "help", "help message" )
    ("spheres", bpo::value<int>(&num_spheres)->default_value(100), "number of spheres")
    ("rays", bpo::value<int>(&num_rays)->default_value(10000), "number of rays")
    ("L", bpo::value<float>(&L)->default_value(100.0), "size of the domain")
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

  std::cout << "ArborX version: " << ArborX::version() << std::endl;
  std::cout << "ArborX hash   : " << ArborX::gitCommitHash() << std::endl;
  std::cout << "Kokkos version: " << KokkosExt::version() << std::endl;

  std::uniform_real_distribution<float> uniform{0.0, 1.0};
  std::default_random_engine gen;
  auto rand_uniform = [&]() { return uniform(gen); };

  // Random parameters for Gaussian distribution of radii
  float const mu_R = 1.0;
  float const sigma_R = mu_R / 3.0;

  std::normal_distribution<> normal{mu_R, sigma_R};
  auto rand_normal = [&]() { return std::max(normal(gen), 0.0); };

  // Construct spheres
  //
  // The centers of spheres are uniformly sampling the domain. The radii of
  // spheres have Gaussian (mu_R, sigma_R) sampling.
  Kokkos::View<ArborX::Sphere *, MemorySpace> spheres(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "spheres"), num_spheres);
  auto spheres_host = Kokkos::create_mirror_view(spheres);
  for (int i = 0; i < num_spheres; ++i)
  {
    spheres_host(i) = {
        {rand_uniform() * L, rand_uniform() * L, rand_uniform() * L},
        rand_normal()};
  }
  Kokkos::deep_copy(spheres, spheres_host);

  // Construct rays
  //
  // The origins of rays are uniformly sampling the bottom surface of the
  // domain. The direction vectors are uniformly sampling of a cosine-weighted
  // hemisphere, It requires expressing the direction vector in the spherical
  // coordinates as:
  //    {sinpolar * cosazimuth, sinpolar * sinazimuth, cospolar}
  // A detailed description can be found in the slides here (slide 47):
  // https://cg.informatik.uni-freiburg.de/course_notes/graphics2_08_renderingEquation.pdf
  Kokkos::View<ArborX::Experimental::Ray *, MemorySpace> rays(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "rays"), num_rays);
  auto rays_host = Kokkos::create_mirror_view(rays);

  for (int i = 0; i < num_rays; ++i)
  {
    float xi_1 = rand_uniform();
    float xi_2 = rand_uniform();

    rays_host(i) = {ArborX::Point{rand_uniform() * L, rand_uniform() * L, 0.f},
                    ArborX::Experimental::Vector{
                        float(std::cos(2 * M_PI * xi_2) * std::sqrt(xi_1)),
                        float(std::sin(2 * M_PI * xi_2) * std::sqrt(xi_1)),
                        std::sqrt(1.f - xi_1)}};
  }
  Kokkos::deep_copy(rays, rays_host);

  Kokkos::Timer timer;

  ExecutionSpace exec_space{};

  exec_space.fence();
  timer.reset();
  ArborX::BVH<MemorySpace> bvh{exec_space,
                               SpheresToBoxes<MemorySpace>{spheres}};

  Kokkos::View<float *, MemorySpace> accumulator("accumulator", num_rays);
  bvh.query(exec_space, Rays<MemorySpace>{rays},
            AccumRaySphereInterDist<MemorySpace>{spheres, accumulator});
  exec_space.fence();
  auto time = timer.seconds();

  auto accumulator_avg =
      ArborX::accumulate(exec_space, accumulator, 0.f) / num_rays;

  printf("time          : %.3f   [%.3fM ray/sec]\n", time,
         num_rays / (1000000 * time));
  printf("ray avg       : %.3f\n", accumulator_avg);

  return EXIT_SUCCESS;
}
