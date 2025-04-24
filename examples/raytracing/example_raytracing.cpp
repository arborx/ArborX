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

/*
 * This example demonstrates how to use ArborX for a raytracing example where
 * rays carry energy that they deposit onto given boxes as they traverse them.
 * The order in which these rays (which originate from one of the boxes) hit
 * the boxes is important in this case, since the ray loses energy on
 * intersection. The example shows two different ways to do that:
 * 1.) using a specialized traversal that orders all intersection in a heap
 *     so that the callbacks for a specific ray are called in the correct order
 *     (OrderedIntersectsBased namespace).
 * 2.) storing all intersections and doing the deposition of energy in a
 *     postprocessing step (IntersectsBased namespace).
 */

#include <ArborX.hpp>
#include <ArborX_Ray.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <boost/program_options.hpp>

#include <iostream>
#include <numeric>

using Point = ArborX::Point<3>;
using Box = ArborX::Box<3>;
using Ray = ArborX::Experimental::Ray<>;

// The total energy that is distributed across all rays.
float const total_energy = 4000.f;

// Energy a rays loses when passing through a cell.
KOKKOS_INLINE_FUNCTION float lost_energy(float ray_energy, float path_length)
{
  using Kokkos::expm1;
  return -ray_energy * expm1(-path_length);
}

namespace OrderedIntersectsBased
{
/*
 * Storage for the rays and access traits used in the query/traverse.
 */
template <typename MemorySpace>
struct Rays
{
  Kokkos::View<Ray *, MemorySpace> _rays;
};

/*
 * Callback to directly deposit energy.
 */
template <typename MemorySpace>
struct DepositEnergy
{
  Kokkos::View<Box *, MemorySpace> _boxes;
  Kokkos::View<float *, MemorySpace> _ray_energy;
  Kokkos::View<float *, MemorySpace> _energy;

  template <typename Predicate, typename Value>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate,
                                  Value const &value) const
  {
    float length;
    float entrylength;
    int const primitive_index = value.index;
    auto const &ray = ArborX::getGeometry(predicate);
    auto const &box = _boxes(primitive_index);
    int const predicate_index = ArborX::getData(predicate);
    float const kappa = 1.;
    overlapDistance(ray, box, length, entrylength);
    float const optical_path_length = kappa * length;

    float const energy_deposited =
        lost_energy(_ray_energy(predicate_index), optical_path_length);
    _ray_energy(predicate_index) += energy_deposited;
    Kokkos::atomic_add(&_energy(primitive_index), energy_deposited);
  }
};
} // namespace OrderedIntersectsBased

template <typename MemorySpace>
struct ArborX::AccessTraits<OrderedIntersectsBased::Rays<MemorySpace>>
{
  using memory_space = MemorySpace;
  using size_type = std::size_t;

  KOKKOS_FUNCTION
  static size_type size(OrderedIntersectsBased::Rays<MemorySpace> const &rays)
  {
    return rays._rays.extent(0);
  }
  KOKKOS_FUNCTION
  static auto get(OrderedIntersectsBased::Rays<MemorySpace> const &rays,
                  size_type i)
  {
    return attach(ordered_intersects(rays._rays(i)), (int)i);
  }
};

namespace IntersectsBased
{

/*
 * IntersectedCell is a storage container for all intersections between rays and
 * boxes that are detected when calling the AccumulateRaySphereIntersections
 * struct. The member variables that are relevant for sorting the intersection
 * according to box and ray are contained in the base class
 * IntersectedCellForSorting as performance improvement.
 */
struct IntersectedCellForSorting
{
  float entrylength;
  int ray_id;
  friend KOKKOS_FUNCTION bool operator<(IntersectedCellForSorting const &l,
                                        IntersectedCellForSorting const &r)
  {
    if (l.ray_id == r.ray_id)
      return l.entrylength < r.entrylength;
    return l.ray_id < r.ray_id;
  }
};

struct IntersectedCell : public IntersectedCellForSorting
{
  float optical_path_length; // optical distance through box
  int cell_id;               // box ID
  KOKKOS_FUNCTION IntersectedCell() = default;
  KOKKOS_FUNCTION IntersectedCell(float entry_length, float path_length,
                                  int primitive_index, int predicate_index)
      : IntersectedCellForSorting{entry_length, predicate_index}
      , optical_path_length(path_length)
      , cell_id(primitive_index)
  {}
};

/*
 * Callback for storing all intersections.
 */
template <typename MemorySpace>
struct AccumulateRaySphereIntersections
{
  Kokkos::View<Box *, MemorySpace> _boxes;

  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate,
                                  Value const &value,
                                  OutputFunctor const &out) const
  {
    float length;
    float entrylength;
    int const primitive_index = value.index;
    auto const &ray = ArborX::getGeometry(predicate);
    auto const &box = _boxes(primitive_index);
    int const predicate_index = ArborX::getData(predicate);
    float const kappa = 1.;
    overlapDistance(ray, box, length, entrylength);
    out(IntersectedCell{/*entrylength*/ entrylength,
                        /*optical_path_length*/ kappa * length,
                        /*cell_id*/ primitive_index,
                        /*ray_id*/ predicate_index});
  }
};
} // namespace IntersectsBased

int main(int argc, char *argv[])
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  Kokkos::ScopeGuard guard(argc, argv);

  namespace bpo = boost::program_options;

  int nx;
  int ny;
  int nz;
  int rays_per_box;
  float lx;
  float ly;
  float lz;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help", "help message" )
    ("rays per box", bpo::value<int>(&rays_per_box)->default_value(10), "number of rays")
    ("lx", bpo::value<float>(&lx)->default_value(100.0), "Length of X side")
    ("ly", bpo::value<float>(&ly)->default_value(100.0), "Length of Y side")
    ("lz", bpo::value<float>(&lz)->default_value(100.0), "Length of Z side")
    ("nx", bpo::value<int>(&nx)->default_value(10), "number of X boxes")
    ("ny", bpo::value<int>(&ny)->default_value(10), "number of Y boxes")
    ("nz", bpo::value<int>(&nz)->default_value(10), "number of Z boxes")
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

  int num_boxes = nx * ny * nz;
  float dx = lx / (float)nx;
  float dy = ly / (float)ny;
  float dz = lz / (float)nz;

  ExecutionSpace exec_space{};

  Kokkos::Profiling::pushRegion("Example::problem_setup");
  Kokkos::Profiling::pushRegion("Example::make_grid");
  Kokkos::View<Box *, MemorySpace> boxes(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "Example::boxes"),
      num_boxes);
  Kokkos::parallel_for(
      "Example::initialize_boxes",
      Kokkos::MDRangePolicy(exec_space, {0, 0, 0}, {nx, ny, nz}),
      KOKKOS_LAMBDA(int i, int j, int k) {
        int const box_id = i + nx * j + nx * ny * k;
        boxes(box_id) = {{i * dx, j * dy, k * dz},
                         {(i + 1) * dx, (j + 1) * dy, (k + 1) * dz}};
      });
  Kokkos::Profiling::popRegion();

  // For every box shoot rays from random (uniformly distributed) points inside
  // the box in random (uniformly distributed) directions.
  Kokkos::Profiling::pushRegion("Example::make_rays");
  Kokkos::View<Ray *, MemorySpace> rays(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::rays"),
      rays_per_box * num_boxes);
  {
    using RandPoolType = Kokkos::Random_XorShift64_Pool<>;
    RandPoolType rand_pool(5374857);
    using GeneratorType = RandPoolType::generator_type;

    Kokkos::parallel_for(
        "Example::initialize_rays",
        Kokkos::MDRangePolicy(exec_space, {0, 0}, {num_boxes, rays_per_box}),
        KOKKOS_LAMBDA(size_t const i, size_t const j) {
          // The origins of rays are uniformly distributed in the boxes. The
          // direction vectors are uniformly sampling of a full sphere.
          GeneratorType g = rand_pool.get_state();
          using Kokkos::cos;
          using Kokkos::sin;
          using Kokkos::acos;

          using Vector = ArborX::Details::Vector<3>;

          auto const &b = boxes(i);
          Point origin{b.minCorner()[0] +
                           Kokkos::rand<GeneratorType, float>::draw(g, dx),
                       b.minCorner()[1] +
                           Kokkos::rand<GeneratorType, float>::draw(g, dy),
                       b.minCorner()[2] +
                           Kokkos::rand<GeneratorType, float>::draw(g, dz)};

          float upsilon = Kokkos::rand<GeneratorType, float>::draw(
              g, 2.f * Kokkos::numbers::pi_v<float>);
          float theta =
              acos(1 - 2 * Kokkos::rand<GeneratorType, float>::draw(g));
          Vector direction{cos(upsilon) * sin(theta), sin(upsilon) * sin(theta),
                           cos(theta)};

          rays(j + i * rays_per_box) = Ray{origin, direction};

          rand_pool.free_state(g);
        });
  }
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();

  // Construct BVH
  ArborX::BoundingVolumeHierarchy bvh{
      exec_space, ArborX::Experimental::attach_indices(boxes)};

  // OrderedIntersects-based approach
  Kokkos::View<float *, MemorySpace> energy_ordered_intersects;
  {
    Kokkos::Profiling::pushRegion("Example::ordered_intersects_approach");
    Kokkos::View<float *, MemorySpace> ray_energy(
        Kokkos::view_alloc("Example::ray_energy", Kokkos::WithoutInitializing),
        rays_per_box * num_boxes);
    Kokkos::deep_copy(ray_energy, (total_energy * dx * dy * dz) / rays_per_box);
    energy_ordered_intersects = Kokkos::View<float *, MemorySpace>(
        "Example::energy_ordered_intersects", num_boxes);

    bvh.query(exec_space, OrderedIntersectsBased::Rays<MemorySpace>{rays},
              OrderedIntersectsBased::DepositEnergy<MemorySpace>{
                  boxes, ray_energy, energy_ordered_intersects});
    Kokkos::Profiling::popRegion();
  }

  // Intersects-based approach
  Kokkos::View<float *, MemorySpace> energy_intersects;
  {
    Kokkos::Profiling::pushRegion("Example::intersects_approach");
    Kokkos::View<IntersectsBased::IntersectedCell *> values("Example::values",
                                                            0);
    Kokkos::View<int *> offsets("Example::offsets", 0);
    bvh.query(
        exec_space,
        ArborX::Experimental::attach_indices<int>(
            ArborX::Experimental::make_intersects(rays)),
        IntersectsBased::AccumulateRaySphereIntersections<MemorySpace>{boxes},
        values, offsets);

    Kokkos::View<IntersectsBased::IntersectedCellForSorting *, MemorySpace>
        sort_array(Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                                      "Example::sort_array"),
                   values.size());
    Kokkos::parallel_for(
        "Example::copy sort_array",
        Kokkos::RangePolicy(exec_space, 0, values.size()),
        KOKKOS_LAMBDA(int i) { sort_array(i) = values(i); });
    Kokkos::Profiling::pushRegion("Example::sorting by key");
    // FIXME Users should not need to reach into the Details namespace.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) ||               \
    defined(KOKKOS_ENABLE_SYCL)
    auto permutation = ArborX::Details::sortObjects(exec_space, sort_array);
#else
    auto sort_array_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sort_array);
    Kokkos::View<int *, Kokkos::HostSpace> permutation_host(
        Kokkos::view_alloc("Example::permutation", Kokkos::WithoutInitializing),
        sort_array_host.size());
    std::iota(permutation_host.data(),
              permutation_host.data() + sort_array_host.size(), 0);
    std::sort(permutation_host.data(),
              permutation_host.data() + sort_array_host.size(),
              [&](int const &a, int const &b) {
                return (sort_array_host(a) < sort_array_host(b));
              });
    auto permutation =
        Kokkos::create_mirror_view_and_copy(MemorySpace{}, permutation_host);
#endif
    Kokkos::Profiling::popRegion();

    energy_intersects = Kokkos::View<float *, MemorySpace>(
        "Example::energy_intersects", num_boxes);
    Kokkos::parallel_for(
        "Example::deposit_energy",
        Kokkos::RangePolicy(exec_space, 0, rays_per_box * num_boxes),
        KOKKOS_LAMBDA(int i) {
          float ray_energy = (total_energy * dx * dy * dz) / rays_per_box;
          for (int j = offsets(i); j < offsets(i + 1); ++j)
          {
            const auto &v = values(permutation(j));
            float const energy_deposited =
                lost_energy(ray_energy, v.optical_path_length);
            ray_energy += energy_deposited;
            Kokkos::atomic_add(&energy_intersects(v.cell_id), energy_deposited);
          }
        });
    Kokkos::Profiling::popRegion();
  }

  // Now check that the results we got are the same apart from numerical errors
  // introduced by depositing energy to a particular cell from separate rays
  // in different order.
  int n_errors;
  float rel_tol = 1.e-5;
  Kokkos::parallel_reduce(
      "Example::compare", Kokkos::RangePolicy(exec_space, 0, num_boxes),
      KOKKOS_LAMBDA(int i, int &error) {
        using Kokkos::fabs;
        float const abs_error =
            fabs(energy_ordered_intersects(i) - energy_intersects(i)) /
            fabs(energy_intersects(i));
        if (abs_error > rel_tol * fabs(energy_intersects(i)))
        {
          Kokkos::printf("%d: %f != %f, relative error: %f\n", i,
                         energy_ordered_intersects(i), energy_intersects(i),
                         abs_error / fabs(energy_intersects(i)));
          ++error;
        }
      },
      n_errors);
  std::cerr << "errors = " << n_errors << '\n';
  if (n_errors > 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
