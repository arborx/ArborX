/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX.hpp>
#include <ArborX_HyperTriangle.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>

// Perform intersection queries using 2D triangles on a regular mesh as
// primitives and intersection with points as queries. One point per triangle.
// __________
// |\x|\x|\x|
// |x\|x\|x\|
// __________
// |\x|\x|\x|
// |x\|x\|x\|
// __________
// |\x|\x|\x|
// |x\|x\|x\|
// __________

using Point = ArborX::ExperimentalHyperGeometry::Point<2>;
using Triangle = ArborX::ExperimentalHyperGeometry::Triangle<2>;

template <typename ExecutionSpace, typename MemorySpace>
void buildPoints(ExecutionSpace const &space, float Lx, float Ly, int nx,
                 int ny, Kokkos::View<Point *, MemorySpace> &points)
{
  int n = nx * ny;
  float hx = Lx / (nx - 1);
  float hy = Ly / (ny - 1);

  KokkosExt::reallocWithoutInitializing(space, points, 2 * n);
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecutionSpace>(space, {0, 0},
                                                             {nx, ny}),
      KOKKOS_LAMBDA(int i, int j) {
        auto index = [nx](int i, int j) { return i + j * nx; };

        points[2 * index(i, j) + 0] = {(i + .25f) * hx, (j + .25f) * hy};
        points[2 * index(i, j) + 1] = {(i + .75f) * hx, (j + .75f) * hy};
      });
}

template <typename ExecutionSpace, typename MemorySpace>
void buildTriangles(ExecutionSpace const &space, float Lx, float Ly, int nx,
                    int ny, Kokkos::View<Triangle *, MemorySpace> &triangles)
{
  int n = nx * ny;
  float hx = Lx / (nx - 1);
  float hy = Ly / (ny - 1);

  KokkosExt::reallocWithoutInitializing(space, triangles, 2 * n);

  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecutionSpace>(space, {0, 0},
                                                             {nx, ny}),
      KOKKOS_LAMBDA(int i, int j) {
        Point bottom_left{i * hx, j * hy};
        Point bottom_right{(i + 1) * hx, j * hy};
        Point top_left{i * hx, (j + 1) * hy};
        Point top_right{(i + 1) * hx, (j + 1) * hy};

        auto index = [nx](int i, int j) { return i + j * nx; };

        triangles[2 * index(i, j) + 0] = {top_left, bottom_left, bottom_right};
        triangles[2 * index(i, j) + 1] = {top_left, bottom_right, top_right};
      });
}

// Store the points that represent the queries
template <typename MemorySpace>
struct Points
{
  Kokkos::View<Point *, MemorySpace> _points;
};

template <typename MemorySpace>
struct ArborX::AccessTraits<Points<MemorySpace>, ArborX::PredicatesTag>
{
  using memory_space = MemorySpace;
  static KOKKOS_FUNCTION auto size(Points<MemorySpace> const &points)
  {
    return points._points.size();
  }
  static KOKKOS_FUNCTION auto get(Points<MemorySpace> const &points, int i)
  {
    return ArborX::attach(ArborX::intersects(points._points(i)), i);
  }
};

template <typename MemorySpace>
struct TriangleIntersectionCallback
{
  // Store the barycentric coordinates in case there is an intersection. Since
  // the triangles don't overlap in this example, there is at most one triangle
  // that contains a given point and we can abort the search early when we
  // found a match.
  template <typename Query, typename Value>
  KOKKOS_FUNCTION auto operator()(Query const &query, Value const &value) const
  {
    auto const &triangle = value.value;
    auto triangle_index = value.index;

    static_assert(ArborX::GeometryTraits::is_triangle<
                  std::decay_t<decltype(triangle)>>::value);

    Point const &point = getGeometry(getPredicate(query));
    auto query_index = ArborX::getData(query);

    auto const &a = triangle.a;
    auto const &b = triangle.b;
    auto const &c = triangle.c;

    // Find coefficients alpha and beta such that
    // x = a + alpha * (b - a) + beta * (c - a)
    //   = (1 - alpha - beta) * a + alpha * b + beta * c
    // recognizing the linear system
    // ((b - a) (c - a)) (alpha beta)^T = (x - a)
    float u[2] = {b[0] - a[0], b[1] - a[1]};
    float v[2] = {c[0] - a[0], c[1] - a[1]};
    float const det = v[1] * u[0] - v[0] * u[1];
    if (det == 0)
      Kokkos::abort("Degenerate triangles are not supported!");
    float const inv_det = 1.f / det;

    float alpha[2] = {v[1] * inv_det, -v[0] * inv_det};
    float beta[2] = {-u[1] * inv_det, u[0] * inv_det};

    float alpha_coeff =
        alpha[0] * (point[0] - a[0]) + alpha[1] * (point[1] - a[1]);
    float beta_coeff =
        beta[0] * (point[0] - a[0]) + beta[1] * (point[1] - a[1]);

    Kokkos::Array<float, 3> coeffs = {1 - alpha_coeff - beta_coeff, alpha_coeff,
                                      beta_coeff};

    _offsets(query_index) = triangle_index;
    _coefficients(query_index) = coeffs;

    return ArborX::CallbackTreeTraversalControl::early_exit;
  }

  Kokkos::View<int *, MemorySpace> _offsets;
  Kokkos::View<Kokkos::Array<float, 3> *, MemorySpace> _coefficients;
};

// Now that we have encapsulated the objects and queries to be used within the
// Triangles class, we can continue with performing the actual search.
int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;
  ExecutionSpace space;

  constexpr float Lx = 50.f;
  constexpr float Ly = 50.f;
  constexpr int nx = 101;
  constexpr int ny = 101;

  // Create grid with triangles
  Kokkos::View<Point *, MemorySpace> points("Example::points", 0);
  buildPoints(space, Lx, Ly, nx, ny, points);

  int const num_points = points.size();

  Kokkos::View<Triangle *, MemorySpace> triangles("Example::triangles", 0);
  buildTriangles(space, Lx, Ly, nx, ny, triangles);

  // Create BVH tree
  ArborX::BoundingVolumeHierarchy<MemorySpace,
                                  ArborX::PairValueIndex<Triangle>> const
      tree(space, ArborX::Experimental::attach_indices(triangles));

  // Execute the queries
  Kokkos::View<int *, MemorySpace> offsets("Example::offsets", num_points);
  Kokkos::View<Kokkos::Array<float, 3> *, MemorySpace> coefficients(
      "Example::coefficients", num_points);

  tree.query(space, Points<MemorySpace>{points},
             TriangleIntersectionCallback<MemorySpace>{offsets, coefficients});

  // Check the results
  bool success;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_points),
      KOKKOS_LAMBDA(int i, bool &update) {
        constexpr float eps = 1.e-3;

#if KOKKOS_VERSION >= 40200
        using Kokkos::printf;
#elif defined(__SYCL_DEVICE_ONLY__)
        using sycl::ext::oneapi::experimental::printf;
#endif
        if (offsets(i) != i)
        {
          printf("Offsets are wrong for query %d.\n", i);
          update = false;
        }
        auto const &c = coefficients(i);
        auto const &t = triangles(offsets(i));
        auto const &p_ref = points(i);
        Point p{c[0] * t.a[0] + c[1] * t.b[0] + c[2] * t.c[0],
                c[0] * t.a[1] + c[1] * t.b[1] + c[2] * t.c[1]};
        if (Kokkos::abs(p[0] - p_ref[0]) > eps ||
            Kokkos::abs(p[1] - p_ref[1]) > eps)
        {
          printf("Coefficients are wrong for query %d.\n", i);
          update = false;
        }
      },
      Kokkos::LAnd<bool, Kokkos::HostSpace>(success));
  std::cout << "Check " << (success ? "succeeded" : "failed") << std::endl;

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
