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

constexpr float Lx = 50.f;
constexpr float Ly = 50.f;
constexpr int nx = 101;
constexpr int ny = 101;
constexpr int n = nx * ny;
constexpr float hx = Lx / (nx - 1);
constexpr float hy = Ly / (ny - 1);

using Point = ArborX::ExperimentalHyperGeometry::Point<2>;
using Box = ArborX::ExperimentalHyperGeometry::Box<2>;
using Triangle = ArborX::ExperimentalHyperGeometry::Triangle<2>;

#ifdef PRECOMPUTE_MAPPING
// The Mapping class stores the mapping from a unit triangle to a given triangle
// allowing for computing the barycentric coordinates for a given point.
struct Mapping
{
  float alpha[2];
  float beta[2];
  Point p0;

  // x = a + alpha * (b - a) + beta * (c - a)
  //   = (1-beta-alpha) * a + alpha * b + beta * c
  KOKKOS_FUNCTION
  Mapping(Triangle const &triangle)
  {
    auto const &a = triangle.a;
    auto const &b = triangle.b;
    auto const &c = triangle.c;

    Point u = {b[0] - a[0], b[1] - a[1]};
    Point v = {c[0] - a[0], c[1] - a[1]};

    float const det = v[1] * u[0] - v[0] * u[1];
    if (det == 0)
      Kokkos::abort("Degenerate triangles are not supported!");
    float const inv_det = 1.f / det;

    alpha[0] = v[1] * inv_det;
    alpha[1] = -v[0] * inv_det;
    beta[0] = -u[1] * inv_det;
    beta[1] = u[0] * inv_det;
    p0 = a;
  }

  KOKKOS_FUNCTION ArborX::Point get_barycentric_coordinates(Point p) const
  {
    float alpha_coeff = alpha[0] * (p[0] - p0[0]) + alpha[1] * (p[1] - p0[1]);
    float beta_coeff = beta[0] * (p[0] - p0[0]) + beta[1] * (p[1] - p0[1]);
    return {1 - alpha_coeff - beta_coeff, alpha_coeff, beta_coeff};
  }
};
#endif

// Store the points that represent the queries.
template <typename MemorySpace>
class Points
{
public:
  template <typename ExecutionSpace>
  Points(ExecutionSpace const &execution_space)
  {
    initialize(execution_space);
  }

  template <typename ExecutionSpace>
  void initialize(ExecutionSpace const &execution_space)
  {
    _points = Kokkos::View<Point *, MemorySpace>(
        Kokkos::view_alloc(execution_space, Kokkos::WithoutInitializing,
                           "Example::points"),
        2 * n);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecutionSpace>(
            execution_space, {0, 0}, {nx, ny}),
        KOKKOS_CLASS_LAMBDA(int i, int j) {
          auto index = [](int i, int j) { return i + j * nx; };

          _points[2 * index(i, j)] = {(i + .25f) * hx, (j + .25f) * hy};
          _points[2 * index(i, j) + 1] = {(i + .75f) * hx, (j + .75f) * hy};
        });
  }

  KOKKOS_FUNCTION auto operator()(int i) const { return _points(i); }

  KOKKOS_FUNCTION auto size() const { return _points.size(); }

private:
  Kokkos::View<Point *, MemorySpace> _points;
};

template <typename MemorySpace>
class Triangles
{
public:
  template <typename ExecutionSpace>
  Triangles(ExecutionSpace const &execution_space)
  {
    initialize(execution_space);
  }

  // Create non-intersecting triangles on a 2D cartesian grid
  // used for the primitives in the tree construction, and compute and store the
  // mappings used in the queries.
  template <typename ExecutionSpace>
  void initialize(ExecutionSpace const &execution_space)
  {
    _triangles = Kokkos::View<Triangle *, MemorySpace>(
        Kokkos::view_alloc(execution_space, Kokkos::WithoutInitializing,
                           "Example::triangles"),
        2 * n);
#ifdef PRECOMPUTE_MAPPING
    _mappings = Kokkos::View<Mapping *, MemorySpace>(
        Kokkos::view_alloc(execution_space, Kokkos::WithoutInitializing,
                           "Example::mappings"),
        2 * n);
#endif

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecutionSpace>(
            execution_space, {0, 0}, {nx, ny}),
        KOKKOS_CLASS_LAMBDA(int i, int j) {
          Point bl{i * hx, j * hy};
          Point br{(i + 1) * hx, j * hy};
          Point tl{i * hx, (j + 1) * hy};
          Point tr{(i + 1) * hx, (j + 1) * hy};

          auto index = [](int i, int j) { return i + j * nx; };

          _triangles[2 * index(i, j)] = {tl, bl, br};
          _triangles[2 * index(i, j) + 1] = {tl, br, tr};
#ifdef PRECOMPUTE_MAPPING
          _mappings[2 * index(i, j)] = Mapping(_triangles[2 * index(i, j)]);
          _mappings[2 * index(i, j) + 1] =
              Mapping(_triangles[2 * index(i, j) + 1]);
#endif
        });
  }

  KOKKOS_FUNCTION int size() const { return _triangles.size(); }

  KOKKOS_FUNCTION Triangle const &operator()(int i) const
  {
    return _triangles(i);
  }

#ifdef PRECOMPUTE_MAPPING
  KOKKOS_FUNCTION Mapping const &get_mapping(int i) const
  {
    return _mappings(i);
  }
#endif

private:
  Kokkos::View<Triangle *, MemorySpace> _triangles;
#ifdef PRECOMPUTE_MAPPING
  Kokkos::View<Mapping *, MemorySpace> _mappings;
#endif
};

// For creating the bounding volume hierarchy given a Triangles object, we
// need to define the memory space, how to get the total number of objects,
// and how to access a specific box. Since there are corresponding functions in
// the Triangles class, we just resort to them.
template <typename MemorySpace>
struct ArborX::AccessTraits<Triangles<MemorySpace>, ArborX::PrimitivesTag>
{
  using memory_space = MemorySpace;

  static KOKKOS_FUNCTION int size(Triangles<MemorySpace> const &triangles)
  {
    return triangles.size();
  }

  static KOKKOS_FUNCTION auto get(Triangles<MemorySpace> const &triangles,
                                  int i)
  {
    auto const &triangle = triangles(i);
    ArborX::ExperimentalHyperGeometry::Box<2> box{};
    box += triangle.a;
    box += triangle.b;
    box += triangle.c;
    return box;
  }
};

template <typename MemorySpace>
struct ArborX::AccessTraits<Points<MemorySpace>, ArborX::PredicatesTag>
{
  using memory_space = MemorySpace;
  static KOKKOS_FUNCTION int size(Points<MemorySpace> const &points)
  {
    return points.size();
  }
  static KOKKOS_FUNCTION auto get(Points<MemorySpace> const &points, int i)
  {
    return ArborX::attach(ArborX::intersects(points(i)), i);
  }
};

template <typename MemorySpace>
class TriangleIntersectionCallback
{
public:
  TriangleIntersectionCallback(
      Triangles<MemorySpace> triangles,
      Kokkos::View<int *, MemorySpace> offsets,
      Kokkos::View<ArborX::Point *, MemorySpace> coefficients)
      : _triangles(triangles)
      , _offsets(offsets)
      , _coefficients(coefficients)
  {}

  // The search tree consists entirely of boxes, although the primitives are
  // triangles. Thus, a detected collision doesn't mean that the point is
  // actually inside the wrapped triangle and we have to check that here. This
  // also gives us the opportunity to store the barycentric coordinates in case
  // there is an intersection. Since the triangles don't overlap in this
  // example, there is at most one triangle that contains a given point and we
  // can abort the search early when we found a match.
  template <typename Query, typename Primitive>
  KOKKOS_FUNCTION auto operator()(Query const &query,
                                  Primitive const &primitive) const
  {
    Point const &point = getGeometry(getPredicate(query));
    auto query_index = ArborX::getData(query);
    auto triangle_index = primitive.index;

#ifdef PRECOMPUTE_MAPPING
    auto const coeffs = _triangles.get_mapping(triangle_index)
                            .get_barycentric_coordinates(point);
#else
    Triangle const &triangle = _triangles(triangle_index);
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

    ArborX::Point coeffs = {1 - alpha_coeff - beta_coeff, alpha_coeff,
                            beta_coeff};
#endif
    bool intersects = coeffs[0] >= 0 && coeffs[1] >= 0 && coeffs[2] >= 0;

    if (intersects)
    {
      _offsets(query_index) = triangle_index;
      _coefficients(query_index) = coeffs;
      return ArborX::CallbackTreeTraversalControl::early_exit;
    }
    return ArborX::CallbackTreeTraversalControl::normal_continuation;
  }

private:
  Triangles<MemorySpace> _triangles;
  Kokkos::View<int *, MemorySpace> _offsets;
  Kokkos::View<ArborX::Point *, MemorySpace> _coefficients;
};

// Now that we have encapsulated the objects and queries to be used within the
// Triangles class, we can continue with performing the actual search.
int main()
{
  Kokkos::ScopeGuard guard;

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;
  ExecutionSpace execution_space;

  // Create grid with triangles
  Triangles<MemorySpace> triangles(execution_space);

  // Create BVH tree
  ArborX::BoundingVolumeHierarchy<MemorySpace,
                                  ArborX::Details::PairIndexVolume<Box>> const
      tree(execution_space,
           ArborX::Details::LegacyValues<decltype(triangles), Box>{triangles});

  // Create the points used for queries
  Points<MemorySpace> points(execution_space);

  // Execute the queries
  int const n = points.size();
  Kokkos::View<int *, MemorySpace> offsets("Example::offsets", n);
  Kokkos::View<ArborX::Point *, MemorySpace> coefficients(
      "Example::coefficients", n);

  tree.query(execution_space, points,
             TriangleIntersectionCallback<MemorySpace>{triangles, offsets,
                                                       coefficients});

  // Check the results
  bool success;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, n),
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
        if ((Kokkos::abs(p[0] - p_ref[0]) > eps) ||
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
