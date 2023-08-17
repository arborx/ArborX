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

constexpr float Lx = 100.0;
constexpr float Ly = 100.0;
constexpr int nx = 101;
constexpr int ny = 101;
constexpr int n = nx * ny;
constexpr float hx = Lx / (nx - 1);
constexpr float hy = Ly / (ny - 1);

// The Mapping class stores the mapping from a unit triangle to a given triangle
// allowing for computing the barycentric coordinates for a given point.
struct Mapping
{
  float alpha[2];
  float beta[2];
  ArborX::ExperimentalHyperGeometry::Point<2> p0;

  // x = a + alpha * (b - a) + beta * (c - a)
  //   = (1-beta-alpha) * a + alpha * b + beta * c
  KOKKOS_FUNCTION
  Mapping(ArborX::ExperimentalHyperGeometry::Triangle<2> const &triangle)
  {
    auto const &a = triangle.a;
    auto const &b = triangle.b;
    auto const &c = triangle.c;

    ArborX::ExperimentalHyperGeometry::Point<2> u = {b[0] - a[0], b[1] - a[1]};
    ArborX::ExperimentalHyperGeometry::Point<2> v = {c[0] - a[0], c[1] - a[1]};

    float const inv_det = 1. / (v[1] * u[0] - v[0] * u[1]);

    alpha[0] = v[1] * inv_det;
    alpha[1] = -v[0] * inv_det;
    beta[0] = -u[1] * inv_det;
    beta[1] = u[0] * inv_det;
    p0 = a;
  }

  KOKKOS_FUNCTION ArborX::Point
  get_coeff(ArborX::ExperimentalHyperGeometry::Point<2> p) const
  {
    float alpha_coeff = alpha[0] * (p[0] - p0[0]) + alpha[1] * (p[1] - p0[1]);
    float beta_coeff = beta[0] * (p[0] - p0[0]) + beta[1] * (p[1] - p0[1]);
    return {1 - alpha_coeff - beta_coeff, alpha_coeff, beta_coeff};
  }

#ifndef NDEBUG
  // Recover the triangle from the mapping. Only used for debugging.
  KOKKOS_FUNCTION ArborX::ExperimentalHyperGeometry::Triangle<2>
  get_triangle() const
  {
    float const inv_det = 1. / (alpha[0] * beta[1] - alpha[1] * beta[0]);
    ArborX::ExperimentalHyperGeometry::Point<2> a = p0;
    ArborX::ExperimentalHyperGeometry::Point<2> b = {
        {p0[0] + inv_det * beta[1], p0[1] - inv_det * beta[0]}};
    ArborX::ExperimentalHyperGeometry::Point<2> c = {
        {p0[0] - inv_det * alpha[1], p0[1] + inv_det * alpha[0]}};
    return {a, b, c};
  }
#endif
};

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
    _points = Kokkos::View<ArborX::ExperimentalHyperGeometry::Point<2> *,
                           MemorySpace>(
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
  Kokkos::View<ArborX::ExperimentalHyperGeometry::Point<2> *, MemorySpace>
      _points;
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
    _triangles = Kokkos::View<ArborX::ExperimentalHyperGeometry::Triangle<2> *,
                              MemorySpace>(
        Kokkos::view_alloc(execution_space, Kokkos::WithoutInitializing,
                           "Example::triangles"),
        2 * n);
    _mappings = Kokkos::View<Mapping *, MemorySpace>(
        Kokkos::view_alloc(execution_space, Kokkos::WithoutInitializing,
                           "Example::mappings"),
        2 * n);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecutionSpace>(
            execution_space, {0, 0}, {nx, ny}),
        KOKKOS_CLASS_LAMBDA(int i, int j) {
          ArborX::ExperimentalHyperGeometry::Point<2> bl{i * hx, j * hy};
          ArborX::ExperimentalHyperGeometry::Point<2> br{(i + 1) * hx, j * hy};
          ArborX::ExperimentalHyperGeometry::Point<2> tl{i * hx, (j + 1) * hy};
          ArborX::ExperimentalHyperGeometry::Point<2> tr{(i + 1) * hx,
                                                         (j + 1) * hy};

          auto index = [](int i, int j) { return i + j * nx; };

          _triangles[2 * index(i, j)] = {tl, bl, br};
          _mappings[2 * index(i, j)] = Mapping(_triangles[2 * index(i, j)]);
          _triangles[2 * index(i, j) + 1] = {tl, br, tr};
          _mappings[2 * index(i, j) + 1] =
              Mapping(_triangles[2 * index(i, j) + 1]);
        });

#ifndef NDEBUG
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, 2 * n),
        KOKKOS_CLASS_LAMBDA(int k) {
          ArborX::ExperimentalHyperGeometry::Triangle<2> recover_triangle =
              _mappings[k].get_triangle();

          constexpr float eps = 1.e-3;

          for (unsigned int i = 0; i < 2; ++i)
            if (Kokkos::abs(_triangles[k].a[i] - recover_triangle.a[i]) > eps)
              Kokkos::abort(
                  "Mismatch for first point when recovering triangle");

          for (unsigned int i = 0; i < 2; ++i)
            if (Kokkos::abs(_triangles[k].b[i] - recover_triangle.b[i]) > eps)
              Kokkos::abort(
                  "Mismatch for second point when recovering triangle");

          for (unsigned int i = 0; i < 2; ++i)
            if (Kokkos::abs(_triangles[k].c[i] - recover_triangle.c[i]) > eps)
              Kokkos::abort(
                  "Mismatch for third point when recoverning triangle");

          auto const &coeff_a = _mappings[k].get_coeff(_triangles[k].a);
          if ((Kokkos::abs(coeff_a[0] - 1.) > eps) ||
              Kokkos::abs(coeff_a[1]) > eps || Kokkos::abs(coeff_a[2]) > eps)
            Kokkos::abort(
                "Mismatch for coefficients of first point in triangle");

          auto const &coeff_b = _mappings[k].get_coeff(_triangles[k].b);
          if ((Kokkos::abs(coeff_b[0]) > eps) ||
              Kokkos::abs(coeff_b[1] - 1.) > eps ||
              Kokkos::abs(coeff_b[2]) > eps)
            Kokkos::abort(
                "Mismatch for coefficients of first point in triangle");

          auto const &coeff_c = _mappings[k].get_coeff(_triangles[k].c);
          if ((Kokkos::abs(coeff_c[0]) > eps) ||
              Kokkos::abs(coeff_c[1]) > eps ||
              Kokkos::abs(coeff_c[2] - 1.) > eps)
            Kokkos::abort(
                "Mismatch for coefficients of first point in triangle");
        });
#endif
  }

  KOKKOS_FUNCTION int size() const { return _triangles.size(); }

  KOKKOS_FUNCTION ArborX::ExperimentalHyperGeometry::Triangle<2> const &
  operator()(int i) const
  {
    return _triangles(i);
  }

  KOKKOS_FUNCTION Mapping const &get_mapping(int i) const
  {
    return _mappings(i);
  }

private:
  Kokkos::View<ArborX::ExperimentalHyperGeometry::Triangle<2> *, MemorySpace>
      _triangles;
  Kokkos::View<Mapping *, MemorySpace> _mappings;
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
  // actuallt inside the wrapped triangle and we have to check that here. This
  // also gives us the opportunity to store the barycentric coordinates in case
  // there is an intersection. Since the triangles don't overlap in this
  // example, there is at most one triangle that contains a given point and we
  // can abort the search early when we found a match.
  template <typename Query, typename Primitive>
  KOKKOS_FUNCTION auto operator()(Query const &query,
                                  Primitive const &primitive) const
  {
    ArborX::ExperimentalHyperGeometry::Point<2> const &point =
        getGeometry(getPredicate(query));
    auto query_index = ArborX::getData(query);

    auto const coeffs =
        _triangles.get_mapping(primitive.index).get_coeff(point);
    bool intersects = coeffs[0] >= 0 && coeffs[1] >= 0 && coeffs[2] >= 0;

    if (intersects)
    {
      _offsets(query_index) = primitive.index;
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

  std::cout << "Create grid with triangles.\n";
  Triangles<MemorySpace> triangles(execution_space);
  std::cout << "Triangles set up.\n";

  std::cout << "Creating BVH tree.\n";
  ArborX::BasicBoundingVolumeHierarchy<
      MemorySpace, ArborX::Details::PairIndexVolume<
                       ArborX::ExperimentalHyperGeometry::Box<2>>> const
      tree(execution_space, triangles);
  std::cout << "BVH tree set up.\n";

  std::cout << "Create the points used for queries.\n";
  Points<MemorySpace> points(execution_space);
  std::cout << "Points for queries set up.\n";

  std::cout << "Starting the queries.\n";
  int const n = points.size();
  Kokkos::View<int *, MemorySpace> offsets("Example::offsets", n);
  Kokkos::View<ArborX::Point *, MemorySpace> coefficients(
      "Example::coefficients", n);

  tree.query(execution_space, points,
             TriangleIntersectionCallback<MemorySpace>{triangles, offsets,
                                                       coefficients});
  std::cout << "Queries done.\n";

#ifndef NDEBUG
  std::cout << "Starting checking results.\n";
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, n),
      KOKKOS_LAMBDA(int i) {
        constexpr float eps = 1.e-3;

        if (offsets(i) != i)
          Kokkos::abort("Offsets are wrong");
        auto const &c = coefficients(i);
        auto const &t = triangles(offsets(i));
        auto const &p_h = points(i);
        auto const p = ArborX::ExperimentalHyperGeometry::Point<2>{
            c[0] * t.a[0] + c[1] * t.b[0] + c[2] * t.c[0],
            c[0] * t.a[1] + c[1] * t.b[1] + c[2] * t.c[1]};
        if ((Kokkos::abs(p[0] - p_h[0]) > eps) ||
            Kokkos::abs(p[1] - p_h[1]) > eps)
          Kokkos::abort("Coefficients are wrong");
      });
  std::cout << "Checking results successful.\n";
#endif
  execution_space.fence();
}
