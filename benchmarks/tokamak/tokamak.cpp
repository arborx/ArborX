/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
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

#include <fstream>
#include <iostream>

using Box = ArborX::ExperimentalHyperGeometry::Box<2>;
using Point = ArborX::ExperimentalHyperGeometry::Point<2>;
using Triangle = ArborX::ExperimentalHyperGeometry::Triangle<2>;

template <typename MemorySpace>
struct Triangles
{
  // Return the number of triangles.
  KOKKOS_FUNCTION int size() const { return _triangles.size(); }

  // Return the triangle with index i.
  KOKKOS_FUNCTION Triangle const &operator()(int i) const
  {
    return _triangles(i);
  }

  Kokkos::View<ArborX::ExperimentalHyperGeometry::Triangle<2> *, MemorySpace>
      _triangles;
};

// For creating the bounding volume hierarchy given a Triangles object, we
// need to define the memory space, how to get the total number of objects,
// and how to access a specific box. Since there are corresponding functions in
// the Triangles class, we just resort to them.
template <typename DeviceType>
struct ArborX::AccessTraits<Triangles<DeviceType>, ArborX::PrimitivesTag>
{
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION int size(Triangles<DeviceType> const &triangles)
  {
    return triangles.size();
  }
  static KOKKOS_FUNCTION auto get(Triangles<DeviceType> const &triangles, int i)
  {
    auto const &triangle = triangles(i);
    Box box{};
    box += triangle.a;
    box += triangle.b;
    box += triangle.c;
    return box;
  }
};

KOKKOS_FUNCTION
ArborX::Point compute_barycentric_coordinates(Triangle const &triangle,
                                              Point const &point)
{
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
  float beta_coeff = beta[0] * (point[0] - a[0]) + beta[1] * (point[1] - a[1]);

  return {1 - alpha_coeff - beta_coeff, alpha_coeff, beta_coeff};
}

template <typename DeviceType>
class TriangleIntersectionCallback
{
public:
  KOKKOS_FUNCTION TriangleIntersectionCallback(
      Kokkos::View<Triangle *, typename DeviceType::memory_space> triangles)
      : _triangles(triangles)
  {}

  template <typename Query, typename Primitive>
  KOKKOS_FUNCTION auto operator()(Query const &query,
                                  Primitive const &primitive) const
  {
    Point const &point = getGeometry(getPredicate(query));
    auto const &attachment = ArborX::getData(query);
    auto triangle_index = primitive.index;
    Triangle const &triangle = _triangles(triangle_index);

    ArborX::Point coeffs = compute_barycentric_coordinates(triangle, point);

    bool intersects = coeffs[0] >= 0 && coeffs[1] >= 0 && coeffs[2] >= 0;

    if (intersects)
    {
      attachment.triangle_index = triangle_index;
      attachment.coeffs = coeffs;
      return ArborX::CallbackTreeTraversalControl::early_exit;
    }
    return ArborX::CallbackTreeTraversalControl::normal_continuation;
  }

private:
  Kokkos::View<Triangle *, typename DeviceType::memory_space> _triangles;
};

template <typename DeviceType>
Kokkos::View<Triangle *, typename DeviceType::memory_space>
parse_stl(typename DeviceType::execution_space const &execution_space)
{
  std::vector<Triangle> triangles_host;
  std::ifstream stl_file("RZGrid.stl");
  if (!stl_file.good())
    throw std::runtime_error("Cannot open file");
  std::string line;
  std::istringstream in;
  Triangle triangle;
  std::string dummy;
  while (std::getline(stl_file >> std::ws, line))
  {
    if (line.find("outer loop") == std::string::npos)
      continue;

    std::getline(stl_file >> std::ws, line);
    in.str(line);
    in >> dummy >> triangle.a[0] >> triangle.a[1];

    std::getline(stl_file >> std::ws, line);
    in.str(line);
    in >> dummy >> triangle.b[0] >> triangle.b[1];

    std::getline(stl_file >> std::ws, line);
    in.str(line);
    in >> dummy >> triangle.c[0] >> triangle.c[1];

    if (triangles_host.size() == 0)
    {
      std::cout << triangle.a[0] << ' ' << triangle.a[1] << '\n'
                << triangle.b[0] << ' ' << triangle.b[1] << '\n'
                << triangle.c[0] << ' ' << triangle.c[1] << '\n';
    }

    triangles_host.push_back(triangle);
  }

  std::cout << "Read " << triangles_host.size() << " Triangles\n";

  Kokkos::View<Triangle *, typename DeviceType::memory_space> triangles(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "triangles"),
      triangles_host.size());
  Kokkos::deep_copy(execution_space, triangles,
                    Kokkos::View<Triangle *, Kokkos::HostSpace>(
                        triangles_host.data(), triangles_host.size()));

  return {triangles};
}

template <typename DeviceType>
Kokkos::View<Point **, Kokkos::LayoutLeft, typename DeviceType::memory_space>
parse_points(typename DeviceType::execution_space const &execution_space)
{
  std::vector<Point> points_host;
  std::ifstream step_file("RK4Steps.txt");
  if (!step_file.good())
    throw std::runtime_error("Cannot open file");
  Point point;
  int id = 0;
  int size_per_id = -1;
  std::string line;
  while (std::getline(step_file, line))
  {
    std::stringstream in_line, in_word;
    std::string word;
    in_line.str(line);
    int old_id = id;
    for (unsigned int i = 0; i < 3; ++i)
    {
      std::getline(in_line, word, ',');
      in_word << word;
    }
    in_word >> id >> point[0] >> point[1];

    if (old_id != id)
    {
      if (size_per_id == -1)
        size_per_id = points_host.size();
      else if (points_host.size() % size_per_id != 0)
      {
        std::cout << points_host.size() << ' ' << size_per_id << std::endl;
        Kokkos::abort("different sizes!");
      }
    }
    points_host.push_back(point);
    if (points_host.size() < 10)
      std::cout << id << ' ' << point[0] << ' ' << point[1] << std::endl;
  }

  std::cout << "Read " << points_host.size() / size_per_id << " ids which "
            << size_per_id << " elements each.\n";
  Kokkos::View<Point **, Kokkos::LayoutRight, Kokkos::HostSpace>
      points_host_view(points_host.data(), points_host.size() / size_per_id,
                       size_per_id);
  std::cout << "id 0, point 1: " << points_host_view(0, 1)[0] << ' '
            << points_host_view(0, 1)[1] << std::endl;
  std::cout << "id 1, point 0: " << points_host_view(1, 0)[0] << ' '
            << points_host_view(1, 0)[1] << std::endl;

  Kokkos::View<Point **, Kokkos::LayoutLeft, typename DeviceType::memory_space>
      points(Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"),
             points_host.size() / size_per_id, size_per_id);
  auto points_tmp_view = Kokkos::create_mirror_view_and_copy(
      typename DeviceType::memory_space{}, points_host_view);
  Kokkos::deep_copy(execution_space, points, points_tmp_view);

  return points;
}

struct Dummy
{};

// Now that we have encapsulated the objects and queries to be used within the
// Triangles class, we can continue with performing the actual search.
int main()
{
  Kokkos::initialize();
  {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    ExecutionSpace execution_space;

    std::cout << "Create grid with triangles.\n";
    auto triangles = parse_stl<DeviceType>(execution_space);
    auto points = parse_points<DeviceType>(execution_space);

    std::cout << "Creating BVH tree.\n";
    ArborX::BasicBoundingVolumeHierarchy<
        MemorySpace, ArborX::Details::PairIndexVolume<Triangle>> const
        tree(execution_space,
             ArborX::Details::LegacyValues<decltype(triangles), Triangle>{
                 triangles});
    std::cout << "BVH tree set up.\n";

    std::cout << "Starting the queries.\n";
    int const n = points.extent(0);

    struct Attachment
    {
      int &triangle_index;
      ArborX::Point &coeffs;
    };

    std::cout << "n: " << n << std::endl;

    Kokkos::parallel_for(
        "ArborX::TreeTraversal::spatial",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, n),
        KOKKOS_LAMBDA(int i) {
          int triangle_index = 0;
          ArborX::Point coefficients{};
          for (unsigned int j = 0; j < points.extent(1); ++j)
          {
            auto const &point = points(i, j);
            auto const &triangle = triangles(triangle_index);
            auto const &test_coeffs =
                compute_barycentric_coordinates(triangle, point);
            bool intersects = test_coeffs[0] >= 0 && test_coeffs[1] >= 0 &&
                              test_coeffs[2] >= 0;
            if (intersects)
            {
              coefficients = test_coeffs;
              KOKKOS_IMPL_DO_NOT_USE_PRINTF(
                  "%d, %d: %f %f in %d (same), coefficients: %f, %f, %f\n", i,
                  j, point[0], point[1], triangle_index, coefficients[0],
                  coefficients[1], coefficients[2]);
            }
            else
            {
              tree.query(
                  ArborX::Experimental::PerThread{},
                  ArborX::attach(ArborX::intersects(point),
                                 Attachment{triangle_index, coefficients}),
                  TriangleIntersectionCallback<DeviceType>{triangles});
              KOKKOS_IMPL_DO_NOT_USE_PRINTF(
                  "%d, %d: %f %f in %d, coefficients: %f, %f, %f\n", i, j,
                  point[0], point[1], triangle_index, coefficients[0],
                  coefficients[1], coefficients[2]);
            }
          }
        });

    std::cout << "Queries done.\n";
  }

  Kokkos::finalize();
}
