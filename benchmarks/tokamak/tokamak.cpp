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

#include <Kokkos_Core.hpp>

#include <fstream>

template <int dim>
struct Triangle
{
  ArborX::ExperimentalHyperGeometry::Point<dim> a, b, c;
};

struct Mapping
{
  ArborX::ExperimentalHyperGeometry::Point<2> alpha;
  ArborX::ExperimentalHyperGeometry::Point<2> beta;
  ArborX::ExperimentalHyperGeometry::Point<2> p0;

  ArborX::Point get_coeff(ArborX::ExperimentalHyperGeometry::Point<2> p) const
  {
    float alpha_coeff = alpha[0] * (p[0] - p0[0]) + alpha[1] * (p[1] - p0[1]);
    float beta_coeff = beta[0] * (p[0] - p0[0]) + beta[1] * (p[1] - p0[1]);
    return {1 - alpha_coeff - beta_coeff, alpha_coeff, beta_coeff};
  }

  // x = a + alpha * (b - a) + beta * (c - a)
  //   = (1-beta-alpha) * a + alpha * b + beta * c
  void compute(Triangle<2> const &triangle)
  {
    auto const &a = triangle.a;
    auto const &b = triangle.b;
    auto const &c = triangle.c;

    ArborX::ExperimentalHyperGeometry::Point<2> u = {b[0] - a[0], b[1] - a[1]};
    ArborX::ExperimentalHyperGeometry::Point<2> v = {c[0] - a[0], c[1] - a[1]};

    float const inv_det = 1. / (v[1] * u[0] - v[0] * u[1]);

    alpha = ArborX::ExperimentalHyperGeometry::Point<2>{v[1] * inv_det,
                                                        -v[0] * inv_det};
    beta = ArborX::ExperimentalHyperGeometry::Point<2>{-u[1] * inv_det,
                                                       u[0] * inv_det};
    p0 = a;
  }

  Triangle<2> get_triangle() const
  {
    float const inv_det = 1. / (alpha[0] * beta[1] - alpha[1] * beta[0]);
    ArborX::ExperimentalHyperGeometry::Point<2> a = p0;
    ArborX::ExperimentalHyperGeometry::Point<2> b = {
        {p0[0] + inv_det * beta[1], p0[1] - inv_det * beta[0]}};
    ArborX::ExperimentalHyperGeometry::Point<2> c = {
        {p0[0] - inv_det * alpha[1], p0[1] + inv_det * alpha[0]}};
    return {a, b, c};
  }
};

template <typename DeviceType>
struct Triangles
{
  // Return the number of triangles.
  KOKKOS_FUNCTION int size() const { return triangles_.size(); }

  // Return the triangle with index i.
  KOKKOS_FUNCTION Triangle<2> const &get_triangle(int i) const
  {
    return triangles_(i);
  }

  KOKKOS_FUNCTION Mapping const &get_mapping(int i) const
  {
    return mappings_(i);
  }

  Kokkos::View<Triangle<2> *, typename DeviceType::memory_space> triangles_;
  Kokkos::View<Mapping *, typename DeviceType::memory_space> mappings_;
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
    auto const &triangle = triangles.get_triangle(i);
    ArborX::ExperimentalHyperGeometry::Box<2> box{};
    box += triangle.a;
    box += triangle.b;
    box += triangle.c;
    return box;
  }
};

template <typename DeviceType>
class TriangleIntersectionCallback
{
public:
  TriangleIntersectionCallback(Triangles<DeviceType> triangles)
      : triangles_(triangles)
  {}

  template <typename Query>
  KOKKOS_FUNCTION void operator()(
      Query const &query,
      ArborX::Details::PairIndexVolume<
          ArborX::ExperimentalHyperGeometry::Box<2>> const &predicate) const
  {
    ArborX::ExperimentalHyperGeometry::Point<2> const &point =
        getGeometry(getPredicate(query));
    auto const &attachment = ArborX::getData(query);

    auto const &triangle = triangles_.get_triangle(predicate.index);

    auto const coeffs =
        triangles_.get_mapping(predicate.index).get_coeff(point);
    bool intersects = coeffs[0] >= 0 && coeffs[1] >= 0 && coeffs[2] >= 0;

    if (intersects)
    {
      attachment.triangle_index = predicate.index;
      attachment.coeffs = coeffs;
    }
  }

private:
  Triangles<DeviceType> triangles_;
};

template <typename DeviceType>
Triangles<DeviceType>
parse_stl(typename DeviceType::execution_space const &execution_space)
{
  std::vector<Triangle<2>> triangles_host;
  std::vector<Mapping> mappings_host;
  std::ifstream stl_file("RZGrid.stl");
  if (!stl_file.good())
    throw std::runtime_error("Cannot open file");
  std::string line;
  std::istringstream in;
  Mapping mapping;
  Triangle<2> triangle;
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

    mapping.compute(triangle);

    if (triangles_host.size() == 0)
    {
      std::cout << triangle.a[0] << ' ' << triangle.a[1] << '\n'
                << triangle.b[0] << ' ' << triangle.b[1] << '\n'
                << triangle.c[0] << ' ' << triangle.c[1] << '\n';
      std::cout << mapping.alpha[0] << ' ' << mapping.alpha[1] << '\n'
                << mapping.beta[0] << ' ' << mapping.beta[1] << '\n'
                << mapping.p0[0] << ' ' << mapping.p0[1] << '\n';
    }

    triangles_host.push_back(triangle);
    mappings_host.push_back(mapping);
  }

  std::cout << "Read " << triangles_host.size() << " Triangles\n";

  Kokkos::View<Triangle<2> *, typename DeviceType::memory_space> triangles(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "triangles"),
      triangles_host.size());
  Kokkos::deep_copy(execution_space, triangles,
                    Kokkos::View<Triangle<2> *, Kokkos::HostSpace>(
                        triangles_host.data(), triangles_host.size()));

  Kokkos::View<Mapping *, typename DeviceType::memory_space> mappings(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "mappings"),
      mappings_host.size());
  Kokkos::deep_copy(execution_space, mappings,
                    Kokkos::View<Mapping *, Kokkos::HostSpace>(
                        mappings_host.data(), mappings_host.size()));

  return {triangles, mappings};
}

template <typename DeviceType>
Kokkos::View<ArborX::ExperimentalHyperGeometry::Point<2> **, Kokkos::LayoutLeft,
             typename DeviceType::memory_space>
parse_points(typename DeviceType::execution_space const &execution_space)
{
  std::vector<ArborX::ExperimentalHyperGeometry::Point<2>> points_host;
  std::ifstream step_file("RK4Steps.txt");
  if (!step_file.good())
    throw std::runtime_error("Cannot open file");
  Mapping mapping;
  ArborX::ExperimentalHyperGeometry::Point<2> point;
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
  Kokkos::View<ArborX::ExperimentalHyperGeometry::Point<2> **,
               Kokkos::LayoutRight, Kokkos::HostSpace>
      points_host_view(points_host.data(), points_host.size() / size_per_id,
                       size_per_id);
  std::cout << "id 0, point 1: " << points_host_view(0, 1)[0] << ' '
            << points_host_view(0, 1)[1] << std::endl;
  std::cout << "id 1, point 0: " << points_host_view(1, 0)[0] << ' '
            << points_host_view(1, 0)[1] << std::endl;

  Kokkos::View<ArborX::ExperimentalHyperGeometry::Point<2> **,
               Kokkos::LayoutLeft, typename DeviceType::memory_space>
      points(Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"),
             points_host.size() / size_per_id, size_per_id);
  Kokkos::deep_copy(execution_space, points, points_host_view);

  return points;
}

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
        MemorySpace, ArborX::ExperimentalHyperGeometry::Box<2>> const
        tree(execution_space, triangles);
    std::cout << "BVH tree set up.\n";

    std::cout << "Starting the queries.\n";
    int const n = points.extent(0);

    struct Dummy
    {};

    struct Attachment
    {
      int &triangle_index;
      ArborX::Point &coeffs;
    };

    ArborX::Details::TreeTraversal<
        ArborX::BasicBoundingVolumeHierarchy<
            MemorySpace, ArborX::ExperimentalHyperGeometry::Box<2>>,
        Dummy, TriangleIntersectionCallback<DeviceType>,
        ArborX::Details::SpatialPredicateTag,
        decltype(ArborX::attach(
            ArborX::intersects(ArborX::ExperimentalHyperGeometry::Point<2>{}),
            std::declval<Attachment>()))>
        tree_traversal(tree,
                       TriangleIntersectionCallback<DeviceType>{triangles});

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
            auto const &triangle = triangles.get_triangle(triangle_index);
            auto const &test_coeffs =
                triangles.get_mapping(triangle_index).get_coeff(point);
            bool intersects = test_coeffs[0] >= 0 && test_coeffs[1] >= 0 &&
                              test_coeffs[2] >= 0;
            if (intersects)
            {
              coefficients = test_coeffs;
              KOKKOS_IMPL_DO_NOT_USE_PRINTF("%d, %d: same triangle\n", i, j);
            }
            else
            {
              tree_traversal.search(
                  ArborX::attach(ArborX::intersects(point),
                                 Attachment{triangle_index, coefficients}));
              KOKKOS_IMPL_DO_NOT_USE_PRINTF("%d, %d: %d %f %f %f\n", i, j,
                                            triangle_index, coefficients[0],
                                            coefficients[1], coefficients[2]);
            }
          }
        });

    std::cout << "Queries done.\n";
  }

  Kokkos::finalize();
}
