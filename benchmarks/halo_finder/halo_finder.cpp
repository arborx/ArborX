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

#include <ArborX_HaloFinder.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <chrono>
#include <fstream>

std::vector<ArborX::Point> parsePoints(std::string const &filename,
                                       bool binary = false)
{
  std::cout << "Reading in \"" << filename << "\" in "
            << (binary ? "binary" : "text") << " mode...";
  std::cout.flush();

  std::ifstream input;
  if (!binary)
    input.open(filename);
  else
    input.open(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  int num_points = 0;
  std::vector<float> x, y, z;
  if (!binary)
  {
    input >> num_points;

    x.reserve(num_points);
    y.reserve(num_points);
    z.reserve(num_points);

    auto read_float = [&input]() {
      return *(std::istream_iterator<float>(input));
    };
    std::generate_n(std::back_inserter(x), num_points, read_float);
    std::generate_n(std::back_inserter(y), num_points, read_float);
    std::generate_n(std::back_inserter(z), num_points, read_float);
  }
  else
  {
    input.read(reinterpret_cast<char *>(&num_points), sizeof(int));

    x.resize(num_points);
    y.resize(num_points);
    z.resize(num_points);
    input.read(reinterpret_cast<char *>(x.data()), num_points * sizeof(float));
    input.read(reinterpret_cast<char *>(y.data()), num_points * sizeof(float));
    input.read(reinterpret_cast<char *>(z.data()), num_points * sizeof(float));
  }
  input.close();
  std::cout << "done\nRead in " << num_points << " points" << std::endl;

  std::vector<ArborX::Point> v(num_points);
  for (int i = 0; i < num_points; i++)
  {
    v[i] = {x[i], y[i], z[i]};
  }

  return v;
}

template <typename... P, typename T>
auto vec2view(std::vector<T> const &in, std::string const &label = "")
{
  Kokkos::View<T *, P...> out(
      Kokkos::view_alloc(label, Kokkos::WithoutInitializing), in.size());
  auto const start = std::chrono::high_resolution_clock::now();
  Kokkos::deep_copy(out, Kokkos::View<T const *, Kokkos::HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>{
                             in.data(), in.size()});
  auto const end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  printf("deep copy       : %10.3f\n", elapsed_seconds.count());
  return out;
}

template <typename View>
struct Wrapped
{
  View _M_view;
  double _r;
};

template <typename View>
auto wrap(View v, double r)
{
  return Wrapped<View>{v, r};
}

namespace ArborX
{
namespace Traits
{
template <typename View>
struct Access<Wrapped<View>, PredicatesTag>
{
  using memory_space = typename View::memory_space;
  static size_t size(Wrapped<View> const &w) { return w._M_view.extent(0); }
  static KOKKOS_FUNCTION auto get(Wrapped<View> const &w, size_t i)
  {
    return attach(intersects(Sphere{w._M_view(i), w._r}), (int)i);
  }
};
} // namespace Traits
} // namespace ArborX

template <typename MemorySpace>
struct HaloCallback
{
  Kokkos::View<int *, MemorySpace> stat_;

  KOKKOS_INLINE_FUNCTION
  int representative(int const i) const
  {
    int curr = stat_(i);
    if (curr != i)
    {
      int next, prev = i;
      while (curr > (next = stat_(curr)))
      {
        stat_(prev) = next;
        prev = curr;
        curr = next;
      }
    }
    return curr;
  }

  using tag = ArborX::Details::InlineCallbackTag;
  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &query, int j,
                                  Insert const &) const
  {
    int const i = ArborX::getData(query);

    // Only process edge in one direction
    if (i > j)
    {
      // Initialize to the first neighbor that's smaller
      if (Kokkos::atomic_compare_exchange(&stat_(i), i, j) == i)
        return;

      int vstat = representative(i);
      int ostat = representative(j);

      bool repeat;
      do
      {
        repeat = false;
        if (vstat != ostat)
        {
          int ret;
          if (vstat < ostat)
          {
            if ((ret = Kokkos::atomic_compare_exchange(&stat_(ostat), ostat,
                                                       vstat)) != ostat)
            {
              ostat = ret;
              repeat = true;
            }
          }
          else
          {
            if ((ret = Kokkos::atomic_compare_exchange(&stat_(vstat), vstat,
                                                       ostat)) != vstat)
            {
              vstat = ret;
              repeat = true;
            }
          }
        }
      } while (repeat);
    }
  }
};

int main(int argc, char *argv[])
{
  using ExecutionSpace = Kokkos::Cuda;
  using MemorySpace = Kokkos::CudaSpace;

  Kokkos::ScopeGuard guard(argc, argv);

  std::cout << "ArborX version: " << ArborX::version() << std::endl;
  std::cout << "ArborX hash   : " << ArborX::gitCommitHash() << std::endl;

  namespace bpo = boost::program_options;

  std::string filename;
  bool binary;
  double linking_length;
  int min_size;

  bpo::options_description desc("Allowed options");
  // clang-format off
    desc.add_options()
        ( "help", "help message" )
        ( "filename,f", bpo::value<std::string>(&filename), "filename containing data" )
        ( "binary,b", bpo::bool_switch(&binary)->default_value(false), "binary file indicator")
        ( "linking-length,l", bpo::value<double>(&linking_length), "linking length (radius)" )
        ( "min-size,s", bpo::value<int>(&min_size)->default_value(2), "minimum halo size")
        ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << '\n';
    return 1;
  }

  // read in data
  auto const points = parsePoints(filename, binary);
  auto const primitives = vec2view<MemorySpace>(points, "primitives");
  auto const predicates =
      wrap(vec2view<MemorySpace>(points, "predicates"), linking_length);

  int const n = points.size();

  using clock = std::chrono::high_resolution_clock;

  clock::time_point start_total, start;
  std::chrono::duration<double> elapsed_construction, elapsed_query,
      elapsed_halos, elapsed_total;

  ExecutionSpace exec_space;

  start_total = clock::now();

  // build the tree
  start = clock::now();
  ArborX::BVH<MemorySpace> bvh(exec_space, primitives);
  elapsed_construction = clock::now() - start;

  // perform queries
  start = clock::now();
  Kokkos::View<int *, MemorySpace> indices("indices", 0);
  Kokkos::View<int *, MemorySpace> offset("offset", 0);
  Kokkos::View<int *, MemorySpace> stat(
      Kokkos::ViewAllocateWithoutInitializing("stat"), n);
  ArborX::iota(exec_space, stat);
  // indices and offfset are not going to be used, as we never call insert()
  bvh.query(exec_space, predicates, HaloCallback<MemorySpace>{stat}, indices,
            offset);
  // flatten stat
  Kokkos::parallel_for("flatten",
                       Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                       KOKKOS_LAMBDA(int const i) {
                         int next, vstat = stat(i);
                         int const old = vstat;
                         while (vstat > (next = stat(vstat)))
                         {
                           vstat = next;
                         }
                         if (vstat != old)
                           stat(i) = vstat;
                       });
  elapsed_query = clock::now() - start;

  // find halos
  start = clock::now();
  Kokkos::View<int *, MemorySpace> halos_offset("halos_offset", 0);
  Kokkos::View<int *, MemorySpace> halos_indices("halos_indices", 0);
  ArborX::HaloFinder::findHalos(exec_space, stat, halos_offset, halos_indices,
                                min_size);
  elapsed_halos = clock::now() - start;

  elapsed_total = clock::now() - start_total;

#if 0
  int num_halos = halos_offset.size() - 1;

  auto halos_offset_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, halos_offset);
  auto halos_indices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, halos_indices);

  // Print halos
  std::cout << "Halos: \n";
  for (int i = 0; i < num_halos; i++)
  {
    std::cout << "#" << i << ": ";
    for (int j = halos_offset_host(i); j < halos_offset_host(i+1); j++)
    {
      std::cout << " " << halos_indices_host(j);
    }
    std::cout << std::endl;
  }
#endif

#if 0
  int num_halos = halos_offset.size() - 1;

  auto halos_offset_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, halos_offset);
  auto halos_indices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, halos_indices);

  // Compute halo centers
  for (int i = 0; i < num_halos; i++)
  {
    int halo_size = halos_offset_host(i + 1) - halos_offset_host(i);
    ArborX::Point halo_center{0.f, 0.f, 0.f};
    for (int j = halos_offset_host(i); j < halos_offset_host(i + 1); j++)
    {
      auto const &halo_point = points[halos_indices_host(j)];
      halo_center[0] += halo_point[0];
      halo_center[1] += halo_point[1];
      halo_center[2] += halo_point[2];
    }
    halo_center[0] /= halo_size;
    halo_center[1] /= halo_size;
    halo_center[2] /= halo_size;
    if (halo_center[0] >= 0 && halo_center[1] >= 0 && halo_center[2] >= 0 &&
        halo_center[0] < 64 && halo_center[1] < 64 && halo_center[2] < 64)
    {
      printf("%d %e %e %e\n", halo_size, halo_center[0], halo_center[1],
             halo_center[2]);
    }
  }
#endif

  printf("total time      : %10.3f\n", elapsed_total.count());
  printf("-> construction : %10.3f\n", elapsed_construction.count());
  printf("-> query+ccs    : %10.3f\n", elapsed_query.count());
  printf("-> halos        : %10.3f\n", elapsed_halos.count());

  return EXIT_SUCCESS;
}
