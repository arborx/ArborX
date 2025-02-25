#include "data.hpp"

#include <Kokkos_Core.hpp>

#include <fstream>
#include <string>

#include "data_timpl.hpp"

// Explicit instantiations
using MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space;
#define INSTANTIATE_LOADER(DIM)                                                \
  template Kokkos::View<ArborX::Point<DIM> *, MemorySpace>                     \
  loadData<DIM, MemorySpace>(ArborXBenchmark::Parameters const &);
INSTANTIATE_LOADER(2)
INSTANTIATE_LOADER(3)
INSTANTIATE_LOADER(4)
INSTANTIATE_LOADER(5)
INSTANTIATE_LOADER(6)
#undef INSTANTIATE_LOADER

int getDataDimension(std::string const &filename, bool binary)
{
  std::ifstream input;
  if (!binary)
    input.open(filename);
  else
    input.open(filename, std::ifstream::binary);
  if (!input.good())
    throw std::runtime_error("Error reading file \"" + filename + "\"");

  int num_points;
  int dim;
  if (!binary)
  {
    input >> num_points;
    input >> dim;
  }
  else
  {
    input.read(reinterpret_cast<char *>(&num_points), sizeof(int));
    input.read(reinterpret_cast<char *>(&dim), sizeof(int));
  }
  input.close();

  return dim;
}
