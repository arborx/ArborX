#include <Kokkos_Core.hpp>

#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

#if defined( DTK_SEARCH_MPI_UNIT_TEST )
#include <mpi.h>
#endif

struct ExecutionEnvironmentScopeGuard
{
    ExecutionEnvironmentScopeGuard( int argc, char *argv[] )
    {
#if defined( DTK_SEARCH_MPI_UNIT_TEST )
        MPI_Init( &argc, &argv );
#endif
        Kokkos::initialize( argc, argv );
    }
    ~ExecutionEnvironmentScopeGuard()
    {
        Kokkos::finalize();
#if defined( DTK_SEARCH_MPI_UNIT_TEST )
        MPI_Finalize();
#endif
    }
};

bool init_function() { return true; }

int main( int argc, char *argv[] )
{
    ExecutionEnvironmentScopeGuard scope_guard( argc, argv );
    return boost::unit_test::unit_test_main( &init_function, argc, argv );
}
