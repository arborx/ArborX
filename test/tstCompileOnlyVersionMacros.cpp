#include <ArborX_Config.hpp>

#ifndef ARBORX_VERSION
static_assert(false, "ARBORX_VERSION macro is not defined!");
#endif

#ifndef ARBORX_VERSION_MAJOR
static_assert(false, "ARBORX_VERSION_MAJOR macro is not defined!");
#endif

#ifndef ARBORX_VERSION_MINOR
static_assert(false, "ARBORX_VERSION_MINOR macro is not defined!");
#endif

#ifndef ARBORX_VERSION_PATCH
static_assert(false, "ARBORX_VERSION_PATCH macro is not defined!");
#endif

static_assert(0 <= ARBORX_VERSION_MAJOR && ARBORX_VERSION_MAJOR <= 99);
static_assert(0 <= ARBORX_VERSION_MINOR && ARBORX_VERSION_MINOR <= 99);
static_assert(0 <= ARBORX_VERSION_PATCH && ARBORX_VERSION_PATCH <= 99);
static_assert(ARBORX_VERSION == ARBORX_VERSION_MAJOR * 10000 +
                                    ARBORX_VERSION_MINOR * 100 +
                                    ARBORX_VERSION_PATCH);
