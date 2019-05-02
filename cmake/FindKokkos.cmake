#[=======================================================================[.rst:
FindKokkos
-------

Finds the Kokkos library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``Kokkos::Kokkos``
  The Kokkos library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Kokkos_FOUND``
  True if the system has the Kokkos library.
``Kokkos_VERSION``
  The version of the Kokkos library which was found.
``Kokkos_INCLUDE_DIRS``
  Include directories needed to use Kokkos.
``Kokkos_LIBRARIES``
  Libraries needed to link to Kokkos.
``Kokkos_DEVICES``
  Set of backends enabled.
``kokkos_ARCH``
  Target architectures.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Kokkos_INCLUDE_DIR``
  The directory containing ``Kokkos_Core.hpp``.
``Kokkos_LIBRARY``
  The path to the Kokkos library.

#]=======================================================================]

find_package(PkgConfig)
pkg_check_modules(PC_Kokkos QUIET kokkos)

find_path(Kokkos_INCLUDE_DIR
  NAMES Kokkos_Core.hpp
  PATHS ${PC_Kokkos_INCLUDE_DIRS}
)
find_library(Kokkos_LIBRARY
  NAMES kokkos
  PATHS ${PC_Kokkos_LIBRARY_DIRS}
)
find_path(_Kokkos_SETTINGS
  NAMES kokkos_generated_settings.cmake
  PATHS ${PC_Kokkos_PREFIX}
        ${PC_Kokkos_LIBRARY_DIRS}/lib/cmake/Kokkos
)

set(Kokkos_VERSION ${PC_Kokkos_VERSION})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Kokkos
  FOUND_VAR Kokkos_FOUND
  REQUIRED_VARS
    Kokkos_LIBRARY
    Kokkos_INCLUDE_DIR
    _Kokkos_SETTINGS
  VERSION_VAR Kokkos_VERSION
)

include(${_Kokkos_SETTINGS}/kokkos_generated_settings.cmake)
unset(_Kokkos_SETTINGS CACHE)

if(Kokkos_FOUND)
  set(Kokkos_LIBRARIES ${Kokkos_LIBRARY})
  set(Kokkos_INCLUDE_DIRS ${Kokkos_INCLUDE_DIR})
  set(Kokkos_DEFINITIONS ${PC_Kokkos_CFLAGS_OTHER})
  set(Kokkos_DEVICES ${KOKKOS_GMAKE_DEVICES})
  set(Kokkos_ARCH ${KOKKOS_GMAKE_ARCH})
endif()

if(Kokkos_FOUND AND NOT TARGET Kokkos::Kokkos)
  add_library(Kokkos::Kokkos UNKNOWN IMPORTED)
  set_target_properties(Kokkos::Kokkos PROPERTIES
    IMPORTED_LOCATION "${Kokkos_LIBRARY}"
    INTERFACE_COMPILE_OPTIONS "${PC_Kokkos_CFLAGS_OTHER}"
    INTERFACE_LINK_LIBRARIES "${PC_Kokkos_LDFLAGS}"
    INTERFACE_INCLUDE_DIRECTORIES "${Kokkos_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(
  Kokkos_INCLUDE_DIR
  Kokkos_LIBRARY
)
