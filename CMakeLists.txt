INCLUDE(CombinedOption)
INCLUDE(TribitsETISupport)

TRIBITS_SUBPACKAGE(Search)

TRIBITS_ADD_EXPLICIT_INSTANTIATION_OPTION()

TRIBITS_ADD_ETI_SUPPORT()

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_EXPLICIT_INSTANTIATION)

TRIBITS_ADD_OPTION_AND_DEFINE(
  DataTransferKit_ENABLE_EXPLICIT_INSTANTIATION
  HAVE_DATATRANSFERKIT_EXPLICIT_INSTANTIATION
  "Enable explicit template instantiation (ETI) in DTK"
  ${${PACKAGE_NAME}_ENABLE_EXPLICIT_INSTANTIATION}
  )

IF (${PACKAGE_NAME}_ENABLE_EXPLICIT_INSTANTIATION AND NOT ${PROJECT_NAME}_ENABLE_Tpetra)
  GLOBAL_SET(${PACKAGE_NAME}_ENABLE_EXPLICIT_INSTANTIATION  OFF)
  GLOBAL_SET(HAVE_${PACKAGE_NAME_UC}_EXPLICIT_INSTANTIATION OFF)
  MESSAGE(STATUS "Disabling DTK explicit template instantation (ETI) because Tpetra is disabled.")
ENDIF()

#
# Explicit template instantiation (ETI) and test instantiation logic
#

# tpetra/CMakeLists.txt (the package's CMake logic) defines these
# variables.  Despite "ETI" in their names, they exist whether or not
# ETI is defined.  If ETI is defined, these variables govern the set
# of template parameter combinations over which DataTransferKit instantiates.
# Whether or not ETI is defined, the variables govern the set of
# template parameter combinations over which DataTransferKit runs tests.

ASSERT_DEFINED (DataTransferKit_ETI_SCALARS)
ASSERT_DEFINED (DataTransferKit_ETI_LORDS)
ASSERT_DEFINED (DataTransferKit_ETI_GORDS)
ASSERT_DEFINED (DataTransferKit_ETI_NODES)

SET(${PACKAGE_NAME}_ETI_SCALARS "${DataTransferKit_ETI_SCALARS}")
# Exclude all ordinal types (GlobalOrdinal and int).
SET(${PACKAGE_NAME}_ETI_LORDS "${DataTransferKit_ETI_LORDS}")
SET(${PACKAGE_NAME}_ETI_GORDS "${DataTransferKit_ETI_GORDS}")
SET(${PACKAGE_NAME}_ETI_NODES "${DataTransferKit_ETI_NODES}")

# "Export" the names for use in the ETI system.
# If we don't do this, ETI won't see these variables.

GLOBAL_SET(${PACKAGE_NAME}_ETI_SCALARS ${${PACKAGE_NAME}_ETI_SCALARS})
GLOBAL_SET(${PACKAGE_NAME}_ETI_GORDS   ${${PACKAGE_NAME}_ETI_GORDS})
GLOBAL_SET(${PACKAGE_NAME}_ETI_LORDS   ${${PACKAGE_NAME}_ETI_LORDS})
GLOBAL_SET(${PACKAGE_NAME}_ETI_NODES   ${${PACKAGE_NAME}_ETI_NODES})


#
# Add libraries, tests, and examples
#

ADD_SUBDIRECTORY(src)

TRIBITS_INCLUDE_DIRECTORIES(examples/point_clouds)

TRIBITS_ADD_EXAMPLE_DIRECTORIES(examples)
TRIBITS_ADD_TEST_DIRECTORIES(test)

TRIBITS_SUBPACKAGE_POSTPROCESS()
