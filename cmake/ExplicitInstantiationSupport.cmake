include(Join)
MESSAGE(STATUS "${PACKAGE_NAME}: Processing ETI / test support")

# DataTransferKit ETI type fields. S, LO, GO, N correspond to the four
# template parameters of most Tpetra classes: Scalar, LocalOrdinal,
# GlobalOrdinal, and Node.  DataTransferKit shares these with Tpetra, because
# DataTransferKit only works with Tpetra linear algebra objects.
SET(${PACKAGE_NAME}_ETI_FIELDS "S|LO|GO|N")

# Set up a pattern that excludes all complex Scalar types.
# TriBITS' ETI system knows how to interpret this pattern.
TRIBITS_ETI_TYPE_EXPANSION(${PACKAGE_NAME}_ETI_EXCLUDE_SET_COMPLEX "S=std::complex<float>|std::complex<double>" "LO=.*" "GO=.*" "N=.*")

# TriBITS' ETI system expects a set of types to be a string, delimited
# by |.  Each template parameter (e.g., Scalar, LocalOrdinal, ...) has
# its own set.  The JOIN commands below set up those lists.  We use
# the following sets that DataTransferKit defines:
#
# Scalar: DataTransferKit_ETI_SCALARS
# LocalOrdinal: DataTransferKit_ETI_LORDS
# GlobalOrdinal: DataTransferKit_ETI_GORDS
# Node: DataTransferKit_ETI_NODES
#
# Note that the Scalar set from Tpetra includes the Scalar =
# GlobalOrdinal case.  However, DataTransferKit's CMake logic excludes this,
# so we don't have to worry about it here.

JOIN(${PACKAGE_NAME}_ETI_SCALARS "|" FALSE ${${PACKAGE_NAME}_ETI_SCALARS})
JOIN(${PACKAGE_NAME}_ETI_LORDS   "|" FALSE ${${PACKAGE_NAME}_ETI_LORDS}  )
JOIN(${PACKAGE_NAME}_ETI_GORDS   "|" FALSE ${${PACKAGE_NAME}_ETI_GORDS}  )
JOIN(${PACKAGE_NAME}_ETI_NODES   "|" FALSE ${${PACKAGE_NAME}_ETI_NODES}  )

MESSAGE(STATUS "Enabled Scalar types:        ${${PACKAGE_NAME}_ETI_SCALARS}")
MESSAGE(STATUS "Enabled LocalOrdinal types:  ${${PACKAGE_NAME}_ETI_LORDS}")
MESSAGE(STATUS "Enabled GlobalOrdinal types: ${${PACKAGE_NAME}_ETI_GORDS}")
MESSAGE(STATUS "Enabled Node types:          ${${PACKAGE_NAME}_ETI_NODES}")

# Construct the "type expansion" string that TriBITS' ETI system
# expects.  Even if ETI is OFF, we will use this to generate macros
# for instantiating tests.
TRIBITS_ETI_TYPE_EXPANSION(SingleScalarInsts
  "S=${${PACKAGE_NAME}_ETI_SCALARS}"
  "N=${${PACKAGE_NAME}_ETI_NODES}"
  "LO=${${PACKAGE_NAME}_ETI_LORDS}"
  "GO=${${PACKAGE_NAME}_ETI_GORDS}")

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_EXPLICIT_INSTANTIATION)
IF(${PACKAGE_NAME}_ENABLE_EXPLICIT_INSTANTIATION)
  MESSAGE(STATUS "User/Downstream ETI set: ${${PACKAGE_NAME}_ETI_LIBRARYSET}")
  TRIBITS_ADD_ETI_INSTANTIATIONS(${PACKAGE_NAME} ${SingleScalarInsts})
  MESSAGE(STATUS "Excluded type combinations: ${${PACKAGE_NAME}_ETI_EXCLUDE_SET}")
ELSE()
  TRIBITS_ETI_TYPE_EXPANSION(${PACKAGE_NAME}_ETI_LIBRARYSET
    "S=${${PACKAGE_NAME}_ETI_SCALARS}"
    "N=${${PACKAGE_NAME}_ETI_NODES}"
    "LO=${${PACKAGE_NAME}_ETI_LORDS}"
    "GO=${${PACKAGE_NAME}_ETI_GORDS}")
ENDIF()
MESSAGE(STATUS "Set of enabled types, before exclusions: ${${PACKAGE_NAME}_ETI_LIBRARYSET}")

#
# Generate the instantiation macros.  These go into
# DataTransferKit_ETIHelperMacros.h, which is generated from
# DataTransferKit_ETIHelperMacros.h.in (in this directory).
#
TRIBITS_ETI_GENERATE_MACROS("${${PACKAGE_NAME}_ETI_FIELDS}" "${${PACKAGE_NAME}_ETI_LIBRARYSET}" "${${PACKAGE_NAME}_ETI_EXCLUDE_SET}"
                            list_of_manglings eti_typedefs
                            "DTK_INSTANTIATE_L(LO)"         DTK_ETIMACRO_L
                            "DTK_INSTANTIATE_SL(S,LO)"      DTK_ETIMACRO_SL
                            "DTK_INSTANTIATE_LG(LO,GO)"         DTK_ETIMACRO_LG
                            "DTK_INSTANTIATE_SLG(S,LO,GO)"      DTK_ETIMACRO_SLG
                            "DTK_INSTANTIATE_LGN(LO,GO,N)"      DTK_ETIMACRO_LGN
                            "DTK_INSTANTIATE_SLGN(S,LO,GO,N)"     DTK_ETIMACRO_SLGN
                            "DTK_INSTANTIATE_SN(S,N)"     DTK_ETIMACRO_SN
                            "DTK_INSTANTIATE_N(N)"     DTK_ETIMACRO_N
                            )
TRIBITS_ETI_GENERATE_MACROS("${${PACKAGE_NAME}_ETI_FIELDS}" "${${PACKAGE_NAME}_ETI_LIBRARYSET}"
                            "${${PACKAGE_NAME}_ETI_EXCLUDE_SET};${${PACKAGE_NAME}_ETI_EXCLUDE_SET_COMPLEX}"
                            list_of_manglings eti_typedefs
                            "DTK_INSTANTIATE_SL_REAL(S,LO,GO)" DTK_ETIMACRO_SL_REAL
                            "DTK_INSTANTIATE_SLG_REAL(S,LO,GO)" DTK_ETIMACRO_SLG_REAL
                            "DTK_INSTANTIATE_SLGN_REAL(S,LO,GO,N)" DTK_ETIMACRO_SLGN_REAL
                            "DTK_INSTANTIATE_SN_REAL(S,N)" DTK_ETIMACRO_SN_REAL
                            )

# Generate "mangled" typedefs.  Macros sometimes get grumpy when types
# have spaces, colons, or angle brackets in them.  This includes types
# like "long long" or "std::complex<double>".  Thus, we define
# typedefs that remove the offending characters.  The typedefs also
# get written to the generated header file.
TRIBITS_ETI_GENERATE_TYPEDEF_MACRO(DTK_ETI_TYPEDEFS "DTK_ETI_MANGLING_TYPEDEFS" "${eti_typedefs}")

# Generate the header file ${PACKAGE_NAME}_ETIHelperMacros.h, from the file
# ${PACKAGE_NAME}_ETIHelperMacros.h.in
CONFIGURE_FILE(
  ${DataTransferKit_SOURCE_DIR}/packages/Search/cmake/${PACKAGE_NAME}_ETIHelperMacros.h.in
  ${DataTransferKit_BINARY_DIR}/packages/Search/src/${PACKAGE_NAME}_ETIHelperMacros.h)
