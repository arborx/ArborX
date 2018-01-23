/****************************************************************************
 * Copyright (c) 2012-2018 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/
#ifndef DTK_DETAILS_TEUCHOS_SERIALIZATION_TRAITS_HPP
#define DTK_DETAILS_TEUCHOS_SERIALIZATION_TRAITS_HPP

#include <DTK_DetailsBox.hpp>
#include <DTK_DetailsPoint.hpp>
#include <DTK_DetailsPredicate.hpp>
#include <DTK_DetailsSphere.hpp>

#include <Teuchos_SerializationTraits.hpp>

namespace Teuchos
{

template <typename Ordinal>
class SerializationTraits<Ordinal, DataTransferKit::Point>
    : public DirectSerializationTraits<Ordinal, DataTransferKit::Point>
{
};

template <typename Ordinal>
class SerializationTraits<Ordinal, DataTransferKit::Box>
    : public DirectSerializationTraits<Ordinal, DataTransferKit::Box>
{
};

template <typename Ordinal>
class SerializationTraits<Ordinal, DataTransferKit::Sphere>
    : public DirectSerializationTraits<Ordinal, DataTransferKit::Sphere>
{
};

template <typename Ordinal, typename Geometry>
class SerializationTraits<Ordinal, DataTransferKit::Details::Nearest<Geometry>>
    : public DirectSerializationTraits<
          Ordinal, DataTransferKit::Details::Nearest<Geometry>>
{
};

template <typename Ordinal, typename Geometry>
class SerializationTraits<Ordinal,
                          DataTransferKit::Details::Intersects<Geometry>>
    : public DirectSerializationTraits<
          Ordinal, DataTransferKit::Details::Intersects<Geometry>>
{
};

} // end namespace Teuchos

#endif
