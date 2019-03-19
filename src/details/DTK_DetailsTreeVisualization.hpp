/****************************************************************************
 * Copyright (c) 2012-2019 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef DTK_DETAILS_TREE_VISUALIZATION_HPP
#define DTK_DETAILS_TREE_VISUALIZATION_HPP

#include <DTK_DetailsTreeTraversal.hpp>

namespace DataTransferKit
{
namespace Details
{
template <typename DeviceType>
struct TreeVisualization
{
    // You have been warned :)
    using TreeAccess = typename TreeTraversal<
        DeviceType>::DoNotUseUnlessYouKnowWhatYouAreDoing;

    template <typename Tree>
    static std::string getNodeLabel( Node const *node, Tree const &tree )
    {
        auto const node_is_leaf = TreeAccess::isLeaf( node, tree );
        auto const node_index = TreeAccess::getIndex( node, tree );
        std::string label = node_is_leaf ? "l" : "i";
        label.append( std::to_string( node_index ) );
        return label;
    }

    template <typename Tree>
    static std::string getNodeAttributes( Node const *node, Tree const &tree )
    {
        return TreeAccess::isLeaf( node, tree ) ? "[leaf]" : "[internal]";
    }

    template <typename Tree>
    static std::string getEdgeAttributes( Node const *parent, Node const *child,
                                          Tree const &tree )
    {
        return TreeAccess::isLeaf( child, tree ) ? "[pendant]" : "[edge]";
    }

    struct GraphvizVisitor
    {
        std::ostream &_os;

        template <typename Tree>
        void visit( Node const *node, Tree const &tree ) const
        {
            visitNode( node, tree );
            visitEdgesStartingFromNode( node, tree );
        }

        template <typename Tree>
        void visitNode( Node const *node, Tree const &tree ) const
        {
            auto const node_label = getNodeLabel( node, tree );
            auto const node_attributes = getNodeAttributes( node, tree );

            _os << "    " << node_label << " " << node_attributes << ";\n";
        }

        template <typename Tree>
        void visitEdgesStartingFromNode( Node const *node,
                                         Tree const &tree ) const
        {
            auto const node_label = getNodeLabel( node, tree );
            auto const node_is_internal = !TreeAccess::isLeaf( node, tree );

            if ( node_is_internal )
                for ( Node const *child :
                      {node->children.first, node->children.second} )
                {
                    auto const child_label = getNodeLabel( child, tree );
                    auto const edge_attributes =
                        getEdgeAttributes( node, child, tree );

                    _os << "    " << node_label << " -> " << child_label << " "
                        << edge_attributes << ";\n";
                }
        }
    };

    template <typename Tree, typename Visitor>
    static void visitAllIterative( Tree const &tree, Visitor const &visitor )
    {
        Stack<Node const *> stack;
        stack.emplace( TreeAccess::getRoot( tree ) );
        while ( !stack.empty() )
        {
            Node const *node = stack.top();
            stack.pop();

            visitor.visit( node, tree );

            if ( !TreeAccess::isLeaf( node, tree ) )
                for ( Node const *child :
                      {node->children.first, node->children.second} )
                    stack.push( child );
        }
    }
};
} // namespace Details
} // namespace DataTransferKit

#endif
