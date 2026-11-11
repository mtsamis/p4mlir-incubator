// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_IMPL_UTILS_LOCATION_MAPPING_H
#define P4MLIR_IMPL_UTILS_LOCATION_MAPPING_H

#include <deque>

#include "p4mlir/Utilities/IRUtils.h"

namespace P4::P4MLIR::IRUtils {

/// A helper class to store remapped "locations" between a source and a target collections of types.
struct LocationMapping2 {
    struct Source;
    struct Target;

    /// Represents a location. A location is a field in one of the source or target collections.
    /// A location may be empty or have an empty path. A location with an empty path belongs to a
    /// collection and represents a field in that location which is still unspecified.
    struct Location {
        // Special constructor for empty location marker.
        Location() : Location(P4HIR::FieldPath(), std::numeric_limits<unsigned>::max()) {}

        Location(P4HIR::FieldPath path, unsigned index) : path(path), index(index) {}

        /// Return true if this is a non-empty location.
        operator bool() const { return index != std::numeric_limits<unsigned>::max(); }

        /// Get the path in the collection.
        P4HIR::FieldPath getPath() const { return path; }

        /// Return if a path is present.
        bool hasPath() const { return !path.isEmpty(); }

        /// Return the index within the collection.
        unsigned getIndex() const { return index; }

     private:
        P4HIR::FieldPath path;
        unsigned index;
    };

    /// A struct holding information for a partial mapping for a source to a target path.
    /// Represents that (sourceRoot ++ sourceField) has a mapping to (targetRoot ++ targetField).
    struct PartialMapping {
        Location sourceRoot;
        P4HIR::FieldPath sourceField;
        Location targetRoot;
        P4HIR::FieldPath targetField;
    };

 private:
    /// A node to build a trie-like data structure for mapping fields to `Location`s.
    /// Fields are implicitly represented as trails of field indices, starting from any type.
    /// Invariant: If `!children.empty()` then all its leaf nodes have a non-empty mapping.
    /// Only the root may be a leaf node but have no mapping.
    struct MappingNode {
        /// Get or create a child node representing the `index`-th field.
        MappingNode *getOrCreate(unsigned index) {
            auto [it, ins] = children.try_emplace(index);  // value-initializes an empty unique_ptr
            if (ins) it->second = std::make_unique<MappingNode>();
            return it->second.get();
        }

        /// Get a child node representing the `index`-th field if present, return nullptr otherwise.
        MappingNode *get(unsigned index) {
            auto it = children.find(index);
            return (it != children.end()) ? it->second.get() : nullptr;
        }

        /// Returns true if there are no mappings for subfields of the current node.
        bool isLeaf() const { return children.empty(); }

        /// Return children as a range of tuple<index, node*>.
        auto getEntries() {
            return llvm::map_range(children,
                                   [](auto &e) { return std::pair{e.first, e.second.get()}; });
        }

        /// Map from field index to a nested node.
        llvm::DenseMap<unsigned, std::unique_ptr<MappingNode>> children;

        /// The mapping payload.
        Location to;
    };

 public:
    /// Represents an entry in the target collection.
    /// Target collections may initially have no root type, in which case can only create unknown
    /// locations.
    struct Target {
        /// Set the root type after creation. Can only be called once with a non-empty type.
        void setRootType(mlir::Type type) {
            assert(type && "Expected a non-empty type");
            assert(!root && "Cannot change root type once set");
            root = type;
        }

        /// Return a location for `path` within this entry. The entry must have a type specified and
        /// the root type of `path` must match with it.
        Location loc(P4HIR::FieldPath path) {
            assert(root && "Cannot get location when root type is not set");
            assert((path.getRootType() == root) && "Path root type does not match");
            return Location(path, index);
        }

        /// Return a range for all locations in `paths`.
        template <typename CollectionT>
        auto locs(CollectionT paths) {
            return llvm::map_range(paths, [this](P4HIR::FieldPath path) { return loc(path); });
        }

        /// Return an unknown location within this entry.
        Location loc() { return Location(P4HIR::FieldPath(), index); }

     private:
        friend struct LocationMapping2;

        Target(mlir::Type root, unsigned index) : root(root), index(index) {}

        mlir::Type root;
        unsigned index;
    };

    /// Represents an entry in the source collection.
    struct Source {
        /// Return a location for `path` within this entry. The root type of `path` must match the
        /// entry's type.
        Location loc(P4HIR::FieldPath path) {
            assert((path.getRootType() == root) && "Path root type does not match");
            return Location(path, index);
        }

        /// Return a range for all locations in `paths`.
        template <typename CollectionT>
        auto locs(CollectionT paths) {
            return llvm::map_range(paths, [this](P4HIR::FieldPath path) { return loc(path); });
        }

        /// Add a mapping specifying that `from` must be mapped to `to`.
        /// Return true if there was a previous different mapping for `from`.
        bool addMapping(Location from, Location to) {
            assert((from.getIndex() == index) && "Location index mismatch");
            MappingNode *node = getOrCreate(from.getPath());
            bool isNewMapping = !node->to;
            node->to = to;
            return isNewMapping;
        }

        /// Add a mapping specifying that `from` must be mapped to `to`.
        /// Return true if there was a previous different mapping for `from`.
        bool addMapping(P4HIR::FieldPath from, Location to) { return addMapping(loc(from), to); }

        /// Add mappings specifying that elements of `fromCollection` must be mapped to their
        /// corresponding ones in `toCollection`.
        template <typename Collection1T, typename Collection2T>
        void addMappings(Collection1T fromCollection, Collection2T toCollection) {
            for (auto [from, to] : llvm::zip_equal(fromCollection, toCollection))
                addMapping(from, to);
        }

        /// Get all mappings that affect the remapping of `from`.
        llvm::SmallVector<PartialMapping> remap(Location from) {
            assert((from.getIndex() == index) && "Location index mismatch");
            llvm::SmallVector<PartialMapping> result;

            PartialMapping rootMapping;
            MappingNode *cur = &mapping;

            if (cur->to) {
                rootMapping.targetRoot = cur->to;
                auto targetType = rootMapping.targetRoot.getPath().getType();
                rootMapping.targetField = P4HIR::FieldPath(targetType);
            }

            for (auto index : from.getPath().field_indices()) {
                if (rootMapping.targetField)
                    rootMapping.targetField.append(index);

                if (cur) {
                    cur = cur->get(index);

                    if (cur && cur->to) {
                        rootMapping.targetRoot = cur->to;
                        auto targetType = rootMapping.targetRoot.getPath().getType();
                        rootMapping.targetField = P4HIR::FieldPath(targetType);
                    }
                }
            }

            if (!rootMapping.targetRoot)
                return result;

            rootMapping.sourceRoot = from;
            result.push_back(rootMapping);

            return result;
        }

        /// Clear all mappings.
        void clear() { mapping = MappingNode(); }

        /// Return the mapped location for `from` if it exists, otherwise an empty location.
        /// Nested mappings for subfields of `from` are not considered.
        Location getMapping(Location from) {
            assert((from.getIndex() == index) && "Location index mismatch");
            MappingNode *node = get(from.getPath());
            return node ? node->to : Location();
        }

        /// Get vector of paths with their mapped locations. If `toplevel` is true then do not
        /// include mappings nested under other mappings.
        llvm::SmallVector<std::pair<P4HIR::FieldPath, Location>> getMappings(bool toplevel = false) {
            llvm::SmallVector<std::pair<P4HIR::FieldPath, Location>> result;
            visit([&](P4HIR::FieldPath from, MappingNode *node) {
                if (node->to) result.emplace_back(from, node->to);
                return !toplevel;
            });
            llvm::sort(result, [](auto &p1, auto &p2) { return p1.first < p2.first; });
            return result;
        }

        /// Get vector of paths that have a mapped location. If `toplevel` is true then do not
        /// include mappings nested under other mappings.
        llvm::SmallVector<P4HIR::FieldPath> getMappedPaths(bool toplevel = false) {
            llvm::SmallVector<P4HIR::FieldPath> result;
            visit([&](P4HIR::FieldPath from, MappingNode *node) {
                if (node->to) result.push_back(from);
                return !toplevel;
            });
            llvm::sort(result);
            return result;
        }

        /// Get vector of toplevel paths that do not have a mapped location.
        llvm::SmallVector<P4HIR::FieldPath> getUnmappedPaths() {
            llvm::SmallVector<P4HIR::FieldPath> result;
            visit([&](P4HIR::FieldPath from, MappingNode *node) {
                // Stop if `node` has a mapping.
                if (node->to) {
                    return false;
                }

                if (node->isLeaf()) {
                    // Only the root may be a leaf but without a mapping.
                    result.push_back(from);
                    return false;
                }

                // Add all children that have no nested mappings to the result.
                for (auto [index, childPath] : llvm::enumerate(from.children())) {
                    if (node->children.find(index) == node->children.end())
                        result.push_back(childPath);
                }

                return true;
            });
            llvm::sort(result);
            return result;
        }

        /// Return true iff this mapping is complete. A complete mapping is one where all source
        /// fields can be remapped in some target location.
        bool isComplete() { return getUnmappedPaths().empty(); }

     private:
        friend struct LocationMapping2;

        Source(mlir::Type root, unsigned index) : root(root), index(index) {
            assert(root && "Expected a non-empty type");
        }

        /// Get or create a node representing `path`.
        MappingNode *getOrCreate(P4HIR::FieldPath path) {
            assert((path.getRootType() == root) && "Path root type does not match");
            MappingNode *cur = &mapping;
            for (auto index : path.field_indices()) cur = cur->getOrCreate(index);
            return cur;
        }

        /// Get a node representing `path` if it exists, return nullptr otherwise.
        MappingNode *get(P4HIR::FieldPath path) {
            assert((path.getRootType() == root) && "Path root type does not match");
            MappingNode *cur = &mapping;
            for (auto index : path.field_indices()) {
                cur = cur->get(index);
                if (!cur) break;
            }
            return cur;
        }

        /// Perform a preorder DFS traversal of the trie's nodes, calling `cb` on each node.
        void visit(llvm::function_ref<bool(P4HIR::FieldPath, MappingNode *)> cb) {
            visitImpl(cb, &mapping, P4HIR::FieldPath(root));
        }

        void visitImpl(llvm::function_ref<bool(P4HIR::FieldPath, MappingNode *)> cb,
                       MappingNode *node, P4HIR::FieldPath path) {
            if (!cb(path, node)) return;

            for (auto [index, child] : node->getEntries()) visitImpl(cb, child, path[index]);
        }

        /// Dump all mappings.
        void dumpMapping() {
            if (!mapping.isLeaf() || mapping.to)
                dumpMappingImpl(&mapping, P4HIR::FieldPath(root), 2);
        }

        void dumpMappingImpl(MappingNode *node, P4HIR::FieldPath path, unsigned nestLevel) {
            std::string ident(nestLevel * 2, ' ');
            llvm::dbgs() << ident;
            if (path.getType() == root)
                llvm::dbgs() << "root";
            else
                llvm::dbgs() << path;

            if (node->to) {
                llvm::dbgs() << " -> #" << node->to.getIndex();
                if (node->to.getPath()) llvm::dbgs() << ", " << node->to.getPath();
            }

            llvm::dbgs() << "\n";

            if (node->isLeaf()) return;

            auto entries = llvm::to_vector(node->getEntries());
            llvm::sort(entries, [](auto p1, auto p2) { return p1.first < p2.first; });
            for (auto [index, child] : entries) dumpMappingImpl(child, path[index], nestLevel + 1);
        }

        mlir::Type root;
        MappingNode mapping;
        unsigned index;
    };

    /// Dump sources, targets and all mappings.
    void dump() {
        llvm::dbgs() << "Source collections (" << sourcesCount() << "):\n";
        for (auto [index, source] : llvm::enumerate(sources))
            llvm::dbgs() << "  " << index << ": " << source.root << "\n";
        llvm::dbgs() << "Target collections (" << sourcesCount() << "):\n";
        for (auto [index, target] : llvm::enumerate(targets))
            llvm::dbgs() << "  " << index << ": " << target.root << "\n";
        llvm::dbgs() << "Mappings:\n";
        for (auto [index, source] : llvm::enumerate(sources)) {
            llvm::dbgs() << "  " << index << ":\n";
            source.dumpMapping();
        }
    }

    /// Add a mapping specifying that `from` must be mapped to `to`.
    /// Return true if there was a previous different mapping for `from`.
    bool addMapping(Location from, Location to) {
        auto &fromCollection = sources[from.getIndex()];
        return fromCollection.addMapping(from, to);
    }

    /// Add mappings specifying that elements of `fromCollection` must be mapped to their
    /// corresponding ones in `toCollection`.
    template <typename Collection1T, typename Collection2T>
    void addMappings(Collection1T fromCollection, Collection2T toCollection) {
        for (auto [from, to] : llvm::zip_equal(fromCollection, toCollection)) addMapping(from, to);
    }

    /// Get all mappings that affect the remapping of `from`.
    llvm::SmallVector<PartialMapping> remap(Location from) {
        auto &fromCollection = sources[from.getIndex()];
        return fromCollection.remap(from);
    }

    /// Create a new entry in the sources collection.
    Source &createSource(mlir::Type root) {
        // Private constructor does not allow emplace_back.
        sources.push_back(Source(root, sourcesCount()));
        return sources.back();
    }

    /// Create a new entry in the targets collection.
    Target &createTarget(mlir::Type root = mlir::Type()) {
        // Private constructor does not allow emplace_back.
        targets.push_back(Target(root, targetsCount()));
        return targets.back();
    }

    /// Get the size of the source collection.
    unsigned sourcesCount() const { return static_cast<unsigned>(sources.size()); }

    /// Get the size of the target collection.
    unsigned targetsCount() const { return static_cast<unsigned>(targets.size()); }

    /// Get the types of the source collection.
    auto getSourceTypes() const { return llvm::map_range(sources, [](auto &e) { return e.root; }); }

    /// Get the types of the target collection.
    auto getTargetTypes() const { return llvm::map_range(targets, [](auto &e) { return e.root; }); }

private:
    std::deque<Source> sources;
    std::deque<Target> targets;
};

}  // namespace P4::P4MLIR::IRUtils

#endif  // P4MLIR_IMPL_UTILS_LOCATION_MAPPING_H
