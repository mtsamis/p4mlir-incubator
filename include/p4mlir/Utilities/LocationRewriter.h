// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_IMPL_UTILS_LOCATION_REWRITER_H
#define P4MLIR_IMPL_UTILS_LOCATION_REWRITER_H

#include "p4mlir/Utilities/IRUtils.h"
#include "p4mlir/Utilities/LocationMapping.h"
#include "p4mlir/Utilities/PathRewriter.h"
#include "p4mlir/Utilities/PathWalker.h"

namespace P4::P4MLIR::IRUtils {

/// A helper class to replace uses of a source collection to ones in a target collections of values
/// using a corresponding `LocationMapping`.
template <class FieldPathOpsT>
struct LocationRewriter {
    using Location = LocationMapping2::Location;
    using OperandWithLocationT = std::pair<mlir::OpOperand *, Location>;

    LocationRewriter(mlir::RewriterBase &rewriter, LocationMapping2 &mapping,
                     llvm::SmallVector<mlir::Value> from, llvm::SmallVector<mlir::Value> to)
        : rewriter(rewriter),
          mapping(mapping),
          fromValues(std::move(from)),
          toValues(std::move(to)) {
        assert((fromValues.size() == mapping.sourcesCount()) &&
               "Value and type size mismatch for the source collection");
        assert((toValues.size() == mapping.targetsCount()) &&
               "Value and type size mismatch for the target collection");
        for (auto [value, type] : llvm::zip_equal(fromValues, mapping.getSourceTypes()))
            assert((P4HIR::maybeUnref(value.getType()) == type) &&
                   "Value and type mismatch in the source collection");
        for (auto [value, type] : llvm::zip_equal(toValues, mapping.getTargetTypes()))
            assert((P4HIR::maybeUnref(value.getType()) == type) &&
                   "Value and type mismatch in the target collection");
    }

    virtual ~LocationRewriter() {}

    using SplitFieldLocation = std::tuple<unsigned, P4HIR::FieldPath, P4HIR::FieldPath>;
    std::optional<SplitFieldLocation> remap(Location from) {
        auto mappings = mapping.remap(from);
        assert(mappings.size() <= 1);
        if (mappings.size() == 0) return std::nullopt;
        return SplitFieldLocation(mappings[0].targetRoot.getIndex(),
                                  mappings[0].targetRoot.getPath(), mappings[0].targetField);
    }

    /// Return a remapped reference to access `from` at `where` if possible, otherwise return an
    /// empty value.
    mlir::Value getRef(mlir::Operation *where, mlir::Location loc, Location from) {
        auto to = remap(from);
        if (!to) return mlir::Value();

        auto [toIndex, toPath, toField] = *to;
        if (!mlir::isa<P4HIR::ReferenceType>(toValues[toIndex].getType())) return mlir::Value();

        auto fullPath = P4HIR::FieldPath::concat(toPath, toField);
        return PathRewriterT::buildPathAccess(rewriter, where, loc, toValues[toIndex], fullPath);
    }

    /// Return a remapped value to access `from` at `where` if possible, otherwise return an empty
    /// value. If `allowSplit` is true then multiple values from different places in the target
    /// collection may be combined to do so.
    mlir::Value getVal(mlir::Operation *where, mlir::Location loc, Location from,
                       bool allowSplit = true) {
        if (auto to = remap(from)) {
            auto [toIndex, toPath, toField] = *to;
            auto fullPath = P4HIR::FieldPath::concat(toPath, toField);
            mlir::Value res =
                PathRewriterT::buildPathAccess(rewriter, where, loc, toValues[toIndex], fullPath);
            if (mlir::isa<P4HIR::ReferenceType>(res.getType()))
                res = P4HIR::ReadOp::create(rewriter, loc, res);
            return res;
        } else {
            if (!allowSplit) return mlir::Value();

            unsigned index = from.getIndex();
            auto path = from.getPath();

            // Build the read value by combining individual values for its fields.
            if (auto structType = mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(path.getType())) {
                auto fields = llvm::map_to_vector(path.children(), [&](auto childPath) {
                    return getVal(where, loc, Location(childPath, index), allowSplit);
                });
                return P4HIR::StructOp::create(rewriter, loc, structType, fields);
            }

            return mlir::Value();
        }
    }

    /// Perform a remapped assignment to `from` at `where` if possible and return true, otherwise
    /// return false. If `allowSplit` is true then the assignment may be broken into multiple
    /// assignments in the target collection to do so.
    bool setVal(mlir::Operation *where, mlir::Location loc, Location from, mlir::Value val,
                bool allowSplit = true) {
        if (auto to = remap(from)) {
            auto [toIndex, toPath, toField] = *to;
            if (!mlir::isa<P4HIR::ReferenceType>(toValues[toIndex].getType())) return false;

            auto fullPath = P4HIR::FieldPath::concat(toPath, toField);
            mlir::Value res =
                PathRewriterT::buildPathAccess(rewriter, where, loc, toValues[toIndex], fullPath);
            P4HIR::AssignOp::create(rewriter, loc, val, res);
            return true;
        } else {
            if (!allowSplit) return false;

            unsigned index = from.getIndex();
            auto path = from.getPath();

            // Break down the assignment to the individual parts.
            if (auto structType = mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(path.getType())) {
                for (auto field : structType.getFields()) {
                    mlir::Value fieldVal =
                        P4HIR::StructExtractOp::create(rewriter, loc, val, field);
                    setVal(where, loc, Location(path[field], index), fieldVal, allowSplit);
                }
                return true;
            }

            return false;
        }
    }

    /// `op` is an operation that has at least one operand accessing fields of a source collection
    /// object. These operands and the accessed locations are provided in `operands`. If it is
    /// desirable to transform `op` then this should be done here. Return mlir::success() to
    /// continue the rewriting or mlir::failure() to interrupt it.
    virtual mlir::LogicalResult replaceUsesIn(mlir::Operation *op,
                                              llvm::SmallVector<OperandWithLocationT> operands) {
        if (auto readOp = mlir::dyn_cast<P4HIR::ReadOp>(op)) {
            assert((operands.size() == 1) && "Expected single argument replacement");
            auto [operand, from] = operands.front();
            auto newVal = getVal(op, op->getLoc(), from);
            if (!newVal) return op->emitError("Cannot replace read of ") << from.getPath().str();
            this->rewriter.replaceOp(op, newVal);
            return mlir::success();
        } else if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(op)) {
            assert((operands.size() == 1) && "Expected single argument replacement");
            auto [operand, from] = operands.front();
            bool success = setVal(op, op->getLoc(), from, assignOp.getValue());
            if (!success) return op->emitError("Cannot replace write to ") << from.getPath().str();
            this->rewriter.eraseOp(op);
            return mlir::success();
        } else {
            bool success = true;
            this->rewriter.modifyOpInPlace(op, [&]() {
                for (auto [operand, from] : operands) {
                    auto val = operand->get();
                    bool isRef = mlir::isa<P4HIR::ReferenceType>(val.getType());
                    auto newVal =
                        isRef ? getRef(op, val.getLoc(), from) : getVal(op, val.getLoc(), from);
                    if (!newVal) {
                        op->emitError("Cannot replace access of ") << from.getPath().str();
                        success = false;
                        break;
                    }
                    operand->set(newVal);
                }
            });
            return success ? mlir::success() : mlir::failure();
        }
    }

    /// Replace all uses of the source collection with ones from the target collection.
    mlir::LogicalResult transform() {
        for (unsigned index = 0, e = fromValues.size(); index < e; index++)
            if (failed(ValueRewriter(*this, index).transform(true))) return mlir::failure();

        return mlir::success();
    }

 protected:
    using PathRewriterT = PathRewriter<FieldPathOpsT>;

    struct ValueRewriter : public PathRewriterT {
        using OperandWithPathT = typename PathRewriterT::OperandWithPathT;

        ValueRewriter(LocationRewriter &parent, unsigned index)
            : PathRewriterT(parent.rewriter, parent.fromValues[index]),
              parent(parent),
              index(index) {}

        virtual mlir::LogicalResult replaceUsesIn(
            mlir::Operation *op, llvm::SmallVector<OperandWithPathT> operands) override {
            auto operandsWithLocations = llvm::map_to_vector(operands, [&](auto p) {
                auto [operand, path] = p;
                return OperandWithLocationT(operand, Location(path, index));
            });
            return parent.replaceUsesIn(op, operandsWithLocations);
        }

        LocationRewriter &parent;
        unsigned index;
    };

    mlir::RewriterBase &rewriter;
    LocationMapping2 &mapping;
    llvm::SmallVector<mlir::Value> fromValues;
    llvm::SmallVector<mlir::Value> toValues;
};

using P4HIRLocationRewriter = LocationRewriter<P4HIRFieldPathOps>;

}  // namespace P4::P4MLIR::IRUtils

#endif  // P4MLIR_IMPL_UTILS_LOCATION_REWRITER_H
