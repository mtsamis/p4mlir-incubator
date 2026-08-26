// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Types.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/IRUtils.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-flatten-header-structs"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_FLATTENHEADERSTRUCTS
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct FlattenHeaderStructsPass
    : public P4::P4MLIR::impl::FlattenHeaderStructsBase<FlattenHeaderStructsPass> {
    void runOnOperation() override;
};

struct FlattenHeaderStructsRewriter : public IRUtils::IndexableValueRewriter {
    using IndexableValueRewriter::IndexableValueRewriter;

    P4HIR::FieldPath getNewPath(P4HIR::FieldPath hPath, P4HIR::FieldPath fPath) {
        auto newHeaderPath = hPath.withRoot(newRootType);
        auto newFieldName = fPath.getIdentifier("_") + "$";
        return newHeaderPath[newFieldName];
    }

    mlir::LogicalResult replaceUsesInRead(P4HIR::ReadOp op, P4HIR::FieldPath prefixPath,
                                          P4HIR::FieldPath suffixPath) {
        // We're reading a field that will not exist after flattening.
        // Replace it by a synthesized value of individual reads.
        auto loc = op.getLoc();
        std::function<mlir::Value(P4HIR::FieldPath)> synthReadValue =
            [&](P4HIR::FieldPath path) -> mlir::Value {
            if (auto itype = mlir::dyn_cast<P4HIR::IndexableTypeInterface>(path.getType())) {
                auto newFields = llvm::map_to_vector(
                    itype.getFields(), [&](auto field) { return synthReadValue(path[field]); });

                if (auto stype = mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(itype))
                    return P4HIR::StructOp::create(rewriter, loc, stype, newFields);
                else if (auto atype = mlir::dyn_cast<P4HIR::ArrayType>(itype))
                    return P4HIR::StructOp::create(rewriter, loc, atype, newFields);

                llvm_unreachable("Impossible indexable type");
                return mlir::Value();
            } else {
                mlir::Value newRef = getFromPath(loc, newRoot, getNewPath(prefixPath, path));
                return P4HIR::ReadOp::create(rewriter, loc, newRef);
            }
        };
        rewriter.replaceOp(op, synthReadValue(suffixPath));
        return mlir::success();
    }

    mlir::LogicalResult replaceUsesInAssign(P4HIR::AssignOp op, P4HIR::FieldPath prefixPath,
                                            P4HIR::FieldPath suffixPath) {
        // We're writing to a field that will not exist after flattening.
        // Replace it by a sequential assignments to individual fields.
        auto loc = op.getLoc();
        std::function<void(P4HIR::FieldPath, mlir::Value)> assignValue = [&](P4HIR::FieldPath path,
                                                                             mlir::Value value) {
            if (auto itype = mlir::dyn_cast<P4HIR::IndexableTypeInterface>(path.getType())) {
                for (auto field : itype.getFields())
                    assignValue(path[field], getFromField(loc, value, field));
            } else {
                mlir::Value newRef = getFromPath(loc, newRoot, getNewPath(prefixPath, path));
                P4HIR::AssignOp::create(rewriter, loc, value, newRef);
            }
        };

        assignValue(suffixPath, op.getValue());
        rewriter.eraseOp(op);
        return mlir::success();
    }

    virtual mlir::LogicalResult replaceUsesIn(mlir::Operation *op) override {
        auto [operand, path] = getOperandWithPath(op);
        auto [prefixPath, suffixPath] =
            path.split([](auto prefix) { return mlir::isa<P4HIR::HeaderType>(prefix.getType()); });

        if (suffixPath.isEmpty()) {
            // We're adjusting an operation on an field located outside a header.
            mlir::Value newAccess = getFromPath(op->getLoc(), newRoot, path);
            rewriter.modifyOpInPlace(op, [&]() { operand->set(newAccess); });
            return mlir::success();
        }

        if (auto readOp = mlir::dyn_cast<P4HIR::ReadOp>(op)) {
            return replaceUsesInRead(readOp, prefixPath, suffixPath);
        } else if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(op)) {
            return replaceUsesInAssign(assignOp, prefixPath, suffixPath);
        } else {
            return op->emitOpError()
                   << "Cannot replace access to " << path.str() << " in operation";
        }
    }

    virtual mlir::Value getReplacement(mlir::Value value, P4HIR::FieldPath path) override {
        // We can only replace accesses to leaf fields when not rewriting operations.
        if (mlir::isa<P4HIR::IndexableTypeInterface>(path.getType())) return mlir::Value();

        auto [prefixPath, suffixPath] =
            path.split([](auto prefix) { return mlir::isa<P4HIR::HeaderType>(prefix.getType()); });
        if (suffixPath.isEmpty()) return mlir::Value();

        return getFromPath(value.getLoc(), newRoot, getNewPath(prefixPath, suffixPath));
    }
};

static P4HIR::HeaderType getFlattenedHeaderType(P4HIR::HeaderType headerType) {
    auto *ctx = headerType.getContext();
    bool hasFieldsToFlatten = llvm::any_of(headerType.getFields(), [](auto field) {
        return mlir::isa<P4HIR::IndexableTypeInterface>(field.getType());
    });
    if (!hasFieldsToFlatten) return headerType;

    llvm::SmallVector<P4HIR::FieldPath> flattenedFields;
    P4HIR::FieldPath::forEachFieldPath(headerType, [&](auto path) {
        bool isLeaf = !mlir::isa<P4HIR::IndexableTypeInterface>(path.getType());
        if (isLeaf) flattenedFields.push_back(path);
    });

    assert((mlir::isa<P4HIR::ValidBitType>(flattenedFields.back().getType())) &&
           "Expected header valid bit");
    flattenedFields.pop_back();

    auto newFields = llvm::map_to_vector(flattenedFields, [&](auto fieldPath) {
        auto nameAttr = mlir::StringAttr::get(ctx, fieldPath.getIdentifier("_") + "$");
        return P4HIR::FieldInfo(nameAttr, fieldPath.getType());
    });

    return P4HIR::HeaderType::get(ctx, headerType.getName(), newFields);
}

void FlattenHeaderStructsPass::runOnOperation() {
    P4HIRTypeConverter converter;
    converter.addConversion(getFlattenedHeaderType);

    mlir::MLIRContext *ctx = &getContext();
    mlir::IRRewriter rewriter(ctx);

    for (auto rootOp : llvm::make_early_inc_range(getOperation().getOps<P4HIR::ParserOp>())) {
        auto origArgTypes = llvm::to_vector(rootOp.getArgumentTypes());
        rewriter.setInsertionPoint(rootOp);
        auto result =
            P4::P4MLIR::doPlainTypeConversion(rootOp, mlir::ValueRange(), &converter, rewriter);
        if (failed(result)) return signalPassFailure();

        rootOp = mlir::cast<P4HIR::ParserOp>(*result);
        mlir::Region &region = *rootOp.getCallableRegion();

        // Rewrite arguments.
        for (auto [arg, origArgType] : llvm::zip_equal(rootOp.getArguments(), origArgTypes)) {
            if (arg.getType() == origArgType) continue;

            if (failed(FlattenHeaderStructsRewriter(rewriter, arg, origArgType).replace()))
                return signalPassFailure();
        }

        if (failed(mlir::simplifyRegions(rewriter, region, false))) return signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<Pass> P4::P4MLIR::createFlattenHeaderStructsPass() {
    return std::make_unique<FlattenHeaderStructsPass>();
}
