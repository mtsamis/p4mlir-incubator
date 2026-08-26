// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
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

struct FlattenHeaderStructsRewriter : public IRUtils::PathRewriter {
    using PathRewriter::PathRewriter;

    // If (hPath ++ fPath) is accesing a header leaf field that is flattened, return a new field
    // path for that field relative to `newRootType`.
    P4HIR::FieldPath getNewPath(P4HIR::FieldPath hPath, P4HIR::FieldPath fPath) {
        auto newHeaderPath = hPath.withRoot(newRootType);
        auto newFieldName = fPath.getIdentifier("_") + "$";
        if (!newHeaderPath.try_append(newFieldName)) return P4HIR::FieldPath();

        return newHeaderPath;
    }

    // If path is accesing a header leaf field that is flattened, return a new field path for that
    // field relative to `newRootType`.
    P4HIR::FieldPath getNewPath(P4HIR::FieldPath path) {
        auto [prefixPath, suffixPath] =
            path.split([](auto prefix) { return mlir::isa<P4HIR::HeaderType>(prefix.getType()); });
        return getNewPath(prefixPath, suffixPath);
    }

    // Return a (potentially synthesized) value to access `prefixPath.suffixPath` from `newRoot`.
    mlir::Value getValue(mlir::Location loc, P4HIR::FieldPath prefixPath,
                         P4HIR::FieldPath suffixPath) {
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
                mlir::Value newAccess = getFromPath(loc, newRoot, getNewPath(prefixPath, path));
                if (isRef)
                    return P4HIR::ReadOp::create(rewriter, loc, newAccess);
                else
                    return newAccess;
            }
        };
        return synthReadValue(suffixPath);
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
        if (mlir::isa<mlir::UnrealizedConversionCastOp>(op)) {
            return success();
        }

        auto [operand, path] = getOperandWithPath(op);
        auto [prefixPath, suffixPath] =
            path.split([](auto prefix) { return mlir::isa<P4HIR::HeaderType>(prefix.getType()); });

        if (suffixPath.isEmpty() || mlir::isa<P4HIR::HeaderType>(suffixPath.getType())) {
            // We're adjusting an operation on an field located outside a header.
            mlir::Value newAccess = getFromPath(op->getLoc(), newRoot, path);
            rewriter.modifyOpInPlace(op, [&]() { operand->set(newAccess); });
            return mlir::success();
        }

        if (!isRef) {
            mlir::Value newValue = getValue(op->getLoc(), prefixPath, suffixPath);
            rewriter.modifyOpInPlace(op, [&]() { operand->set(newValue); });
            return mlir::success();
        }

        if (auto readOp = mlir::dyn_cast<P4HIR::ReadOp>(op)) {
            rewriter.replaceOp(op, getValue(op->getLoc(), prefixPath, suffixPath));
            return mlir::success();
        } else if (auto assignOp = mlir::dyn_cast<P4HIR::AssignOp>(op)) {
            // Replace assignment with potentially multiple assignments.
            return replaceUsesInAssign(assignOp, prefixPath, suffixPath);
        } else {
            return op->emitOpError()
                   << "Cannot replace access to " << path.str() << " in operation";
        }
    }

    virtual mlir::Value replaceValue(mlir::Value value, P4HIR::FieldPath path) override {
        if (auto newPath = getNewPath(path)) return getFromPath(value.getLoc(), newRoot, newPath);

        return mlir::Value();
    }

    mlir::LogicalResult replace() { return PathRewriter::replace(true); }
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

    auto mod = getOperation();
    mlir::MLIRContext *ctx = &getContext();
    mlir::IRRewriter rewriter(ctx);

    for (auto rootOp : llvm::make_early_inc_range(mod.getOps<mlir::FunctionOpInterface>())) {
        // Rewrite arguments.
        for (auto arg : rootOp.getArguments()) {
            auto flattenedType = converter.convertType(arg.getType());
            if (flattenedType != arg.getType()) {
                rewriter.setInsertionPointAfterValue(arg);
                auto cast =
                    UnrealizedConversionCastOp::create(rewriter, arg.getLoc(), flattenedType, arg);
                mlir::Value newArg = cast->getResult(0);
                if (failed(FlattenHeaderStructsRewriter(rewriter, arg, newArg).replace()))
                    return signalPassFailure();
            }
        }

        // Collect operations.
        llvm::SmallVector<std::pair<mlir::Operation *, mlir::Type>> opsToRewrite;
        rootOp.walk([&](mlir::Operation *op) {
            if (!mlir::isa<P4HIR::VariableOp>(op)) return;

            // All operations we care for generate a single result.
            auto type = op->getResult(0).getType();
            auto flattenedType = converter.convertType(type);
            if (flattenedType == type) return;

            opsToRewrite.emplace_back(op, flattenedType);
        });

        // Rewrite operations.
        for (auto [op, flattenedType] : opsToRewrite) {
            rewriter.setInsertionPoint(op);
            auto loc = op->getLoc();
            mlir::Value newResult;

            if (auto variableOp = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
                newResult = P4HIR::VariableOp::create(rewriter, loc, flattenedType,
                                                      variableOp.getName().value_or(""));
            } else {
                llvm_unreachable("Impossible op");
            }

            if (failed(
                    FlattenHeaderStructsRewriter(rewriter, op->getResult(0), newResult).replace()))
                return signalPassFailure();
        }
    }

    // Finally do dialect conversion. This will fix all types in operations, rewrite signatures and
    // eliminate unrelaized conversions.
    mlir::ConversionTarget target(*ctx);
    mlir::RewritePatternSet patterns(ctx);
    configureUnknownOpDynamicallyLegalByTypes(target, converter);
    populateTypeConversionPattern(patterns, converter);
    if (failed(mlir::applyPartialConversion(mod, target, std::move(patterns)))) signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> P4::P4MLIR::createFlattenHeaderStructsPass() {
    return std::make_unique<FlattenHeaderStructsPass>();
}
