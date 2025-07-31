// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-select-flatten-tuples"

namespace P4::P4MLIR {
#define GEN_PASS_DEF_SELECTFLATTENTUPLES
#include "p4mlir/Transforms/Passes.cpp.inc"

namespace {

struct SelectFlattenTuples : public impl::SelectFlattenTuplesBase<SelectFlattenTuples> {
    void runOnOperation() override;
};

// Flatten tuples in transition_select arguments and case keysets.
class FlattenTuples : public mlir::OpRewritePattern<P4HIR::ParserTransitionSelectOp> {
 public:
    using OpRewritePattern<P4HIR::ParserTransitionSelectOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ParserTransitionSelectOp op,
                                        mlir::PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        llvm::SmallVector<mlir::Value> newArgs;
        llvm::SmallVector<unsigned> tupleReplacementCount;
        bool hasTupleArgs = false;

        // Compute unpacked tuple arguments.
        for (mlir::Value arg : op.getArgs()) {
            auto tupleType = mlir::dyn_cast<mlir::TupleType>(arg.getType());
            if (!tupleType) {
                tupleReplacementCount.push_back(0);
                newArgs.push_back(arg);
                continue;
            }

            hasTupleArgs = true;
            unsigned tupleSize = tupleType.size();
            tupleReplacementCount.push_back(tupleSize);
            for (unsigned i = 0; i < tupleSize; i++)
                newArgs.push_back(rewriter.createOrFold<P4HIR::TupleExtractOp>(loc, arg, i));
        }

        if (!hasTupleArgs)
            return rewriter.notifyMatchFailure(loc, "Select doesn't have tuple arguments.");

        // Update select arguments.
        rewriter.modifyOpInPlace(op, [&]() { op.getArgsMutable().assign(newArgs); });

        for (auto selectCase : op.selects()) {
            if (selectCase.isDefault()) continue;

            llvm::SmallVector<mlir::Value> newYieldArgs;
            auto yield = mlir::cast<P4HIR::YieldOp>(selectCase.getTerminator());
            rewriter.setInsertionPoint(yield);

            // Compute unpacked yield arguments.
            for (auto [arg, replacementCount] :
                 llvm::zip_equal(yield.getArgs(), tupleReplacementCount)) {
                if (replacementCount == 0) {
                    // Argument doesn't correspond to a tuple argument.
                    newYieldArgs.push_back(arg);
                    continue;
                }

                if (P4HIR::isUniversalSetValue(arg)) {
                    // Universal sets are untyped, so just copy the argument for each field.
                    for (unsigned i = 0; i < replacementCount; i++) newYieldArgs.push_back(arg);
                } else {
                    // Get the tuple value to unpack.
                    mlir::Value tupleVal;
                    if (auto setOp = arg.getDefiningOp<P4HIR::SetOp>()) {
                        assert((setOp.getInput().size() == 1) && "Unexpected set argument count");
                        tupleVal = setOp.getInput()[0];
                    } else {
                        auto constSetOp = arg.getDefiningOp<P4HIR::ConstOp>();
                        assert(constSetOp &&
                               "A set of tuple value can only be p4hir.set or p4hir.const");
                        auto setAttr = constSetOp.getValueAs<P4HIR::SetAttr>();
                        assert((setAttr.getKind() == P4HIR::SetKind::Constant) &&
                               "Invalid constant set kind");
                        auto tupleAttr = mlir::cast<mlir::TypedAttr>(setAttr.getMembers()[0]);
                        tupleVal = P4HIR::ConstOp::create(rewriter, arg.getLoc(), tupleAttr);
                    }
                    assert((mlir::isa<mlir::TupleType>(tupleVal.getType())) &&
                           "Expected tuple type");

                    // Unpack to individual field sets.
                    for (unsigned i = 0; i < replacementCount; i++) {
                        auto field =
                            rewriter.createOrFold<P4HIR::TupleExtractOp>(arg.getLoc(), tupleVal, i);
                        auto fieldSet = P4HIR::SetOp::create(rewriter, arg.getLoc(), field);
                        newYieldArgs.push_back(fieldSet);
                    }
                }

                // Update yield arguments.
                rewriter.modifyOpInPlace(op,
                                         [&]() { yield.getArgsMutable().assign(newYieldArgs); });
            }
        }

        return mlir::success();
    }
};

}  // end namespace

void SelectFlattenTuples::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FlattenTuples>(patterns.getContext());
    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed()) signalPassFailure();
}

std::unique_ptr<mlir::Pass> createSelectFlattenTuplesPass() {
    return std::make_unique<SelectFlattenTuples>();
}
}  // namespace P4::P4MLIR
