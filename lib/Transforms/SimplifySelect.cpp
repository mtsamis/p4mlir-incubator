#include <llvm/Support/ErrorHandling.h>

#include "p4mlir/Transforms/Passes.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "p4hir-simplify-select"

namespace P4::P4MLIR {
#define GEN_PASS_DEF_SIMPLIFYSELECT
#include "p4mlir/Transforms/Passes.cpp.inc"

namespace {

struct SimplifySelect : public impl::SimplifySelectBase<SimplifySelect> {
    void runOnOperation() override;
};

class SelectCaseElim : public mlir::OpRewritePattern<P4HIR::ParserTransitionSelectOp> {
    using OpRewritePattern<P4HIR::ParserTransitionSelectOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ParserTransitionSelectOp selectOp,
                                        mlir::PatternRewriter &rewriter) const override {
        mlir::Block *body = &selectOp.getBody().back();
        auto splitPoint = body->begin();
        size_t newCasesCount = 0;
        bool seenDefault = false;
        bool hasUnreachableCases = false;

        for (auto selectCase : selectOp.selects()) {
            if (seenDefault) {
                hasUnreachableCases = true;
                break;
            }
            
            ++newCasesCount;
            ++splitPoint;

            auto caseYieldOp = mlir::dyn_cast<P4HIR::YieldOp>(selectCase.getRegion().back().getTerminator());
            bool isDefaultCase = mlir::isa<P4HIR::UniversalSetOp>(caseYieldOp.getArgs()[0].getDefiningOp());

            if (isDefaultCase) {
                seenDefault = true;
            }
        }

        if (newCasesCount == 1 && seenDefault) {
            // Handle select with single default case.
            auto firstCase = *selectOp.selects().begin();
            rewriter.replaceOpWithNewOp<P4HIR::ParserTransitionOp>(selectOp, firstCase.getState());

            return mlir::success();
        } else if (hasUnreachableCases) {
            // Only keep cases that are before splitPoint.
            mlir::Block *rest = rewriter.splitBlock(body, splitPoint);
            rewriter.eraseBlock(rest);

            return mlir::success();
        } else {
            return mlir::failure();
        }
    }
};

}  // end namespace

void SimplifySelect::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());

    patterns.add<SelectCaseElim>(patterns.getContext());

    walkAndApplyPatterns(getOperation(), std::move(patterns));
}

std::unique_ptr<mlir::Pass> createSimplifySelectPass() { return std::make_unique<SimplifySelect>(); }
}  // namespace P4::P4MLIR
