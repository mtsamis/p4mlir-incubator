#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-remove-soft-cf"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_REMOVESOFTCF
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct RemoveSoftCFPass : public P4::P4MLIR::impl::RemoveSoftCFBase<RemoveSoftCFPass> {
    RemoveSoftCFPass() = default;
    void runOnOperation() override;
};

struct RemoveSoftCF {
    RemoveSoftCF(mlir::RewriterBase &rewriter, mlir::FunctionOpInterface root)
        : rewriter(rewriter), root(root) {}

    void transform() {
        if (!root.getCallableRegion()) return;

        mlir::Block *rootBlock = &root.getCallableRegion()->front();

        // Create a guard and a variable to store the final return value.
        auto loc = mlir::UnknownLoc::get(rewriter.getContext());
        rewriter.setInsertionPointToStart(rootBlock);
        returnGuard.init(rewriter, loc, "return_guard");

        auto returnType = mlir::cast<P4HIR::FuncType>(root.getFunctionType()).getReturnType();

        if (!mlir::isa<P4HIR::VoidType>(returnType))
            returnVar = rewriter.create<P4HIR::VariableOp>(
                loc, P4HIR::ReferenceType::get(returnType), "return_value");

        // Analyze and replace soft control flow in the function's body.
        CFInfo rootInfo = visitBlock(rootBlock);
        assert(!rootInfo.hasControlFlow(CF_Break | CF_Continue) &&
               "Break and Continue statements cannot be outside loops");

        // Replace any remaining soft control flow statements. We don't need to update any guard
        // variables, as they're guaranteed to not used in any guard block.
        replaceSoftControlFlow(rootInfo);

        // Update the real return statement.
        if (returnVar) {
            auto returnOp = mlir::cast<P4HIR::ReturnOp>(rootBlock->getTerminator());
            rewriter.setInsertionPoint(returnOp);
            auto returnValue = rewriter.create<P4HIR::ReadOp>(returnOp.getLoc(), returnVar);
            rewriter.modifyOpInPlace(returnOp, [&]() {
                returnOp.getInputMutable().assign(mlir::ValueRange(returnValue));
            });
        }
    }

 private:
    // Supported soft control flow operations.
    enum ControlFlowType : unsigned {
        CF_None = 0U,
        CF_Return = 1U << 0U,
        CF_Break = 1U << 1U,
        CF_Continue = 1U << 2U
    };

    // Enum to describe the possible places that execution continues after the execution of an
    // operation or block:
    //   - None: Any code following is unreachable.
    //   - Next: Execution continues normally.
    //   - Nested: Execution continues in single nested point.
    //   - Multiple: Execution continues at multiple potential places.
    enum ExecutionType { ET_None, ET_Next, ET_Nested, ET_Multiple };

    // Struct holding control flow information for an operation or block.
    struct CFInfo {
        CFInfo() : cfTypes(CF_None), execType(ET_Next) {}

        // Constructor for control flow operations.
        CFInfo(ControlFlowType type, mlir::Operation *cfOp) : cfTypes(type), execType(ET_None) {
            assert(type != CF_None);
            cfOps.insert(cfOp);
        }

        void addControlFlowOps(const CFInfo &cf) {
            cfOps.insert(cf.cfOps.begin(), cf.cfOps.end());
            cfTypes |= cf.cfTypes;
        }

        // Return true if this operation or block has any control flow operations.
        bool hasControlFlow() const { return (cfTypes != CF_None); }

        // Return true if this operation or block has control flow operations of the provided types.
        bool hasControlFlow(unsigned types) const { return (cfTypes & types); }

        // Holds all nested control flow operations.
        llvm::SmallPtrSet<mlir::Operation *, 2> cfOps;
        // Bit mask of control flow types found in this operation or block.
        unsigned cfTypes;
        // Describes how execution continues after this operation or block is executed.
        ExecutionType execType;
        // If an operation: holds the nested insertion point when `cp` is ET_Nested.
        // If a block: holds the place that execution would continue after execution of the block.
        mlir::Operation *execPoint = {};
    };

    // Helper to create and update boolean guard variables.
    struct GuardVariable {
        void init(mlir::OpBuilder &b, mlir::Location loc, std::string name) {
            guardVar = b.create<P4HIR::VariableOp>(
                loc, P4HIR::ReferenceType::get(P4HIR::BoolType::get(b.getContext())), name);

            assign(b, loc, true);
        }

        void assign(mlir::OpBuilder &b, mlir::Location loc, bool value) {
            auto constTrue =
                b.create<P4HIR::ConstOp>(loc, P4HIR::BoolAttr::get(b.getContext(), value));
            b.create<P4HIR::AssignOp>(loc, constTrue, guardVar);
        }

        mlir::Block *createGuardedBlock(mlir::OpBuilder &b, mlir::Location loc) {
            auto cond = b.create<P4HIR::ReadOp>(loc, guardVar);
            auto ifOp = b.create<P4HIR::IfOp>(
                loc, cond, false,
                [&](mlir::OpBuilder &b, mlir::Location) { P4HIR::buildTerminatedBody(b, loc); });
            return &ifOp.getThenRegion().front();
        }

        P4HIR::VariableOp getVar() const { return guardVar; }
        operator bool() const { return static_cast<bool>(guardVar); }

     private:
        P4HIR::VariableOp guardVar;
    };

    // Helper to replace soft control flow operations described by `desc`.
    // If `returnGuard` is non-empty then it must be assigned to guard execution of operations
    // executed later on. If `seperateBreakGuard` is non-empty then it must be assigned to break
    // from loops.
    void replaceSoftControlFlow(CFInfo &info, GuardVariable returnGuard = {},
                                GuardVariable seperateBreakGuard = {}) {
        for (mlir::Operation *cfOp : info.cfOps) {
            rewriter.setInsertionPoint(cfOp);

            if (mlir::isa<P4HIR::SoftBreakOp>(cfOp)) {
                assert((returnGuard || seperateBreakGuard) && "Missing break guard");
                GuardVariable &breakGuard = seperateBreakGuard ? seperateBreakGuard : returnGuard;
                breakGuard.assign(rewriter, cfOp->getLoc(), false);
            } else if (auto softReturnOp = mlir::dyn_cast<P4HIR::SoftReturnOp>(cfOp)) {
                if (seperateBreakGuard) seperateBreakGuard.assign(rewriter, cfOp->getLoc(), false);
                if (returnGuard) returnGuard.assign(rewriter, cfOp->getLoc(), false);

                if (returnVar)
                    rewriter.create<P4HIR::AssignOp>(softReturnOp.getLoc(),
                                                     softReturnOp.getOperand(0), returnVar);
            } else {
                assert(mlir::isa<P4HIR::SoftContinueOp>(cfOp) && "Unexpected control flow op");
            }

            rewriter.eraseOp(cfOp);
        }

        info.cfOps.clear();
    }

    // Computes control flow information for all operations in a block.
    CFInfo visitBlock(mlir::Block *block) {
        CFInfo blockInfo;
        mlir::Operation *execPoint = &block->front();

        for (mlir::Operation &op : llvm::make_early_inc_range(*block)) {
            if (&op == block->getTerminator()) break;

            // Rest of the block is unreachable, erase ops.
            if (blockInfo.execType == ET_None) {
                SmallVector<Operation *> restOps;
                for (mlir::Operation &op :
                     llvm::make_range(op.getIterator(), block->getTerminator()->getIterator()))
                    restOps.push_back(&op);
                for (mlir::Operation *op : llvm::make_early_inc_range(llvm::reverse(restOps)))
                    rewriter.eraseOp(op);
                execPoint = nullptr;
                break;
            }

            // Create execution guard and convert ET_Multiple to ET_Nested.
            if (blockInfo.execType == ET_Multiple) {
                // Materialize guards on demand.
                // If we're in a loop `visitLoop` will take care of that instead.
                if (!isInLoop) replaceSoftControlFlow(blockInfo, returnGuard);

                rewriter.setInsertionPoint(execPoint);
                mlir::Block *guardedBlock = returnGuard.createGuardedBlock(rewriter, op.getLoc());

                blockInfo.execType = ET_Nested;
                execPoint = &guardedBlock->front();
            }

            if (blockInfo.execType == ET_Nested)
                rewriter.moveOpBefore(&op, execPoint);
            else if (blockInfo.execType == ET_Next)
                execPoint = execPoint->getNextNode();

            auto opInfo = visitOp(block, &op);

            if (opInfo.hasControlFlow()) {
                assert(opInfo.execType != ET_Next);
                blockInfo.addControlFlowOps(opInfo);
                blockInfo.execType = opInfo.execType;

                if (opInfo.execType == ET_Nested) execPoint = opInfo.execPoint;
            }
        }

        blockInfo.execPoint = execPoint;

        return blockInfo;
    }

    // Computes control flow information for a conditional execution operation.
    // It must hold that exactly one of `blocks` is executed each time.
    CFInfo visitConditional(SmallVectorImpl<mlir::Block *> &blocks) {
        // The CFContinuePoint for the statement is:
        //  - None, if all blocks are None.
        //  - Nested, if a single block is Next/Nested and all others are None.
        //  - Multiple, in all other cases.
        CFInfo condInfo;
        condInfo.execType = ET_None;
        for (mlir::Block *block : blocks) {
            CFInfo caseInfo = visitBlock(block);
            condInfo.addControlFlowOps(caseInfo);

            if ((condInfo.execType == ET_None) &&
                (caseInfo.execType == ET_Next || caseInfo.execType == ET_Nested)) {
                condInfo.execPoint = caseInfo.execPoint;
                condInfo.execType = ET_Nested;
            } else if (caseInfo.execType != ET_None) {
                condInfo.execType = ET_Multiple;
            }
        }

        return condInfo;
    }

    // Computes control flow information for a loop like operation where `block` is the loop's body.
    // `breakGuardBuilder` should transform `loop` so that the given break guard is inserted.
    CFInfo visitLoop(mlir::Operation *loop, mlir::Block *block,
                     llvm::function_ref<void(GuardVariable &)> breakGuardBuilder) {
        bool origIsInLoop = isInLoop;
        isInLoop = true;

        CFInfo forInfo;
        CFInfo bodyInfo = visitBlock(block);

        // All break and continue operations will be replaced, so the loops continue point is
        // defined by the presence of return.
        if (bodyInfo.hasControlFlow(CF_Return)) {
            forInfo.cfTypes = CF_Return;
            forInfo.execType = ET_Multiple;
        } else {
            forInfo.execType = ET_Next;
        }

        // We only need to insert a loop guard if a break or return statement is found in the body.
        // Continue statements are implicitly implemented by the restructuring done by visitBlock.
        GuardVariable seperateBreakGuard;
        if (bodyInfo.hasControlFlow(CF_Break | CF_Return)) {
            if (bodyInfo.hasControlFlow(CF_Break)) {
                rewriter.setInsertionPoint(loop);
                seperateBreakGuard.init(rewriter, loop->getLoc(), "loop_break_guard");
                breakGuardBuilder(seperateBreakGuard);
            } else {
                // If we only have returns then we reuse the return guard for breaking the loop.
                breakGuardBuilder(returnGuard);
            }
        }

        // Replace all control flow operations within the loop.
        replaceSoftControlFlow(bodyInfo, returnGuard, seperateBreakGuard);

        isInLoop = origIsInLoop;

        return forInfo;
    }

    // Computes control flow information for an operation.
    CFInfo visitOp(mlir::Block *block, mlir::Operation *op) {
        if (mlir::isa<P4HIR::SoftReturnOp>(op)) return CFInfo{CF_Return, op};
        if (mlir::isa<P4HIR::SoftBreakOp>(op)) return CFInfo{CF_Break, op};
        if (mlir::isa<P4HIR::SoftContinueOp>(op)) return CFInfo{CF_Continue, op};

        if (auto scopeOp = mlir::dyn_cast<P4HIR::ScopeOp>(op))
            return visitBlock(&scopeOp.getScopeRegion().front());

        if (auto ifOp = mlir::dyn_cast<P4HIR::IfOp>(op)) {
            // Create else block if non existent.
            mlir::Region *elseRegion = &ifOp.getElseRegion();
            if (elseRegion->empty()) {
                mlir::Block *block = rewriter.createBlock(elseRegion, elseRegion->begin());
                rewriter.setInsertionPointToStart(block);
                rewriter.create<P4HIR::YieldOp>(ifOp.getLoc());
            }

            SmallVector<mlir::Block *, 2> blocks = {&ifOp.getThenRegion().front(),
                                                    &ifOp.getElseRegion().front()};
            return visitConditional(blocks);
        }

        if (auto switchOp = mlir::dyn_cast<P4HIR::SwitchOp>(op)) {
            // Create default case if non existent.
            if (!switchOp.getDefaultCase()) {
                rewriter.setInsertionPoint(switchOp.getBody().back().getTerminator());
                auto loc = switchOp.getLoc();
                rewriter.create<P4HIR::CaseOp>(
                    loc, mlir::ArrayAttr::get(rewriter.getContext(), {}),
                    P4HIR::CaseOpKind::Default,
                    [&](mlir::OpBuilder &b, mlir::Location) { b.create<P4HIR::YieldOp>(loc); });
            }

            auto blocks =
                llvm::to_vector(llvm::map_range(switchOp.cases(), [](P4HIR::CaseOp caseOp) {
                    return &caseOp.getCaseRegion().front();
                }));
            return visitConditional(blocks);
        }

        if (auto forOp = mlir::dyn_cast<P4HIR::ForOp>(op)) {
            mlir::Block *bodyBlock = &forOp.getBodyRegion().front();
            return visitLoop(forOp, bodyBlock, [&](GuardVariable &breakGuard) {
                mlir::Block *condBlock = &forOp.getCondRegion().front();
                auto cond = mlir::cast<P4HIR::ConditionOp>(condBlock->getTerminator());
                rewriter.setInsertionPoint(cond);
                auto loc = cond.getLoc();
                auto newCond = rewriter.create<P4HIR::TernaryOp>(
                    loc, cond.getCondition(),
                    [&](mlir::OpBuilder &b, mlir::Location) {
                        auto breakCond = b.create<P4HIR::ReadOp>(loc, breakGuard.getVar());
                        b.create<P4HIR::YieldOp>(loc, mlir::ValueRange(breakCond));
                    },
                    [&](mlir::OpBuilder &b, mlir::Location) {
                        auto constFalse = b.create<P4HIR::ConstOp>(
                            loc, P4HIR::BoolAttr::get(rewriter.getContext(), false));
                        b.create<P4HIR::YieldOp>(loc, mlir::ValueRange(constFalse));
                    });

                rewriter.modifyOpInPlace(
                    cond, [&]() { cond.getConditionMutable().assign(newCond.getResult()); });
            });
        }

        if (auto forInOp = mlir::dyn_cast<P4HIR::ForInOp>(op)) {
            mlir::Block *bodyBlock = &forInOp.getBodyRegion().front();
            return visitLoop(forInOp, bodyBlock, [&](GuardVariable &breakGuard) {
                mlir::Operation *firstOp = &bodyBlock->front();
                rewriter.setInsertionPointToStart(bodyBlock);
                mlir::Block *guardedBlock =
                    breakGuard.createGuardedBlock(rewriter, forInOp.getLoc());

                auto moveOps = llvm::make_range(firstOp->getIterator(),
                                                bodyBlock->getTerminator()->getIterator());
                for (mlir::Operation &op : llvm::make_early_inc_range(moveOps))
                    rewriter.moveOpBefore(&op, guardedBlock->getTerminator());
            });
        }

        // Normal operation without nested control flow.
        [[maybe_unused]] auto checkNestedCF = [](mlir::Operation *op) -> mlir::WalkResult {
            if (mlir::isa<P4HIR::SoftReturnOp, P4HIR::SoftBreakOp, P4HIR::SoftContinueOp>(op))
                return WalkResult::interrupt();
            return WalkResult::advance();
        };
        assert(!op->walk(checkNestedCF).wasInterrupted() && "Unexpected control flow in operation");

        return CFInfo{};
    }

    mlir::RewriterBase &rewriter;
    mlir::FunctionOpInterface root;

    GuardVariable returnGuard;
    P4HIR::VariableOp returnVar;
    bool isInLoop = false;
};

void RemoveSoftCFPass::runOnOperation() {
    mlir::IRRewriter rewriter(&getContext());

    getOperation()->walk([&](mlir::Operation *op) {
        if (mlir::isa<P4HIR::FuncOp, P4HIR::ControlOp>(op))
            RemoveSoftCF(rewriter, mlir::cast<mlir::FunctionOpInterface>(op)).transform();
    });
}
}  // end anonymous namespace

std::unique_ptr<Pass> P4::P4MLIR::createRemoveSoftCFPass() {
    return std::make_unique<RemoveSoftCFPass>();
}
