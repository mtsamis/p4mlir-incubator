#include "llvm/Support/Casting.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "p4mlir/Conversion/Passes.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-lower-to-llvm"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_LOWERP4HIRTOLLVM
#include "p4mlir/Conversion/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct LowerP4HIRToLLVMPass : public P4::P4MLIR::impl::LowerP4HIRToLLVMBase<LowerP4HIRToLLVMPass> {
    LowerP4HIRToLLVMPass() = default;
    void runOnOperation() override;
};

void LowerP4HIRToLLVMPass::runOnOperation() {
    mlir::ModuleOp mod = getOperation();

    mlir::LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    mlir::LLVMTypeConverter typeConverter(&getContext());

    mlir::RewritePatternSet patterns(&getContext());

    P4HIR::populateP4HIRToLLVMConversionPatterns(typeConverter, patterns);

    mlir::populateSCFToControlFlowConversionPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateSCFToControlFlowConversionPatterns(patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    if (mlir::failed(mlir::applyPartialConversion(mod, target, std::move(patterns))))
        signalPassFailure();
}
}  // end anonymous namespace

void P4HIR::populateP4HIRToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                                  mlir::RewritePatternSet &patterns) {}

std::unique_ptr<Pass> P4::P4MLIR::createLowerP4HIRToLLVMPass() {
    return std::make_unique<LowerP4HIRToLLVMPass>();
}

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//
namespace {
/// Implement the interface to convert P4HIR to LLVM.
struct P4HIRToLLVMDialectInterface : public mlir::ConvertToLLVMPatternInterface {
    using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
    void loadDependentDialects(MLIRContext *context) const final {
        context->loadDialect<P4HIR::P4HIRDialect>();
    }

    /// Hook for derived dialect interface to provide conversion patterns
    /// and mark dialect legal for the conversion target.
    void populateConvertToLLVMConversionPatterns(ConversionTarget &target,
                                                 LLVMTypeConverter &typeConverter,
                                                 RewritePatternSet &patterns) const final {
        P4HIR::populateP4HIRToLLVMConversionPatterns(typeConverter, patterns);
    }
};
}  // namespace

void P4HIR::registerConvertP4HIRToLLVMInterface(DialectRegistry &registry) {
    registry.addExtension(+[](MLIRContext *ctx, P4HIR::P4HIRDialect *dialect) {
        dialect->addInterfaces<P4HIRToLLVMDialectInterface>();
    });
}
