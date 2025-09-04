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
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Conversion/P4ToLLVM.h"
#include "p4mlir/Conversion/Passes.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "lower-to-llvm"

using namespace mlir;
using namespace P4::P4MLIR;
using namespace P4::P4ToLLVM;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_LOWERTOLLVM
#include "p4mlir/Conversion/Passes.cpp.inc"
}  // namespace P4::P4MLIR

namespace {
// This is a temporary construct; in the end we should have a eBPF epcific pass that adds the P4HIR
// and CoreLib conversion plus anything else needed.
struct LowerToLLVMPass : public P4::P4MLIR::impl::LowerToLLVMBase<LowerToLLVMPass> {
    LowerToLLVMPass() = default;
    void runOnOperation() override;
};

void LowerToLLVMPass::runOnOperation() {
    mlir::ModuleOp mod = getOperation();

    mlir::LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    P4LLVMTypeConverter typeConverter(&getContext());
    configureP4HIRTypeConverter(typeConverter);
    P4HIR::configureP4HIRToLLVMTypeConverter(typeConverter);
    P4HIR::configureCoreLibToLLVMTypeConverter(typeConverter);

    mlir::RewritePatternSet patterns(&getContext());
    P4HIR::populateP4HIRToLLVMConversionPatterns(typeConverter, patterns);
    P4HIR::populateCoreLibToLLVMConversionPatterns(typeConverter, patterns);

    if (mlir::failed(mlir::applyPartialConversion(mod, target, std::move(patterns))))
        signalPassFailure();
}
}  // end anonymous namespace

std::unique_ptr<Pass> P4::P4MLIR::createLowerToLLVMPass() {
    return std::make_unique<LowerToLLVMPass>();
}
