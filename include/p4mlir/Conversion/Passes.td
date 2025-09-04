//===----------------------------------------------------------------------===//
//
// This file contains definitions for all dialect conversions.
//
//===----------------------------------------------------------------------===//

#ifndef P4MLIR_CONVERSION_PASSES_TD
#define P4MLIR_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerToP4CoreLib : Pass<"lower-to-p4corelib", "mlir::ModuleOp"> {
  let summary = "Convert P4HIR extern calls to P4 corelib ops";
  let description = [{
    This pass converts explicit P4HIR extern calls into P4 corelib operations
  }];
  let dependentDialects = [
      "::P4::P4MLIR::P4HIR::P4HIRDialect",
      "::P4::P4MLIR::P4CoreLib::P4CoreLibDialect",
  ];
}

def LowerP4HIRToLLVM : Pass<"lower-p4hir-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert the P4HIR ops to P4HIR ops with LLVM dialect";
  let constructor = "::P4::P4MLIR::createLowerP4HIRToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def LowerToLLVM : Pass<"lower-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert the P4HIR and CoreLib to the LLVM dialect";
  let constructor = "::P4::P4MLIR::createLowerToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

#endif // P4MLIR_CONVERSION_PASSES_TD
