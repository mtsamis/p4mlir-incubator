//===----------------------------------------------------------------------===//
//
// This fle contains the declarations to register conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef P4MLIR_CONVERSION_PASSES_H
#define P4MLIR_CONVERSION_PASSES_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "p4mlir/Conversion/P4ToLLVM.h"

namespace mlir {
class LLVMTypeConverter;
class TypeConverter;
class DialectRegistry;
class RewritePatternSet;
}  // namespace mlir

namespace P4::P4MLIR {

#define GEN_PASS_DECL_LOWERTOP4CORELIB
#define GEN_PASS_DECL_LOWERP4HIRTOLLVM
#define GEN_PASS_DECL_LOWEREBPFTOLLVM
#include "p4mlir/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createLowerToP4CoreLibPass();
std::unique_ptr<mlir::Pass> createLowerP4HIRToLLVMPass();
std::unique_ptr<mlir::Pass> createLowerEBPFToLLVMPass();

// TODO move these elsewhere.
namespace P4HIR {
void populateP4HIRToLLVMConversionPatterns(P4::P4ToLLVM::P4LLVMTypeConverter &converter,
                                           mlir::RewritePatternSet &patterns);
void configureP4HIRToLLVMTypeConverter(P4::P4ToLLVM::P4LLVMTypeConverter &converter);

void populateCoreLibToLLVMConversionPatterns(P4::P4ToLLVM::P4LLVMTypeConverter &converter,
                                           mlir::RewritePatternSet &patterns);
void configureCoreLibToLLVMTypeConverter(P4::P4ToLLVM::P4LLVMTypeConverter &converter);
}  // namespace P4HIR

// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "p4mlir/Conversion/Passes.h.inc"

}  // namespace P4::P4MLIR

#endif  // P4MLIR_CONVERSION_PASSES_H
