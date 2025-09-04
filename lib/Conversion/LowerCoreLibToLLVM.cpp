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
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Types.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "core-lib-lower-to-llvm"

using namespace mlir;
using namespace P4::P4MLIR;
using namespace P4::P4ToLLVM;

namespace {}  // end anonymous namespace

void P4HIR::populateCoreLibToLLVMConversionPatterns(P4LLVMTypeConverter &converter,
                                                    mlir::RewritePatternSet &patterns) {
    // patterns.add<>(converter);
}

void P4HIR::configureCoreLibToLLVMTypeConverter(P4LLVMTypeConverter &converter) {
    converter.addConversion([&](P4CoreLib::PacketInType packetInType) {
        return LLVM::LLVMPointerType::get(packetInType.getContext());
    });

    converter.addConversion([&](P4CoreLib::PacketOutType packetOutType) {
        return LLVM::LLVMPointerType::get(packetOutType.getContext());
    });

    // Ad-hoc and temporary for proof-of-concept.
    auto somePtr = LLVM::LLVMPointerType::get(&converter.getContext());
    auto someInt = mlir::IntegerType::get(&converter.getContext(), 32);
    auto packetInObj = P4Obj::createAggregate(&converter.getContext(), "_core_packet_in", {somePtr, someInt});
    converter.registerObjType(std::move(packetInObj));
}
