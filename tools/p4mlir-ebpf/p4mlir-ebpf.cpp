#include <cassert>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "p4mlir/Conversion/Passes.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4HIR/P4HIRToLLVMIRTranslation.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"
#include "p4mlir/Transforms/Passes.h"

using namespace P4::P4MLIR;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input p4hir file>"),
                                          cl::init("-"), cl::value_desc("filename"));

static cl::opt<std::string> outputFilename("o", cl::desc("<output file>"), cl::init("-"),
                                           cl::value_desc("filename"));

int loadMLIR(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module) {
    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);

    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return -1;
    }

    return 0;
}

// Run any passes necessary before translation to LLVM IR.
int processMLIR(mlir::OwningOpRef<mlir::ModuleOp> &module) {
    mlir::PassManager pm(module.get()->getName());
    // Apply any generic pass manager command line options and run the pipeline.
    if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) return -1;

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(createLowerToP4CoreLibPass());
    pm.addPass(createSerEnumEliminationPass());
    pm.addPass(createFlattenCFGPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(createLowerToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());

    if (mlir::failed(pm.run(*module))) return -1;

    return 0;
}

// Convert the module to LLVM IR in a new LLVM IR context.
std::unique_ptr<llvm::Module> convertToLLVMIR(llvm::LLVMContext &llvmContext,
                                              mlir::ModuleOp module) {
    auto llvmMod = mlir::translateModuleToLLVMIR(module, llvmContext);

    if (!llvmMod) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return {};
    }

    bool enableOpt = true;
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);
    if (auto err = optPipeline(llvmMod.get())) {
        llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
        return {};
    }

    return llvmMod;
}

int main(int argc, char **argv) {
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "p4hir-ebpf compiler\n");

    mlir::DialectRegistry registry;
    // P4HIR::registerP4CoreLibDialectTranslation(registry);
    P4HIR::registerP4HIRDialectTranslation(registry);
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    mlir::OwningOpRef<mlir::ModuleOp> mod;
    if (int error = loadMLIR(context, mod)) return error;

    llvm::dbgs() << "Loaded module:\n";
    mod->dump();

    if (int error = processMLIR(mod)) return error;

    llvm::dbgs() << "After processing:\n";
    mod->dump();

    llvm::LLVMContext llvmContext;
    auto llvmMod = convertToLLVMIR(llvmContext, *mod);
    if (!llvmMod) return -1;

    llvm::dbgs() << "LLVM IR:\n";
    llvm::dbgs() << *llvmMod << "\n";

    return 0;
}
