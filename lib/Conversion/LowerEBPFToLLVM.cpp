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
#define GEN_PASS_DEF_LOWEREBPFTOLLVM
#include "p4mlir/Conversion/Passes.cpp.inc"
}  // namespace P4::P4MLIR

namespace {
// This is a temporary construct; in the end we should have a eBPF epcific pass that adds the P4HIR
// and CoreLib conversion plus anything else needed.
struct LowerEBPFToLLVMPass : public P4::P4MLIR::impl::LowerEBPFToLLVMBase<LowerEBPFToLLVMPass> {
    LowerEBPFToLLVMPass() = default;
    void runOnOperation() override;
};

struct InstantiateMainConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::InstantiateOp> {
    InstantiateMainConversion(P4LLVMTypeConverter &converter)
        : ConvertOpToLLVMPattern(converter), converter(&converter) {}

    template <typename OpTy>
    static OpTy unwrapPackageArg(mlir::Value arg) {
        if (auto constructOp = mlir::dyn_cast<P4HIR::ConstructOp>(arg.getDefiningOp())) {
            auto mod = constructOp->getParentOfType<mlir::ModuleOp>();
            return mod.lookupSymbol<OpTy>(constructOp.getCallee());
        }

        return {};
    }

    // Create the entry point function for an ebpfFilter package.
    mlir::LogicalResult createEBPFFilter(mlir::ConversionPatternRewriter &rewriter,
                                         mlir::Location loc, P4HIR::ParserOp parser,
                                         P4HIR::ControlOp filter) const {
        // Our package looks like this:
        //   parser parse<H>(packet_in packet, out H headers);
        //   control filter<H>(inout H headers, out bool accept);
        //   package ebpfFilter<H>(parse<H> prs, filter<H> filt);

        // The signature is `bool ebpf_filter(ptr packetStart, u32 packetLen)`
        auto resTy = rewriter.getIntegerType(1);
        auto packetStartTy = rewriter.getType<LLVM::LLVMPointerType>();
        auto packetLenTy = rewriter.getIntegerType(32);
        auto funcType = LLVM::LLVMFunctionType::get(resTy, {packetStartTy, packetLenTy});

        auto func = rewriter.create<LLVM::LLVMFuncOp>(loc, "ebpf_filter", funcType);
        Block *entryBlock = func.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // TODO create/extend infrastructure for member functions in P4Obj?
        auto callApplyFn = [&](mlir::Operation *op, mlir::ValueRange args) {
            auto mod = op->getParentOfType<mlir::ModuleOp>();

            auto instFnName = specialInitFn(op);
            if (auto instFn = mod.lookupSymbol<LLVM::LLVMFuncOp>(instFnName)) {
                rewriter.create<LLVM::CallOp>(loc, instFn, mlir::ValueRange());
            }

            auto applyFnName = specialApplyFn(op);
            auto applyFn = mod.lookupSymbol<LLVM::LLVMFuncOp>(applyFnName);
            rewriter.create<LLVM::CallOp>(loc, applyFn, args);
        };

        auto parserObj = converter->getObjType(parser)->allocaLLVM(rewriter, loc);
        auto packetInObj = converter->getObjType(removeRef(parser.getArgumentTypes()[0]))
                               ->allocaLLVM(rewriter, loc);
        auto headerObj = converter->getObjType(removeRef(parser.getArgumentTypes()[1]))
                             ->allocaLLVM(rewriter, loc);

        auto parserInst = parserObj.getPtrLLVM(rewriter, loc);
        auto packetInInst = packetInObj.getPtrLLVM(rewriter, loc);
        auto headerInst = headerObj.getPtrLLVM(rewriter, loc);

        callApplyFn(parser, {parserInst, packetInInst, headerInst});

        auto filterObj = converter->getObjType(filter)->allocaLLVM(rewriter, loc);
        auto outObj = P4Obj::create(resTy).allocaLLVM(rewriter, loc);  // TODO universal APIs

        auto filterInst = filterObj.getPtrLLVM(rewriter, loc);
        auto outInst = outObj.getPtrLLVM(rewriter, loc);

        callApplyFn(filter, {filterInst, headerInst, outInst});

        rewriter.create<LLVM::ReturnOp>(loc, outObj.getValueLLVM(rewriter, loc));

        return mlir::success();
    }

    mlir::LogicalResult matchAndRewrite(P4HIR::InstantiateOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto mod = op->getParentOfType<mlir::ModuleOp>();
        auto instTarget = mod.lookupSymbol<P4HIR::PackageOp>(op.getCallee());
        if (!instTarget) return mlir::failure();

        // We only handle the main package instantiation here.
        // Generic instantiations are handled in the P4HIRToLLVM lowering patterns.
        if (op.getSymName() != "main") return mlir::failure();

        if (instTarget.getSymName() == "ebpfFilter") {
            auto packageArgs = op.getArgOperands();
            if (packageArgs.size() != 2)
                return rewriter.notifyMatchFailure(op, "Incorrect argument count for ebpfFilter");

            auto parser = unwrapPackageArg<P4HIR::ParserOp>(packageArgs[0]);
            auto filter = unwrapPackageArg<P4HIR::ControlOp>(packageArgs[1]);
            if (!parser || !filter)
                return rewriter.notifyMatchFailure(op, "Incorrect argument types for ebpfFilter");

            auto status = createEBPFFilter(rewriter, op.getLoc(), parser, filter);
            if (status.failed()) return status;
        } else {
            return rewriter.notifyMatchFailure(op, "Unexpected package instantiation");
        }

        rewriter.eraseOp(op);

        return mlir::success();
    }

 private:
    P4LLVMTypeConverter *converter;
};

template <typename OpTy>
struct EraseOpConversion : public mlir::ConvertOpToLLVMPattern<OpTy> {
    using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(OpTy op, typename ConvertOpToLLVMPattern<OpTy>::OpAdaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

// This feels really bad; find some alternative?
// In any case, it's both slower and feels incorrect to do getOrCreate in
// the conversion pattern as convertType is const.
void walkAllTypes(mlir::ModuleOp mod, P4LLVMTypeConverter *converter) {
    llvm::SetVector<Type> allTypes;

    auto registerType = [&](mlir::Type type) { allTypes.insert(removeRef(type)); };

    mod.walk([&](mlir::Operation *op) {
        for (Value result : op->getResults()) registerType(result.getType());

        for (Value operand : op->getOperands()) registerType(operand.getType());

        for (Region &region : op->getRegions())
            for (Block &block : region)
                for (BlockArgument arg : block.getArguments()) registerType(arg.getType());
    });

    for (Type type : allTypes) {
        if (auto structType = mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(type)) {
            converter->registerObjType(P4Obj::fromStructP4(converter, structType));
        }
    }
}

void LowerEBPFToLLVMPass::runOnOperation() {
    mlir::ModuleOp mod = getOperation();

    mlir::LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    P4LLVMTypeConverter typeConverter(&getContext());
    configureP4HIRTypeConverter(typeConverter);
    P4HIR::configureP4HIRToLLVMTypeConverter(typeConverter);
    P4HIR::configureCoreLibToLLVMTypeConverter(typeConverter);

    walkAllTypes(mod, &typeConverter);  // Temporary hack.

    mlir::RewritePatternSet patterns(&getContext());
    P4HIR::populateP4HIRToLLVMConversionPatterns(typeConverter, patterns);
    P4HIR::populateCoreLibToLLVMConversionPatterns(typeConverter, patterns);

    patterns.add<InstantiateMainConversion, EraseOpConversion<P4HIR::ExternOp>,
                 EraseOpConversion<P4HIR::PackageOp>, EraseOpConversion<P4HIR::OverloadSetOp>,
                 EraseOpConversion<P4HIR::ConstructOp>,
                 /* EraseOpConversion<P4HIR::InstantiateOp>, */
                 EraseOpConversion<P4HIR::CallMethodOp>>(typeConverter);

    if (mlir::failed(mlir::applyPartialConversion(mod, target, std::move(patterns))))
        signalPassFailure();
}
}  // end anonymous namespace

std::unique_ptr<Pass> P4::P4MLIR::createLowerEBPFToLLVMPass() {
    return std::make_unique<LowerEBPFToLLVMPass>();
}
