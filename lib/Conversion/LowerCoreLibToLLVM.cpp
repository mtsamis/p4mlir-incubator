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

namespace {

struct PacketExtractOpConversion : public mlir::ConvertOpToLLVMPattern<P4CoreLib::PacketExtractOp> {
    PacketExtractOpConversion(P4LLVMTypeConverter &converter)
        : ConvertOpToLLVMPattern(converter), converter(&converter) {}

    size_t extractMemberImpl(mlir::RewriterBase &rewriter, mlir::Location loc,
                             const P4Obj::ObjAccess &packetIn, size_t bitOffset,
                             const P4Obj::ObjAccess &dest) const {
        if (mlir::isa<P4HIR::ValidBitType>(dest->getTypeP4())) {
            // Validity bit is lowered as a one-element struct.
            auto ptr = dest.getPtrLLVM(rewriter, loc);
            auto validityBitObj = P4Obj::fromStructLLVM(rewriter.getContext(), dest->getTypeLLVM());
            auto innerBit = validityBitObj.getMember(ptr, 0);
            auto constTrue = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getBoolAttr(true));
            innerBit.setValueLLVM(rewriter, loc, constTrue);
            return 8;
        }

        // TODO cleanup to do here :)
        auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto byteTy = rewriter.getIntegerType(8);

        size_t destWidthBits = mlir::cast<mlir::IntegerType>(dest->getTypeLLVM()).getWidth();
        size_t destWidthBytes = (destWidthBits + 7) / 8;
        size_t byteOffset = bitOffset / 8;
        size_t bytesToRead = llvm::NextPowerOf2(destWidthBytes);

        // Memcpy for alignment (check if LLVM spec allows unaligned access w/o memcpy).
        auto tempType = rewriter.getIntegerType(bytesToRead * 8);

        // auto temp = P4Obj::create(tempType).allocaLLVM(rewriter, loc);
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto one = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(64),
                                                     rewriter.getI64IntegerAttr(1));
        auto allocaOp = rewriter.create<LLVM::AllocaOp>(loc, ptrType, tempType, one);
        auto temp = allocaOp->getResult(0);
        // this alloca will be replaced anyway.

        auto src = packetIn.getMember(0).getValueLLVM(rewriter, loc);
        src = rewriter.create<LLVM::GEPOp>(loc, ptrTy, byteTy, src,
                                           ArrayRef<LLVM::GEPArg>{byteOffset});  // Check if ok.
        auto bytesToReadVal = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getIntegerType(64), rewriter.getI64IntegerAttr(bytesToRead));
        rewriter.create<LLVM::MemcpyOp>(loc, temp, src, bytesToReadVal, false);
        mlir::Value tempVal = rewriter.create<LLVM::LoadOp>(loc, tempType, temp).getRes();

        if ((bitOffset % 8) != 0) {
            auto shiftAmount = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getIntegerAttr(tempType, bitOffset % 8));
            tempVal = rewriter.create<LLVM::LShrOp>(loc, tempVal, shiftAmount);
        }

        if ((bytesToRead * 8) != destWidthBits)
            tempVal = rewriter.create<LLVM::TruncOp>(loc, dest->getTypeLLVM(), tempVal);

        dest.setValueLLVM(rewriter, loc, tempVal);

        return destWidthBits;
    }

    size_t extractImpl(mlir::RewriterBase &rewriter, mlir::Location loc,
                       const P4Obj::ObjAccess &packetIn, size_t bitOffset,
                       P4Obj::ObjAccess dest) const {
        if (dest->isAggregate()) {
            size_t bitsConsumed = 0;
            for (size_t i = 0; i < dest->getMemberCount(); i++) {
                bitsConsumed += extractImpl(rewriter, loc, packetIn, bitOffset + bitsConsumed,
                                            dest.getMember(i));
            }
            return bitsConsumed;
        } else {
            return extractMemberImpl(rewriter, loc, packetIn, bitOffset, dest);
        }
    }

    mlir::LogicalResult matchAndRewrite(P4CoreLib::PacketExtractOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        // bitsToExtract = sizeofInBits(headerLValue);
        // lastBitNeeded = this.nextBitIndex + bitsToExtract;
        // ParserModel.verify(this.lengthInBits >= lastBitNeeded, error.PacketTooShort);
        // headerLValue = this.data.extractBits(this.nextBitIndex, bitsToExtract);
        // headerLValue.valid$ = true;
        // if headerLValue.isNext$ {
        //     verify(headerLValue.nextIndex$ < headerLValue.size, error.StackOutOfBounds);
        //     headerLValue.nextIndex$ = headerLValue.nextIndex$ + 1;
        // }
        // this.nextBitIndex += bitsToExtract;

        // TODO checks and other things missing, etc
        // TODO clean up as well.
        // TODO also optimize codegen a bit.
        auto refType = mlir::dyn_cast<P4HIR::ReferenceType>(op.getHdr().getType());
        if (!refType) return mlir::failure();

        auto structType = mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(refType.getObjectType());
        if (!structType) return mlir::failure();

        auto packetInObj = converter->getObjType("_core_packet_in");
        auto headerObj = P4Obj::fromStructP4(getTypeConverter(), structType);
        auto packetIn = packetInObj->thisObj(adaptor.getPacketIn());
        auto header = headerObj.thisObj(adaptor.getHdr());

        auto nextBitIndexMember = packetIn.getMember(1);
        auto curNextBitIdx = nextBitIndexMember.getValueLLVM(rewriter, op.getLoc());
        size_t bitsConsumed = extractImpl(rewriter, op.getLoc(), packetIn, 0, header);
        auto bitsConsumedVal = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), rewriter.getI32IntegerAttr(bitsConsumed));
        auto newNextBitIdx =
            rewriter.create<LLVM::AddOp>(op.getLoc(), curNextBitIdx, bitsConsumedVal);
        nextBitIndexMember.setValueLLVM(rewriter, op.getLoc(), newNextBitIdx);

        rewriter.eraseOp(op);

        return mlir::success();
    }

 private:
    P4LLVMTypeConverter *converter;
};

}  // end anonymous namespace

void P4HIR::populateCoreLibToLLVMConversionPatterns(P4LLVMTypeConverter &converter,
                                                    mlir::RewritePatternSet &patterns) {
    patterns.add<PacketExtractOpConversion>(converter);
}

void P4HIR::configureCoreLibToLLVMTypeConverter(P4LLVMTypeConverter &converter) {
    converter.addConversion([&](P4CoreLib::PacketInType packetInType) {
        return LLVM::LLVMPointerType::get(packetInType.getContext());
    });

    converter.addConversion([&](P4CoreLib::PacketOutType packetOutType) {
        return LLVM::LLVMPointerType::get(packetOutType.getContext());
    });

    // struct packet_in {
    //     byte[] data;
    //     unsigned lengthInBits;
    //     unsigned nextBitIndex;
    // }
    auto ptrTy = LLVM::LLVMPointerType::get(&converter.getContext());
    auto unsignedTy = mlir::IntegerType::get(&converter.getContext(), 32);
    auto packetInObj = P4Obj::createAggregate(&converter.getContext(), "_core_packet_in",
                                              {ptrTy, unsignedTy, unsignedTy});
    converter.registerObjType(std::move(packetInObj));
}
