#ifndef P4MLIR_CONVERSION_P4TOLLVM_H
#define P4MLIR_CONVERSION_P4TOLLVM_H

#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <type_traits>

#include "llvm/Support/Casting.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Types.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

namespace P4::P4ToLLVM {

using namespace mlir;
using namespace P4::P4MLIR;

inline std::string nameFor(mlir::Operation *op) {
    if (auto parserOp = mlir::dyn_cast<P4HIR::ParserOp>(op)) {
        return (llvm::Twine("_p_") + parserOp.getSymName()).str();
    } else if (auto controlOp = mlir::dyn_cast<P4HIR::ControlOp>(op)) {
        return (llvm::Twine("_c_") + controlOp.getSymName()).str();
    } else if (auto externOp = mlir::dyn_cast<P4HIR::ExternOp>(op)) {
        return (llvm::Twine("_e_") + externOp.getSymName()).str();
    } else {
        llvm_unreachable("Invalid op");
        return {};
    }
}

inline std::string nameFor(mlir::Type type) {
    if (auto structLikeType = mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(type)) {
        return (llvm::Twine("_s_") + structLikeType.getName()).str();
    } else if (auto packetInType = mlir::dyn_cast<P4CoreLib::PacketInType>(type)) {
        return "_core_packet_in";
    } else if (auto packetOutType = mlir::dyn_cast<P4CoreLib::PacketOutType>(type)) {
        return "_core_packet_out";
    } else {
        llvm_unreachable("Invalid type");
        return {};
    }
}

inline mlir::Type removeRef(mlir::Type type) {
    if (auto refType = mlir::dyn_cast<P4HIR::ReferenceType>(type)) return refType.getObjectType();

    return type;
}

inline std::string memberFn(llvm::StringRef parent, llvm::StringRef child) {
    return (parent + "_m_" + child).str();
}

inline std::string memberFn(llvm::StringRef parent, P4HIR::ParserStateOp stateOp) {
    return memberFn(parent, stateOp.getSymName());
}

inline std::string memberFn(llvm::StringRef parent, P4HIR::FuncOp funcOp) {
    return memberFn(parent, funcOp.getSymName());
}

inline std::string specialInitFn(mlir::Operation *parent) {
    return memberFn(nameFor(parent), mlir::cast<mlir::SymbolOpInterface>(parent).getName());
}

inline std::string specialApplyFn(llvm::StringRef parent) { return memberFn(parent, "_apply"); }

inline std::string specialApplyFn(mlir::Operation *parent) {
    return specialApplyFn(nameFor(parent));
}

class P4Obj {
 public:
    static P4Obj create(const TypeConverter *converter, mlir::Type p4Type) {
        P4Obj res;
        res.p4Type = p4Type;
        res.llvmType = converter->convertType(p4Type);
        return res;
    }

    static P4Obj create(mlir::Type llvmType) {
        P4Obj res;
        res.llvmType = llvmType;
        return res;
    }

    static P4Obj createAggregateImpl(mlir::MLIRContext *ctx, const std::string &name,
                                     mlir::TypeRange types,
                                     llvm::function_ref<P4Obj(mlir::Type)> createFn) {
        P4Obj res;
        res.name = name;
        for (mlir::Type type : types) res.members.push_back(createFn(type));

        auto llvmTypes =
            llvm::map_to_vector(res.members, [&](auto &member) { return member.llvmType; });

        auto llvmStructType = LLVM::LLVMStructType::getIdentified(ctx, name);
        [[maybe_unused]] auto status = llvmStructType.setBody(llvmTypes, true);
        assert(status.succeeded() && "Expected successful mutation");
        res.llvmType = llvmStructType;

        return res;
    }

    static P4Obj createAggregate(mlir::MLIRContext *ctx, const std::string &name,
                                 mlir::TypeRange llvmTypes) {
        return createAggregateImpl(ctx, name, llvmTypes,
                                   [&](mlir::Type llvmType) { return create(llvmType); });
    }

    static P4Obj createAggregate(const mlir::LLVMTypeConverter *converter, const std::string &name,
                                 mlir::TypeRange p4Types) {
        return createAggregateImpl(&converter->getContext(), name, p4Types,
                                   [&](mlir::Type p4Type) { return create(converter, p4Type); });
    }

    static P4Obj fromStructP4(const mlir::LLVMTypeConverter *converter, mlir::Type type) {
        auto p4StructType = mlir::cast<P4HIR::StructLikeTypeInterface>(type);
        auto memberTypes = llvm::map_to_vector(p4StructType.getFields(),
                                               [](const auto &member) { return member.type; });
        return createAggregateImpl(
            &converter->getContext(), nameFor(p4StructType), memberTypes, [&](mlir::Type p4Type) {
                if (auto structMember = mlir::dyn_cast<P4HIR::StructLikeTypeInterface>(p4Type)) {
                    return fromStructP4(converter, structMember);
                } else {
                    return create(converter, p4Type);
                }
            });
    }

    static P4Obj fromStructLLVM(mlir::MLIRContext *ctx, mlir::Type type) {
        auto llvmStructType = mlir::cast<LLVM::LLVMStructType>(type);
        return createAggregateImpl(
            ctx, llvmStructType.getName().str(), llvmStructType.getBody(),
            [&](mlir::Type llvmType) {
                if (auto structMember = mlir::dyn_cast<LLVM::LLVMStructType>(llvmType)) {
                    return fromStructLLVM(ctx, structMember);
                } else {
                    return create(llvmType);
                }
            });
    }

    struct ObjAccess {
        template <
            typename... Indices,
            typename std::enable_if_t<(std::is_convertible_v<Indices, size_t> && ...), bool> = true>
        ObjAccess getMember(Indices... indices) const {
            size_t tempArr[] = {static_cast<size_t>(indices)...};
            return getMember(tempArr, sizeof...(Indices));
        }

        ObjAccess getMember(SmallVectorImpl<size_t> &indices) const {
            return getMember(indices.data(), indices.size());
        }

        ObjAccess getMember(const size_t *indices, size_t count) const {
            ObjAccess res;
            res.root = root;
            res.rootPtr = rootPtr;
            res.gepArgs = gepArgs;
            res.member = member;
            for (size_t i = 0; i < count; i++) {
                res.member = &res.member->members[indices[i]];
                res.gepArgs.push_back(indices[i]);
            }
            return res;
        }

        mlir::Value getPtrLLVM(mlir::RewriterBase &rewriter, mlir::Location loc) const {
            if (gepArgs.size() == 1) {
                assert((mlir::cast<LLVM::GEPConstantIndex>(gepArgs[0]) == 0));
                return rootPtr;
            } else {
                auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
                auto gepOp =
                    rewriter.create<LLVM::GEPOp>(loc, ptrType, root->llvmType, rootPtr, gepArgs);
                return gepOp.getRes();
            }
        }

        mlir::Value getValueLLVM(mlir::RewriterBase &rewriter, mlir::Location loc) const {
            auto ptr = getPtrLLVM(rewriter, loc);
            auto loadOp = rewriter.create<LLVM::LoadOp>(loc, member->llvmType, ptr);
            return loadOp.getRes();
        }

        void setValueLLVM(mlir::RewriterBase &rewriter, mlir::Location loc, mlir::Value val) const {
            assert((member->llvmType == val.getType()) &&
                   "Member type and write value type are not equal");
            auto ptr = getPtrLLVM(rewriter, loc);
            rewriter.create<LLVM::StoreOp>(loc, val, ptr);
        }

        mlir::Value getPtrP4(mlir::RewriterBase &rewriter, mlir::Location loc) const {
            assert(member->p4Type && "Missing P4 type");
            auto resType = P4HIR::ReferenceType::get(rewriter.getContext(), member->p4Type);
            auto llvmPtr = getPtrLLVM(rewriter, loc);
            auto res = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resType, llvmPtr);
            assert(res->getNumResults() == 1);
            return res->getResult(0);
        }

        mlir::Value getValueP4(mlir::RewriterBase &rewriter, mlir::Location loc) const {
            assert(member->p4Type && "Missing P4 type");
            auto llvmValue = getValueLLVM(rewriter, loc);
            auto res =
                rewriter.create<mlir::UnrealizedConversionCastOp>(loc, member->p4Type, llvmValue);
            assert(res->getNumResults() == 1);
            return res->getResult(0);
        }

        operator const P4Obj *() const { return member; }
        const P4Obj *operator->() const { return member; }

        void dump() const {
            llvm::dbgs() << "Root " << root->getName() << " = " << rootPtr << " : "
                         << root->getTypeLLVM() << "\n";
            llvm::dbgs() << "GEP args [";
            for (LLVM::GEPArg gep : gepArgs)
                llvm::dbgs() << mlir::cast<LLVM::GEPConstantIndex>(gep) << ", ";
            llvm::dbgs() << "]\n";
            llvm::dbgs() << "Member " << member->getName() << " : " << member->getTypeLLVM()
                         << "\n";
        }

        const P4Obj *member;
        const P4Obj *root;
        mlir::Value rootPtr;
        llvm::SmallVector<LLVM::GEPArg, 4> gepArgs;
    };

    ObjAccess thisObj(mlir::Value ptr) const {
        ObjAccess res;
        res.rootPtr = ptr;
        res.member = this;
        res.root = this;
        res.gepArgs.push_back(0);
        return res;
    }

    template <typename... Args>
    ObjAccess getMember(mlir::Value ptr, Args... args) const {
        return thisObj(ptr).getMember(std::forward<Args>(args)...);
    }

    mlir::Value allocaLLVM(mlir::RewriterBase &rewriter, mlir::Location loc) const {
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto one = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(64),
                                                     rewriter.getI64IntegerAttr(1));
        auto allocaOp = rewriter.create<LLVM::AllocaOp>(loc, ptrType, llvmType, one);
        return allocaOp->getResult(0);
    }

    bool isAggregate() const { return !members.empty(); }
    size_t getMemberCount() const { return members.size(); }
    llvm::StringRef getName() const { return name; }
    mlir::Type getTypeLLVM() const { return llvmType; }
    mlir::Type getTypeP4() const { return p4Type; }

 private:
    std::vector<P4Obj> members;
    std::string name;
    mlir::Type p4Type;
    mlir::Type llvmType;
};

class P4LLVMTypeConverter : public mlir::LLVMTypeConverter {
 public:
    using mlir::LLVMTypeConverter::LLVMTypeConverter;

    P4Obj *registerObjType(P4Obj obj) {
        auto objPtr = std::make_unique<P4Obj>(std::move(obj));
        llvm::StringRef nameRef = objPtr->getName();
        [[maybe_unused]] auto [it, ins] = defToObj.insert({nameRef, std::move(objPtr)});
        assert(ins && "Cannot register with duplicate name");
        return it->second.get();
    }

    // TODO Use a ptr or obj union type to provide universal getObjType?

    P4Obj *getObjType(llvm::StringRef name) {
        auto it = defToObj.find(name);
        return (it != defToObj.end()) ? it->second.get() : nullptr;
    }

    P4Obj *getObjType(mlir::Operation *op) { return getObjType(nameFor(op)); }

    P4Obj *getObjType(mlir::Type type) { return getObjType(nameFor(type)); }

 private:
    llvm::DenseMap<llvm::StringRef, std::unique_ptr<P4Obj>> defToObj;
};

}  // namespace P4::P4ToLLVM

#endif  // P4MLIR_CONVERSION_P4TOLLVM_H
