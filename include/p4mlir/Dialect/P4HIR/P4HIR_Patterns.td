#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Ops.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Types.td"

def AddIntAttrs : NativeCodeCall<"addIntegerAttrs($_builder, $0, $1, $2)">;
def SubIntAttrs : NativeCodeCall<"subIntegerAttrs($_builder, $0, $1, $2)">;
def MulIntAttrs : NativeCodeCall<"mulIntegerAttrs($_builder, $0, $1, $2)">;

def CreateConstOp : NativeCodeCall<"$_builder.create<P4HIR::ConstOp>($_loc, $0)">;

// def ConstMatcher : Constraint<
//     CPred<"::mlir::success(matchConstant(...))">, "">;
// class IntMinValue<int n> : AttrConstraint<
//     CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getInt() >= " # n>,
//     "whose minimum value is " # n>;

// addi(addi(x, c0), c1) -> addi(x, c0 + c1)
def AddIAddConstant : Pat<
  (BinOp:$res BinOpKind_Add,
    (BinOp BinOpKind_Add, $x, (ConstOp $c0, $_, $_)),
    (ConstOp $c1, $_, $_)),
  (BinOp BinOpKind_Add, $x, (CreateConstOp (AddIntAttrs $res, $c0, $c1)))
>;

// addi(subi(x, c0), c1) -> addi(x, c1 - c0)
def AddISubConstantRHS : Pat<
  (BinOp:$res BinOpKind_Add,
    (BinOp BinOpKind_Sub, $x, (ConstOp $c0, $_, $_)),
    (ConstOp $c1, $_, $_)),
  (BinOp BinOpKind_Add, $x, (CreateConstOp (SubIntAttrs $res, $c1, $c0)))
>;

// addi(subi(c0, x), c1) -> subi(c0 + c1, x)
def AddISubConstantLHS : Pat<
  (BinOp:$res BinOpKind_Add,
    (BinOp BinOpKind_Sub, (ConstOp $c0, $_, $_), $x),
    (ConstOp $c1, $_, $_)),
  (BinOp BinOpKind_Sub, (CreateConstOp (AddIntAttrs $res, $c0, $c1)), $x)
>;

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_PATTERNS_TD
