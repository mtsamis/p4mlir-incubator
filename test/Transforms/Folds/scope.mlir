// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!i32i = !p4hir.int<32>
#int3_i32i = #p4hir.int<3> : !i32i

// CHECK-LABEL: module
module {
  // CHECK-LABEL: f1
  p4hir.func @f1() {
    // CHECK-NEXT: p4hir.return
    p4hir.scope {
      p4hir.scope {
        p4hir.scope {
        }
      }
    }
    p4hir.scope {
      p4hir.scope {
      }
    }

    p4hir.return
  }

  // CHECK-LABEL: f2
  p4hir.func @f2() {
    // CHECK-NEXT: p4hir.scope annotations {some_attr1 = []} {
    // CHECK-NEXT: }
    // CHECK-NEXT: p4hir.scope annotations {some_attr2 = []} {
    // CHECK-NEXT: }
    // CHECK-NEXT: p4hir.return

    p4hir.scope {
      p4hir.scope annotations {some_attr1 = []} {
        p4hir.scope {
        }
      }
    }

    p4hir.scope {
      p4hir.scope {
        p4hir.scope annotations {some_attr2 = []} {
        }
      }
    }

    p4hir.return
  }

  // CHECK-LABEL: f3
  p4hir.func @f3(%arg0: !p4hir.ref<!i32i>) {
    // CHECK-NEXT: %c3_i32i = p4hir.const #int3_i32i
    // CHECK-NEXT: p4hir.assign %c3_i32i, %arg0 : <!i32i>
    // CHECK-NEXT: p4hir.return

    p4hir.scope {
      p4hir.scope {
        %c3_i32i = p4hir.const #int3_i32i
        p4hir.assign %c3_i32i, %arg0 : <!i32i>
      }
    }

    p4hir.return
  }
}
