// tk_cfg.hpp (generator): the build settings a code object must share with the binary that launches its kernels: the ones that
// change an argument struct's layout or a compile-time constant of the model (HcArgs has a field more under FNX_KBENCH, FNX_MALL_PF adds the prefetch
// word, FNX_NCARD is in every split constant). gen/tkgen.py compiles a variant with the target's own definitions, the variant carries FNX_TK_CFG as the
// device string tk_cfg, the manifest records it, and tk.hpp refuses a manifest whose string is not this binary's.
// Not in the string: settings that change only a kernel's body (FNX_MALL_NT, a family's -D parameters): those are what a variant is for.
// FNX_TK_IFACE: raise it with a change of an argument struct or of a kernel's parameter list that keeps every size (the launch also checks the
// kernel's argument bytes against the code object's, which catches a changed size without it).
#pragma once

#define FNX_TK_IFACE 1

#define FNX_TK_STR2(x) #x
#define FNX_TK_STR(x) FNX_TK_STR2(x)

#if FNX_HIP
#define FNX_TK_C_HIP "1"
#else
#define FNX_TK_C_HIP "0"
#endif
#if FNX_KBENCH
#define FNX_TK_C_KBENCH "1"
#else
#define FNX_TK_C_KBENCH "0"
#endif
#if FNX_ATT_HEADS
#define FNX_TK_C_HEADS "1"
#else
#define FNX_TK_C_HEADS "0"
#endif
#if FNX_ATT_MM
#define FNX_TK_C_ATT_MM "1"
#else
#define FNX_TK_C_ATT_MM "0"
#endif
#if FNX_MM
#define FNX_TK_C_MM "1"
#else
#define FNX_TK_C_MM "0"
#endif
#if FNX_MM_HC
#define FNX_TK_C_MM_HC "1"
#else
#define FNX_TK_C_MM_HC "0"
#endif
#if FNX_MALL_PF
#define FNX_TK_C_MALL_PF "1"
#else
#define FNX_TK_C_MALL_PF "0"
#endif
#if FNX_ROUTER_BF16
#define FNX_TK_C_RT16 "1"
#else
#define FNX_TK_C_RT16 "0"
#endif
#if FNX_DF2
#define FNX_TK_C_DF2 "1"
#else
#define FNX_TK_C_DF2 "0"
#endif
#if FNX_HOSTRING
#define FNX_TK_C_HOSTRING "1"
#else
#define FNX_TK_C_HOSTRING "0"
#endif
#ifdef FNX_PF_JOB
#define FNX_TK_C_PF_JOB FNX_TK_STR(FNX_PF_JOB)
#else
#define FNX_TK_C_PF_JOB "64"
#endif
#ifdef FNX_NCARD
#define FNX_TK_C_NCARD FNX_TK_STR(FNX_NCARD)
#else
#define FNX_TK_C_NCARD "0"
#endif
#ifdef FNX_ARCH
#define FNX_TK_C_ARCH FNX_TK_STR(FNX_ARCH)
#else
#define FNX_TK_C_ARCH "0"
#endif
#ifdef FNX_27B
#define FNX_TK_C_27B FNX_TK_STR(FNX_27B)
#else
#define FNX_TK_C_27B "0"
#endif

// no spaces: the manifest is split at white space
#define FNX_TK_CFG                                                                                                                              \
    "iface=" FNX_TK_STR(FNX_TK_IFACE) ";hip=" FNX_TK_C_HIP ";ncard=" FNX_TK_C_NCARD ";arch=" FNX_TK_C_ARCH ";27b=" FNX_TK_C_27B ";kbench="   \
    FNX_TK_C_KBENCH ";heads=" FNX_TK_C_HEADS ";attmm=" FNX_TK_C_ATT_MM ";mm=" FNX_TK_C_MM ";mmhc=" FNX_TK_C_MM_HC ";mallpf=" FNX_TK_C_MALL_PF \
    ";rt16=" FNX_TK_C_RT16 ";df2=" FNX_TK_C_DF2 ";hostring=" FNX_TK_C_HOSTRING ";pfjob=" FNX_TK_C_PF_JOB
