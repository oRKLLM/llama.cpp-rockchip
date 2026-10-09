/* isolate the M=256 K=256 N=32 int8 mismatch via the DIRECT library path (no orkd). Sweeps neighbours +
 * single vs multi-core to pin the trigger dimension + the first bad row. */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>

static uint32_t g=999;
static signed char r8(void){ g = g*1103515245u + 12345u; return (signed char)(((g>>16)&0x7f) - 40); }

static void test(ork_npu *c, int M, int K, int N, int force1){
    signed char *A = malloc((size_t)M*K), *B = malloc((size_t)K*N);
    int *C = malloc((size_t)M*N*4), *ref = malloc((size_t)M*N*4);
    g = 999; for (int i=0;i<M*K;i++) A[i]=r8(); for (int i=0;i<K*N;i++) B[i]=r8();
    for (int m=0;m<M;m++) for (int n=0;n<N;n++){ long a=0; for (int k=0;k<K;k++) a += (long)A[m*K+k]*B[k*N+n]; ref[m*N+n]=(int)a; }
    ork_w *w = ork_mm_pack_i8(c,K,N,B);
    if (!w){ printf("M=%d K=%d N=%d: PACK FAIL\n",M,K,N); free(A);free(B);free(C);free(ref); return; }
    ork_npu_set_core_budget(c, force1?1:0);
    int rc = ork_mm_run_i8(c,w,M,A,C);
    ork_mm_free(c,w);
    int bad=0, frow=-1; for (int i=0;i<M*N;i++) if (C[i]!=ref[i]){ bad++; if(frow<0) frow=i/N; }
    printf("M=%-3d K=%-4d N=%-3d %-6s rc=%d : %-4s bad=%d/%d", M,K,N, force1?"1core":"multi", rc, bad?"FAIL":"ok", bad, M*N);
    if (bad) printf("  firstBadRow=%d", frow);
    printf("\n");
    free(A);free(B);free(C);free(ref);
}
int main(void){
    ork_npu *c = ork_npu_init(); if(!c){ printf("no npu\n"); return 1; }
    printf("cores=%d\n", ork_npu_cores(c));
    test(c,256,256,32,0);   /* the reported failure (multi-core) */
    test(c,256,256,32,1);   /* same, forced single-core -> is it the multi-core path? */
    test(c,128,256,32,0);   /* smaller M -> is large-M the trigger? */
    test(c,192,256,32,0);
    test(c,256,256,64,0);   /* larger N -> is N=32 the trigger? */
    test(c,256,512,32,0);   /* larger K -> is K=256 the trigger? */
    test(c,256,128,32,0);
    ork_npu_free(c);
    return 0;
}
