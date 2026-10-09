/* nail the run_loop M-tile bug mechanism for M=256 K=256 N=32: exact bad-row range, and is tile1 a STALE
 * copy of tile0's output (NPU read stale A) or garbage? */
#include "ork_npu.h"
#include <stdio.h>
#include <stdlib.h>
static uint32_t g=999;
static signed char r8(void){ g=g*1103515245u+12345u; return (signed char)(((g>>16)&0x7f)-40); }
int main(void){
    ork_npu *c = ork_npu_init(); if(!c){ printf("no npu\n"); return 1; }
    enum { M=256, K=256, N=32 };
    signed char *A=malloc((size_t)M*K), *B=malloc((size_t)K*N);
    int *C=malloc((size_t)M*N*4), *ref=malloc((size_t)M*N*4);
    g=999; for(int i=0;i<M*K;i++)A[i]=r8(); for(int i=0;i<K*N;i++)B[i]=r8();
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long a=0; for(int k=0;k<K;k++) a+=(long)A[m*K+k]*B[k*N+n]; ref[m*N+n]=(int)a; }
    ork_w *w=ork_mm_pack_i8(c,K,N,B);
    ork_npu_set_core_budget(c,1);
    ork_mm_run_i8(c,w,M,A,C);
    ork_mm_free(c,w);
    int lo=-1,hi=-1,nbad=0;
    for(int m=0;m<M;m++){ int rb=0; for(int n=0;n<N;n++) if(C[m*N+n]!=ref[m*N+n]){rb=1;nbad++;} if(rb){ if(lo<0)lo=m; hi=m; } }
    printf("bad rows [%d..%d], nbadElem=%d, rowsBad~%d\n", lo, hi, nbad, nbad/N);
    for(int m=125;m<=132 && m<M;m++) printf("  row %3d: C[.,0..2]=%d,%d,%d  ref=%d,%d,%d\n", m, C[m*N],C[m*N+1],C[m*N+2], ref[m*N],ref[m*N+1],ref[m*N+2]);
    /* stale-tile0 hypothesis: does row (128+r) == row r ? */
    int stalematch=0, chk=64; for(int r=0;r<chk;r++){ int eq=1; for(int n=0;n<N;n++) if(C[(128+r)*N+n]!=C[r*N+n])eq=0; if(eq)stalematch++; }
    printf("  rows 128+r == rows r (stale tile0) for %d/%d of r=0..63\n", stalematch, chk);
    /* zero hypothesis: is the bad region zero? */
    int zeros=0; for(int n=0;n<N;n++) if(C[128*N+n]==0)zeros++;
    printf("  row128 zeros: %d/%d\n", zeros, N);
    free(A);free(B);free(C);free(ref); ork_npu_free(c); return 0;
}
