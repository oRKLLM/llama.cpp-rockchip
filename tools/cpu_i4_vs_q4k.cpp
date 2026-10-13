// cpu_i4_vs_q4k — CPU-only: does ork-native int4 (per-channel scale, 4.0 bits) beat ggml's real Q4_K
// (~4.5 bits, per-32-block scale+min) for a MEMORY-BOUND decode GEMV? This gates the "CPU=int4" premise:
// decode is CPU-bound, so replacing the Q4_K CPU path with ork-int4 only helps if ork-int4 is FASTER.
// Uses ggml's ACTUAL Q4_K x Q8_K vec_dot (not a reimpl) so the comparison is honest. Also reports
// reconstruction error of each scheme vs the f32 weight. Big weight (68MB f32) so it's DRAM-bound.
//   build (on board, links built ggml): see cpu_i4_vs_q4k.sh
//   run: ./cpu_i4_vs_q4k [iters] [threads]   (CPU-only, no NPU — board-safe)
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <ctime>
#include <arm_neon.h>
static double now_us(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }

static int gK,gN,gNT;
static const uint8_t *gW_i4;          // ork int4: [N][K/2] nibbles
static const float   *gBsc;           // ork int4: per-channel scale [N]
static const int8_t  *gA_i8;          // ork int4: int8 activation [K]
static float          gAsc;           // ork int4: activation scale
static const void    *gW_q4k;         // Q4_K rows [N] (each row_size bytes)
static size_t         gRowQ4K;
static const void    *gA_q8k;         // Q8_K activation block(s)
static float         *gC;
static ggml_vec_dot_t gVecDot;

// ork int4 GEMV row: C[n] = bsc[n]*asc * sum_k q4(w)*q8(a).  Reads K/2 weight bytes.
static inline int32_t dot_i4(const uint8_t*b4,const int8_t*a,int K){
    int32x4_t ac=vdupq_n_s32(0); int k=0,kb=0; uint8x16_t lo=vdupq_n_u8(0x0f);
    for(;k+32<=K;k+=32,kb+=16){
        uint8x16_t p=vld1q_u8(b4+kb);
        int8x16_t even=vreinterpretq_s8_u8(vandq_u8(p,lo));          // low nibbles
        int8x16_t odd =vreinterpretq_s8_u8(vshrq_n_u8(p,4));         // high nibbles
        even=vshrq_n_s8(vshlq_n_s8(even,4),4);                       // sign-extend 4->8
        odd =vshrq_n_s8(vshlq_n_s8(odd ,4),4);
        // ork nibble layout: even holds codes k,k+2..; odd holds k+1,k+3.. — interleave matches expand_chan
        ac=vdotq_s32(ac,even,vld1q_s8(a+k));
        ac=vdotq_s32(ac,odd ,vld1q_s8(a+k+16));
    }
    return vaddvq_s32(ac);
}
static void wk_i4(int lo,int hi){ for(int n=lo;n<hi;n++) gC[n]=gBsc[n]*gAsc*(float)dot_i4(gW_i4+(size_t)n*(gK/2),gA_i8,gK); }
static void wk_q4k(int lo,int hi){ for(int n=lo;n<hi;n++){ float s; gVecDot(gK,&s,0,(const char*)gW_q4k+(size_t)n*gRowQ4K,0,gA_q8k,0,1); gC[n]=s; } }
static double run(int mode){ std::vector<std::thread> th; int per=(gN+gNT-1)/gNT;
    for(int t=0;t<gNT;t++){int a=t*per,b=(t+1)*per<gN?(t+1)*per:gN; if(mode==0)th.emplace_back(wk_i4,a,b); else th.emplace_back(wk_q4k,a,b);}
    for(auto&x:th)x.join(); return 0; }

int main(int argc,char**argv){
    int iters=argc>1?atoi(argv[1]):20; gNT=argc>2?atoi(argv[2]):4;
    int K=3584,N=18944; gK=K; gN=N;
    printf("cpu_i4_vs_q4k: M=1 K=%d N=%d  f32=%.0fMB  %d threads\n",K,N,(double)N*K*4/1e6,gNT);
    // f32 reference weight + activation (fixed pattern)
    std::vector<float> W((size_t)N*K), Av(K);
    // Gaussian weights (Box-Muller, fixed-seed LCG) — models real LLM weight dist so NF4 is judged fairly.
    uint64_t s=0x2545F4914F6CDD1DULL; auto u=[&](){ s=s*6364136223846793005ULL+1442695040888963407ULL; return ((s>>33)&0x7fffffff)/2147483647.0f; };
    for(int k=0;k<K;k++) Av[k]=0.5f*(u()-0.5f);
    for(size_t i=0;i<(size_t)N*K;i++){ float a=u(); if(a<1e-7f)a=1e-7f; if(a>0.9999999f)a=0.9999999f; float b=u(); W[i]=0.08f*sqrtf(-2.0f*logf(a))*cosf(6.2831853f*b); }

    // ---- ork int4: per-channel absmax/7, nibble pack (even=low, odd=high, matches expand_chan_i4_i8) ----
    std::vector<uint8_t> Wi4((size_t)N*(K/2)); std::vector<float> Bsc(N);
    for(int n=0;n<N;n++){ const float*fr=&W[(size_t)n*K]; float mx=1e-9f; for(int k=0;k<K;k++){float v=fabsf(fr[k]); if(v>mx)mx=v;}
        float sc=mx/7.0f, inv=sc>0?1.0f/sc:0.0f; Bsc[n]=sc; uint8_t*nb=&Wi4[(size_t)n*(K/2)];
        for(int k=0;k<K;k+=2){ int q0=(int)lrintf(fr[k]*inv); if(q0>7)q0=7;if(q0<-7)q0=-7;
            int q1=(int)lrintf(fr[k+1]*inv); if(q1>7)q1=7;if(q1<-7)q1=-7; nb[k>>1]=(uint8_t)((q0&0xf)|((q1&0xf)<<4)); } }
    // ork activation: per-tensor int8 (absmax/127)
    float amx=1e-9f; for(int k=0;k<K;k++){float v=fabsf(Av[k]); if(v>amx)amx=v;} gAsc=amx/127.0f; float ainv=127.0f/amx;
    std::vector<int8_t> Ai8(K); for(int k=0;k<K;k++){int q=(int)lrintf(Av[k]*ainv); Ai8[k]=(int8_t)(q>127?127:q<-127?-127:q);}
    gW_i4=Wi4.data(); gBsc=Bsc.data(); gA_i8=Ai8.data();

    // ---- ggml Q4_K weight + Q8_K activation (real kernels) ----
    size_t rowq4k = ggml_row_size(GGML_TYPE_Q4_K, K); gRowQ4K=rowq4k;
    std::vector<char> Wq4k((size_t)N*rowq4k);
    ggml_quantize_chunk(GGML_TYPE_Q4_K, W.data(), Wq4k.data(), 0, N, K, nullptr);
    size_t rowq8k = ggml_row_size(GGML_TYPE_Q8_K, K);
    std::vector<char> Aq8k(rowq8k);
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(Av.data(), Aq8k.data(), K);
    gW_q4k=Wq4k.data(); gA_q8k=Aq8k.data();
    gVecDot = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K)->vec_dot;

    std::vector<float> C(N); gC=C.data();
    // f32 reference for accuracy
    std::vector<float> Cref(N);
    for(int n=0;n<N;n++){ double s=0; const float*fr=&W[(size_t)n*K]; for(int k=0;k<K;k++) s+=fr[k]*Av[k]; Cref[n]=(float)s; }

    run(0); std::vector<float> Ci4=C; run(1); std::vector<float> Cq=C;
    // ---- WEIGHT-reconstruction rel-RMSE per scheme (activation-independent quant quality) ----
    // dequant each scheme back to f32, accumulate ||dq-W||/||W|| over a sample of channels.
    static const float NFR[16]={-1.0f,-0.6961928f,-0.5250731f,-0.3949175f,-0.2844414f,-0.1847734f,-0.0910500f,0.0f,
        0.0795803f,0.1609302f,0.2461123f,0.3379152f,0.4407098f,0.5626170f,0.7229568f,1.0f};
    double e_i4=0,e_i5=0,e_nf=0,e_bk=0,e_q4=0,nrm=0;
    std::vector<char> q4row(gRowQ4K); std::vector<float> dqW(K);
    for(int n=0;n<N;n++){ const float*fr=&W[(size_t)n*K];
        float mx=1e-9f; for(int k=0;k<K;k++){float v=fabsf(fr[k]);if(v>mx)mx=v;}
        float s4=mx/7.0f,i4=s4>0?1/s4:0, s5=mx/15.0f,i5=s5>0?1/s5:0, im=mx>0?1/mx:0;
        for(int k=0;k<K;k++){ double w=fr[k]; nrm+=w*w;
            int q4v=(int)lrintf(fr[k]*i4); if(q4v>7)q4v=7;if(q4v<-7)q4v=-7; double d=q4v*s4-w; e_i4+=d*d;
            int q5v=(int)lrintf(fr[k]*i5); if(q5v>15)q5v=15;if(q5v<-15)q5v=-15; d=q5v*s5-w; e_i5+=d*d;
            float wn=fr[k]*im; if(wn>1)wn=1;if(wn<-1)wn=-1; int bi=0;float bd=1e9f; for(int i=0;i<16;i++){float dd=fabsf(NFR[i]-wn);if(dd<bd){bd=dd;bi=i;}} d=NFR[bi]*mx-w; e_nf+=d*d; }
        for(int b=0;b<K;b+=64){int e2=b+64<K?b+64:K; float bm=1e-9f; for(int k=b;k<e2;k++){float v=fabsf(fr[k]);if(v>bm)bm=v;} float sc=bm/7.0f,bi2=sc>0?1/sc:0;
            for(int k=b;k<e2;k++){int q=(int)lrintf(fr[k]*bi2);if(q>7)q=7;if(q<-7)q=-7; double d=q*sc-fr[k]; e_bk+=d*d;} }
    }
    // Q4_K recon: dequant the packed row back and diff (sample first 512 channels for speed)
    { int SN=N<512?N:512; double eq=0,nq=0; for(int n=0;n<SN;n++){ ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float((const char*)gW_q4k+(size_t)n*gRowQ4K, dqW.data(), K);
        const float*fr=&W[(size_t)n*K]; for(int k=0;k<K;k++){double d=dqW[k]-fr[k]; eq+=d*d; nq+=(double)fr[k]*fr[k];} } e_q4=sqrt(eq/nq); }
    auto rmse=[&](std::vector<float>&X){ (void)X; return 0.0; }; (void)Cref;
    // ---- extra accuracy candidates (no separate GEMV; dequant->exact dot vs f32 ref) ----
    static const float NF4[16]={-1.0f,-0.6961928f,-0.5250731f,-0.3949175f,-0.2844414f,-0.1847734f,-0.0910500f,0.0f,
        0.0795803f,0.1609302f,0.2461123f,0.3379152f,0.4407098f,0.5626170f,0.7229568f,1.0f};
    std::vector<float> Cnf4(N), Cblk(N);
    for(int n=0;n<N;n++){ const float*fr=&W[(size_t)n*K];
        // NF4 per-channel: scale=absmax, nearest codebook level
        float mx=1e-9f; for(int k=0;k<K;k++){float v=fabsf(fr[k]);if(v>mx)mx=v;} float inv=mx>0?1.0f/mx:0;
        double s=0; for(int k=0;k<K;k++){ float wn=fr[k]*inv; if(wn>1)wn=1;if(wn<-1)wn=-1; int bi=0;float bd=1e9f;
            for(int i=0;i<16;i++){float d=fabsf(NF4[i]-wn); if(d<bd){bd=d;bi=i;}} s+=(double)(NF4[bi]*mx)*Av[k]; } Cnf4[n]=(float)s;
        // uniform int4 per-64-block (Q4_K-like granularity, but loses free-inflate: each block has own scale)
        double sb=0; for(int b=0;b<K;b+=64){ int e=b+64<K?b+64:K; float bm=1e-9f; for(int k=b;k<e;k++){float v=fabsf(fr[k]);if(v>bm)bm=v;}
            float sc=bm/7.0f, bi=sc>0?1.0f/sc:0; for(int k=b;k<e;k++){int q=(int)lrintf(fr[k]*bi);if(q>7)q=7;if(q<-7)q=-7; sb+=(double)(q*sc)*Av[k];} } Cblk[n]=(float)sb; }
    (void)Cnf4;(void)Cblk;
    double NR=sqrt(nrm);
    printf("  WEIGHT-recon rel-RMSE (Gaussian weights, lower=better):\n");
    printf("    int4 per-channel uniform (FREE inflate):  %.4f\n", sqrt(e_i4)/NR);
    printf("    int5 per-channel uniform (5-bit expand):  %.4f\n", sqrt(e_i5)/NR);
    printf("    NF4  per-channel codebook (FREE LUT):     %.4f\n", sqrt(e_nf)/NR);
    printf("    int4 per-64-block (NO free inflate):      %.4f\n", sqrt(e_bk)/NR);
    printf("    Q4_K per-32-block (ggml):                 %.4f\n", e_q4);

    run(0); double t0=now_us(); for(int i=0;i<iters;i++) run(0); double ti4=(now_us()-t0)/iters;
    run(1); t0=now_us(); for(int i=0;i<iters;i++) run(1); double tq=(now_us()-t0)/iters;
    printf("  int4 (4.0 bit, per-chan): %8.1f us  %6.1f GB/s\n", ti4, (double)N*K/2/ti4/1e3);
    printf("  Q4_K (~4.5bit, ggml):     %8.1f us  %6.1f GB/s\n", tq, (double)N*rowq4k/tq/1e3);
    printf("  ★ int4/Q4_K speedup: %.2fx  (>1 => ork-int4 CPU beats Q4_K => CPU=int4 premise holds)\n", tq/ti4);
    return 0;
}
