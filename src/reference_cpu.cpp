#include "reference_cpu.hpp"
#include <cmath>
#include <algorithm>

static int bitlen(int n){ int b=0; while((1<<b)<n) ++b; return b; }
static int bitrev(int x, int b){ int r=0; for(int i=0;i<b;++i){ r=(r<<1)|(x&1); x>>=1; } return r; }

void cpu_fft_forward_radix2(const std::vector<std::complex<double>>& x,
                            std::vector<std::complex<double>>& out){
    int n=(int)x.size(); out=x;
    int B=bitlen(n);
    for(int i=0;i<n;++i){ int j=bitrev(i,B); if(j>i) std::swap(out[i],out[j]); }
    for(int m=2;m<=n;m<<=1){
        int j=m>>1;
        for(int k=0;k<n;k+=m){
            for(int j1=0;j1<j;++j1){
                double theta = (double)(n/m) * (double)(2*j1+1) * (-M_PI/(double)n);
                std::complex<double> tw(cos(theta), sin(theta));
                auto u=out[k+j1], v=out[k+j1+j]*tw;
                out[k+j1]=u+v; out[k+j1+j]=u-v;
            }
        }
    }
}
void cpu_ifft_radix2(const std::vector<std::complex<double>>& X,
                     std::vector<std::complex<double>>& x){
    int n=(int)X.size(); x=X;
    int B=bitlen(n);
    for(int i=0;i<n;++i){ int j=bitrev(i,B); if(j>i) std::swap(x[i],x[j]); }
    for(int m=2;m<=n;m<<=1){
        int j=m>>1;
        for(int k=0;k<n;k+=m){
            for(int j1=0;j1<j;++j1){
                double theta = (double)(n/m) * (double)(2*j1+1) * (+M_PI/(double)n);
                std::complex<double> tw(cos(theta), sin(theta));
                auto u=x[k+j1], v=x[k+j1+j]*tw;
                x[k+j1]=u+v; x[k+j1+j]=u-v;
            }
        }
    }
    for(int i=0;i<n;++i) x[i] /= (double)n;
}

static uint32_t mod_pow_u(uint64_t a, uint64_t e){
    uint64_t r=1%Q; a%=Q; while(e){ if(e&1) r=(r*a)%Q; a=(a*a)%Q; e>>=1; } return (uint32_t)r;
}

void cpu_ntt_forward(const std::vector<uint32_t>& a, std::vector<uint32_t>& A){
    int n=(int)a.size(); A=a;
    int B=bitlen(n);
    for(int i=0;i<n;++i){ int j=bitrev(i,B); if(j>i) std::swap(A[i],A[j]); }
    uint32_t g=3;
    uint32_t gamma = mod_pow_u(g, (Q-1)/(2*n));
    for(int m=2;m<=n;m<<=1){
        int j=m>>1;
        uint32_t gp = mod_pow_u(gamma, n/m);
        for(int k=0;k<n;k+=m){
            for(int j1=0;j1<j;++j1){
                uint64_t e=(uint64_t)(2*j1+1);
                uint32_t tw = mod_pow_u(gp, e);
                uint32_t u=A[k+j1], v=(uint32_t)(((unsigned long long)A[k+j1+j]*tw)%Q);
                A[k+j1]= (u+v)%Q; A[k+j1+j]=(u+Q-v)%Q;
            }
        }
    }
}

void cpu_schoolbook_negacyclic_fft(const std::vector<std::complex<double>>& a,
                                   const std::vector<std::complex<double>>& b,
                                   std::vector<std::complex<double>>& c){
    int n=(int)a.size(); c.assign(n, {0,0});
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            int k=i+j; auto prod=a[i]*b[j];
            if(k>=n) c[k-n]-=prod; else c[k]+=prod;
        }
    }
}
void cpu_schoolbook_negacyclic_ntt(const std::vector<uint32_t>& a,
                                   const std::vector<uint32_t>& b,
                                   std::vector<uint32_t>& c){
    int n=(int)a.size(); c.assign(n,0u);
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            int k=i+j; uint32_t prod=(uint32_t)(((unsigned long long)a[i]*b[j])%Q);
            if(k>=n) c[k-n]=(c[k-n]+Q-prod)%Q; else c[k]=(c[k]+prod)%Q;
        }
    }
}
