#include <cstdint>
#include <cmath>
#include <omp.h>
extern "C" void risk_pair_counts(const float* x, int k, int64_t n, int threads, int64_t* less, int64_t* equal) {
    #pragma omp parallel for schedule(dynamic,1) num_threads(threads)
    for(int i=0;i<k;i++) {
        equal[int64_t(i)*k+i]=n;
        const float* a=x+int64_t(i)*n;
        for(int j=i+1;j<k;j++) {
            const float* b=x+int64_t(j)*n;
            int64_t lo=0,eq=0;
            #pragma omp simd reduction(+:lo,eq)
            for(int64_t p=0;p<n;p++) {lo+=(a[p]<b[p]);eq+=(a[p]==b[p]);}
            less[int64_t(i)*k+j]=lo;less[int64_t(j)*k+i]=n-lo-eq;
            equal[int64_t(i)*k+j]=eq;equal[int64_t(j)*k+i]=eq;
        }
    }
}
