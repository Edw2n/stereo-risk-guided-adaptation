#include <cstdint>
#include <limits>
extern "C" void candidate_concordance(const float* risk, const float* error,
                                      int64_t n, int k, double* out) {
    for (int64_t p=0; p<n; ++p) {
        const float* r=risk+p*k; const float* e=error+p*k;
        int good=0, ties=0, comparable=0;
        for (int i=0; i<k; ++i) for (int j=i+1; j<k; ++j) {
            if (e[i]==e[j]) continue;
            ++comparable;
            if (r[i]==r[j]) ++ties;
            else if ((r[i]<r[j])==(e[i]<e[j])) ++good;
        }
        out[p]=comparable ? (good+0.5*ties)/comparable : std::numeric_limits<double>::quiet_NaN();
    }
}
