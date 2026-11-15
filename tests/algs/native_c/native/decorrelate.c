#include <math.h>

/* out = y - slope * (x - mean(x)), slope from ordinary least squares; returns the slope. */
double decorrelate(const double *y, const double *x, double *out, int n) {
    double mx = 0.0, my = 0.0, sxx = 0.0, sxy = 0.0;
    int i;
    for (i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    for (i = 0; i < n; i++) { sxx += (x[i] - mx) * (x[i] - mx); sxy += (x[i] - mx) * (y[i] - my); }
    double slope = sxx > 0.0 ? sxy / sxx : 0.0;
    #pragma omp parallel for
    for (i = 0; i < n; i++) out[i] = y[i] - slope * (x[i] - mx);
    return slope;
}
