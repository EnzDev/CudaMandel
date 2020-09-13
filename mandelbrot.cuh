/*
 * Utility function for use in __global__ GPU context
 */
#ifndef MANDELBROT_FUN
#define MANDELBROT_FUN
// Store the RGB component into a 3 component flat array
#define RGB(TABLE, POSITION, R, G, B) \
    TABLE[(POSITION) + 0] = R; \
    TABLE[(POSITION) + 1] = G; \
    TABLE[(POSITION) + 2] = B; \

// Compute a float modulus
#define cfmod(a, b) (a - (b * int(a/b)))

// Compute the absolute value
#define cabs(N) (((N)<0)?(-(N)):(N))
#endif //MANDELBROT_FUN


#ifndef CUDAMANDEL_MANDELBROT_CUH
#define CUDAMANDEL_MANDELBROT_CUH

__global__ void translate(
        double *r, double *i,
        double width, double height,
        double offset_r, double offset_i
);

__global__ void mandelbrot(int *iteration_table, const double *r, const double *i, int max);

__global__ void colorize(unsigned char *pixel_table, const int *iteration_table, int max_loop);

#endif //CUDAMANDEL_MANDELBROT_CUH
