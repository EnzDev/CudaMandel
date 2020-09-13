#include "constants.cuh"
#include "mandelbrot.cuh"


/**
 * Translate the coordinates from the block coordinates into their real and imaginary values in the zoom space
 * @param r Data table for coordinate storage
 * @param i Data table for coordinate storage
 */
__global__ void translate(
        double *r, double *i,
        const double width, const double height,
        const double offset_r, const double offset_i
) {
    r[blockIdx.y * SIZE + blockIdx.x] = (double(blockIdx.x) * width  / double(SIZE)) - (width / 2.) + offset_r;
    i[blockIdx.y * SIZE + blockIdx.x] = (double(blockIdx.y) * height / double(SIZE)) - (height / 2.) + offset_i;
}

/*
 * Compute how much iterations are needed to escape the mandelbrot set
 */
__global__ void mandelbrot(int *iteration_table, const double *r, const double *i, const int max) {
    // Retrieve the coordinate from the pixel map
    double  c_r = r[blockIdx.x * SIZE + blockIdx.y],
            c_i = i[blockIdx.x * SIZE + blockIdx.y];

    double temp_z_r, z_r = c_r;
    double temp_z_i, z_i = c_i;

    int iteration;
    for (iteration = 0; iteration < max && (z_r * z_r + z_i * z_i <= 16.); iteration++) {
        // z*z + c
        temp_z_r = (z_r * z_r) - (z_i * z_i) + c_r;
        temp_z_i = (z_r * z_i * 2.0        ) + c_i;

        z_r = temp_z_r, z_i = temp_z_i;
    }

    iteration_table[blockIdx.x * SIZE + blockIdx.y] = iteration;
}

/**
 * Transform an integer table into their 3 component color [255, 255, 255]
 * @param pixel_table 3 * pixel numbers
 * @param iteration_table
 * @param max_loop
 */
__global__ void colorize(unsigned char *pixel_table, const int *iteration_table, const int max_loop) {
    int iteration = iteration_table[blockIdx.x * SIZE + blockIdx.y];

    if (iteration == max_loop) {
        RGB(pixel_table, (blockIdx.x * SIZE + blockIdx.y) * 3, 0u, 0u, 0u)
        return;
    }

    double h = cfmod(iteration * 360. / COLOR_LOOP, 360.);
    double hp = h / 60.0;
    double z = 1.0 - cabs(cfmod(hp, 2.) - 1.);

    if (hp < 1.0) {
        RGB(pixel_table, (blockIdx.x * SIZE + blockIdx.y) * 3, F_FACTOR, C_FACTOR * z, Z_FACTOR)
    } else if (hp < 2.0) {
        RGB(pixel_table, (blockIdx.x * SIZE + blockIdx.y) * 3, C_FACTOR * z, F_FACTOR, Z_FACTOR)
    } else if (hp < 3.0) {
        RGB(pixel_table, (blockIdx.x * SIZE + blockIdx.y) * 3, Z_FACTOR, F_FACTOR, C_FACTOR * z)
    } else if (hp < 4.0) {
        RGB(pixel_table, (blockIdx.x * SIZE + blockIdx.y) * 3, Z_FACTOR, C_FACTOR * z, F_FACTOR)
    } else if (hp < 5.0) {
        RGB(pixel_table, (blockIdx.x * SIZE + blockIdx.y) * 3, C_FACTOR * z, Z_FACTOR, F_FACTOR)
    } else if (hp < 6.0) {
        RGB(pixel_table, (blockIdx.x * SIZE + blockIdx.y) * 3, F_FACTOR, Z_FACTOR, C_FACTOR * z)
    } else {
        RGB(pixel_table, (blockIdx.x * SIZE + blockIdx.y) * 3, Z_FACTOR, Z_FACTOR, Z_FACTOR)
    }
}