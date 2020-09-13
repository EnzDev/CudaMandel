#include <iostream>
#include <chrono>

#include "png.cuh"
#include "constants.cuh"
#include "mandelbrot.cuh"


#define ZOOM_FACTOR 1.01

using namespace std::chrono; // Prevent the hassle of having this long prefix

int main() {
    int max_loop = 100;

    // Storage for iteration result out of the Mandelbrot function
    int *iteration_table;

    // Storage for pixel representation
    unsigned char *pixel_table;

    // Storages for real and imaginary coordinates to process
    double *r, *i;

    // Initialize image storage for iteration numbers and pixel rendering
    cudaMallocManaged(&iteration_table, SIZE * SIZE * sizeof(int));
    cudaMallocManaged(&pixel_table, 3 * SIZE * SIZE * sizeof(unsigned char));

    cudaMallocManaged(&r, SIZE * SIZE * sizeof(double));
    cudaMallocManaged(&i, SIZE * SIZE * sizeof(double));

    // Create a grid of SIZE*SIZE for the gpu functions
    dim3 blockSize = dim3(SIZE, SIZE);

    // Width of the real view
    double zoom = 2.;
    double zf = ZOOM_FACTOR;
    unsigned int frame = 0;

    // Skip this amount of frames
    steady_clock::time_point preframe = steady_clock::now();
    for (; frame < 8000; frame++) {
        zoom = zoom / zf;
        max_loop += 2;
    }
    std::cout << "Prepared frame to " << frame << " in "
    << duration_cast<microseconds>(steady_clock::now() - preframe).count() << "us" << std::endl;


    // Render frames until the max frame allowed
    for (; frame < 100000; frame++) {
        steady_clock::time_point begin = steady_clock::now();

        // Fill the table with initial coordinates
        // translate<<<blockSize, 1>>>(r, i, zoom, zoom, -1.7499, 0.); // Another interesting zoom point
        // translate<<<blockSize, 1>>>(r, i, zoom, zoom, -0.16070135, 1.0375665); // Another interesting zoom point
        translate<<<blockSize, 1>>>(r, i, zoom, zoom, 0.281717921930775, 0.5771052841488505);

        cudaDeviceSynchronize();
        steady_clock::time_point translated = steady_clock::now();

        // Compute Mandelbrot for each pixel
        mandelbrot<<<blockSize, 1>>>(iteration_table, r, i, max_loop);
        cudaDeviceSynchronize();
        steady_clock::time_point mandelbrotT = steady_clock::now();

        // Transform the iterations into beautiful colors
        colorize<<<blockSize, 1>>>(pixel_table, iteration_table, max_loop);
        cudaDeviceSynchronize();
        steady_clock::time_point color = steady_clock::now();

        // Store the image into a PNG
        write_frame(pixel_table, frame, SIZE);
        steady_clock::time_point end = steady_clock::now();

        // Print timings
        std::cout << "\rDone at zoom " << zoom << " frame " << frame
                  << ". Rendered in "   << duration_cast<milliseconds>( end - begin              ).count() << "ms"
                  << ", translated in " << duration_cast<microseconds>( translated - begin       ).count() << "us"
                  << ", mandelbrot in " << duration_cast<milliseconds>( mandelbrotT - translated ).count() << "ms"
                  << ", colorized in "  << duration_cast<microseconds>( color - mandelbrotT      ).count() << "us"
                  << ", stored in "     << duration_cast<milliseconds>( end - color              ).count() << "ms"
                  << std::flush;

        // Zoom more
        zoom = zoom / ZOOM_FACTOR;
        max_loop += 2;
    }

    // Write 3 more frame with the last frame
    write_frame(pixel_table, ++frame, SIZE);
    write_frame(pixel_table, ++frame, SIZE);
    write_frame(pixel_table, frame + 1, SIZE);

    return 0;
}

