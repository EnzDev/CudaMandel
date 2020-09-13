#include <sstream>
#include <lodepng.h>
#include "png.cuh"

void write_frame(const unsigned char *pixels, unsigned int frame, const int size) {
    std::string formatFrame = std::to_string(frame);
    formatFrame.insert(formatFrame.begin(), 4 - formatFrame.length(), '0');

    std::stringstream imageName;
    imageName << "smal/mand_" << size << "_" << formatFrame << ".png";

    lodepng_encode24_file(imageName.str().data(), pixels, size, size);
}