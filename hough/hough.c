#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main() {
    int width, height, bpp;

    uint8_t* rgb_image = stbi_load("ipad.jpg", &width, &height, &bpp, 3);
    printf("Image size: %d px by %d px, bpp: %d\n", width, height, bpp);

    stbi_image_free(rgb_image);

    return 0;
}