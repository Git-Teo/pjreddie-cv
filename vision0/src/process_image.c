#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

float get_pixel(image im, int x, int y, int c)
{
    x = max(0, min(x, im.w));
    y = max(0, min(y, im.h));
    printf("%d %d\n", im.w, im.h);
    return im.data[x+y*im.h+im.h*im.w*c];
}

void set_pixel(image im, int x, int y, int c, float v)
{
    if (x < 0 || x >= im.w || y < 0 || y >= im.h) {
        return;
    }
    im.data[x+y*im.w+im.h*im.w*c] = v;
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    for (int x=0; x < im.w; x++) {
        for (int y=0; y < im.h; y++) {
            for (int c=0; c < im.c; c++) {
                set_pixel(copy, x, y, c, get_pixel(im, x, y, c));
            }
        }
    }
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
    float g = 0;
    for (int x=0; x < im.w; x++) {
        for (int y=0; y < im.h; y++) {
            g = get_pixel(im, x, y, 0) * 0.299 + get_pixel(im, x, y, 1) * 0.587 + get_pixel(im, x, y, 2) * 0.114;
            set_pixel(gray, x, y, 0, g);
        }
    }
    return gray;
}

void shift_image(image im, int c, float v)
{
    for (int x=0; x < im.w; x++) {
        for (int y=0; y < im.h; y++) {
            for (int c=0; c < im.c; c++) {
                set_pixel(im, x, y, c, get_pixel(im, x, y, c) + v);
            }
        }
    }
}

void clamp_image(image im)
{
    for (int x=0; x < im.w; x++) {
        for (int y=0; y < im.h; y++) {
            for (int c=0; c < im.c; c++) {
                set_pixel(im, x, y, c, fmin(1, fmax(0, get_pixel(im, x, y, c))));
            }
        }
    }
}


// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    // TODO Fill this in
}

void hsv_to_rgb(image im)
{
    // TODO Fill this in
}
