#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#ifdef _MSC_VER
#include "getopt/getopt.h"
#include <intrin.h>
#else
#include <unistd.h>
#include <getopt.h>
#if defined __x86_64__ || defined __i386__
#include <x86intrin.h>
#elif defined __ARM_NEON__
#include <arm_neon.h>
#endif
#endif // __MSC_VER

#include "PngHelper.hpp"

constexpr int RunCount = 9;
constexpr int MedianIdx = RunCount / 2;

uint64_t GetTime()
{
    return std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
}

namespace convolution_filter
{
    const uint8_t identity[] = { 0, 0, 0,
                                 0, 1, 0,
                                 0, 0, 0 };

    const int sharpen[] = { 0, -1, 0,
                            -1, 5, -1,
                            0, -1, 0 };

    const int vprewitt[] = { 1, 0, -1,
                             1, 0, -1,
                             1, 0, -1 };

    const int hprewitt[] = { -1, -1, -1,
                              0, 0, 0,
                             1, 1, 1 };

    const int vsobel[] = { -1, 0, 1,
                           -2, 0, 2,
                           -1, 0, 1 };

    const int hsobel[] = { -1, -2, -1,
                            0, 0, 0,
                            1, 2, 1 };
}

template <typename F>
void ApplyFilter(const uint8_t* input, uint8_t* output, int w, int h, const F filter[], int filterSize )
{
    const int offset = filterSize / 2;
    uint32_t* curr = (uint32_t*)input;
    uint32_t* out = (uint32_t*)output;
#ifdef __AVX2__
    //mult alpha channel by 0 so it can be safely added
    __m256i f1r0 = _mm256_set_epi16(0, 0, 0, 0, 0, filter[2], filter[1], filter[0],
        0, filter[2], filter[1], filter[0], 0, filter[2], filter[1], filter[0]);
    __m256i f1r1 = _mm256_set_epi16(0, 0, 0, 0, 0, filter[5], filter[4], filter[3],
        0, filter[5], filter[4], filter[3], 0, filter[5], filter[4], filter[3]);
    __m256i f1r2 = _mm256_set_epi16(0, 0, 0, 0, 0, filter[8], filter[7], filter[6],
        0, filter[8], filter[7], filter[6], 0, filter[8], filter[7], filter[6]);
#endif
    for (int i = 0; i < h - 2; i++)
    {
        for (int j = 0; j < w - offset-1; j++)
        {
#ifdef __AVX2__
            //load data
            uint32_t* tmp = curr;
            __m128i d0 = _mm_loadu_si128(((__m128i*)tmp) + 0);
            tmp += w;
            __m128i d1 = _mm_loadu_si128(((__m128i*)tmp));
            tmp += w;
            __m128i d2 = _mm_loadu_si128(((__m128i*)tmp));

            //do shuffle while it is still 8bit per channel
            __m128i smask = _mm_set_epi8(128, 11, 7, 3, 128, 10, 6, 2, 128, 9, 5, 1, 128, 8, 4, 0);
            __m128i d0s = _mm_shuffle_epi8(d0, smask);
            __m128i d1s = _mm_shuffle_epi8(d1, smask);
            __m128i d2s = _mm_shuffle_epi8(d2, smask);

            //convert to 16bit and mult with filter row
            __m256i r0_16 = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(d0s), f1r0);
            __m256i r1_16 = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(d1s), f1r1);
            __m256i r2_16 = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(d2s), f1r2);

            //sum rows
            __m256i rowsum = _mm256_add_epi16(_mm256_add_epi16(r0_16, r1_16), r2_16);

            //add horizontally
            __m256i hz = _mm256_hadd_epi16(rowsum, _mm256_set1_epi32(0));         // RR GG BB AA 00 00 00 00
            __m128i hz128 = _mm256_extracti128_si256(hz, 0);                      // RR GG BB AA
            __m128i hz2 = _mm_hadd_epi16(hz128, _mm256_extractf128_si256(hz, 1)); // RG BA 00 00

            //clamp values to 0-255
            __m128i clamped = _mm_max_epi16(_mm_min_epi16(hz2, _mm_set1_epi16(255)), _mm_set1_epi16(0));
            __m128i clampedsh = _mm_shuffle_epi8(clamped, _mm_set_epi8(128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 8, 2, 0));

            int rgb = _mm_extract_epi32(clampedsh, 0);
            *out = rgb | ( 0xff << 24);
#else
            uint8_t pixCh[3][12] = { 0 };
            uint32_t* tmp = curr;
            for (int r = 0; r < 3; r++)
            {
                tmp = curr + (r * w);
                memcpy(&pixCh[r][0], tmp, 12);
            }

            int rgb[3] = { 0 };
            int idx = 0;
            for (int row = 0; row < 3; row++)
            {
                for (int c = 0; c < 12; c++)
                {
                    int modulo = c % 4;
                    if (modulo == 3)
                    {
                        ++idx;
                    }
                    else
                    {
                        rgb[modulo] += pixCh[row][c] * filter[idx];
                    }
                }
            }

            rgb[0] = std::clamp(rgb[0], 0, 255);
            rgb[1] = std::clamp(rgb[1], 0, 255);
            rgb[2] = std::clamp(rgb[2], 0, 255);
            *out = (uint8_t)rgb[0] | ((uint8_t)rgb[1] << 8) | ((uint8_t)rgb[2] << 16) | (0xff << 24);
#endif
            ++out;
            ++curr;
        }
        curr += (2*offset);
    }
}

void PrintUsage()
{
    fprintf( stderr, "Usage: simd-demo [options] input.png {output.png}\n" );
    fprintf( stderr, "  Options:\n" );
    fprintf( stderr, "  -s                     apply sharpen filter\n" );
    fprintf( stderr, "  -l                     apply sobel filter\n");
    fprintf( stderr, "  -b                     benchmark mode\n" );
}

enum Options
{
    Sharp,
    Sobel
};

struct option lopts[] = {
    { "sharp", no_argument, nullptr, Sharp },
    { "sobel", no_argument, nullptr, Sobel },
    {}
};

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        PrintUsage();
        return -1;
    }
    else
    {
        bool sharpen = true;
        bool benchmark = false;
        int c;
        while( ( c = getopt_long( argc, argv, "slb", lopts, nullptr ) ) != -1)
        {
            switch( c )
            {
            case '?':
                PrintUsage();
                return 1;
            case Sharp:
            case 's':
                sharpen = true;
                break;
            case Sobel:
            case 'l':
                sharpen = false;
                break;
            case 'b':
                benchmark = true;
                break;
            default:
                break;
            }
        }

        const char* input = nullptr;
        const char* output = nullptr;
        if (benchmark)
        {
            if (argc - optind < 1)
            {
                PrintUsage();
                return 1;
            }
            input = argv[optind];
        }
        else
        {
            if (argc - optind < 2)
            {
                PrintUsage();
                return 1;
            }
            input = argv[optind];
            output = argv[optind + 1];
        }
        simd_demo::PngHelper png(input);
        uint32_t* result = new uint32_t[png.Width() * png.Height()];
        const int filterSize = 3;
        auto filter = sharpen ? convolution_filter::sharpen : convolution_filter::vsobel;
        if (benchmark)
        {
            uint64_t timeData[RunCount];
            for (int i = 0; i < RunCount; i++)
            {
                const auto start = GetTime();
                ApplyFilter(png.Data(), (uint8_t*)result, png.Width(), png.Height(), filter, filterSize);
                const auto end = GetTime();
                timeData[i] = end - start;
            }
            std::sort(timeData, timeData + RunCount);
            const auto median = timeData[MedianIdx] / 1000.f;
            std::cout << "filter: " << (sharpen ? "sharpen" : "sobel") << std::endl;
            std::cout << "benchmark: " << (benchmark ? "yes" : "no") << std::endl;
            printf("Median filtering time for %i runs: %0.3f ms (%0.3f Mpx/s) single threaded\n", RunCount, median, png.Width() *png.Height() / (median * 1000));
        }
        else
        {
            ApplyFilter(png.Data(), (uint8_t*)result, png.Width(), png.Height(), filter, filterSize);
            simd_demo::PngHelper filtered((uint8_t*)result, png.Width() - (filterSize / 2) * 2, png.Height() - (filterSize / 2) * 2, 4);
            filtered.Write(output);
        }
        delete[] result;
    }

    return 0;
}
