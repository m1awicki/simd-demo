#include "PngHelper.hpp"
#include "libpng/png.h"
#include <algorithm>
#include <cassert>

namespace simd_demo
{
    PngHelper::PngHelper(const char* name)
        : m_data(nullptr)
        , m_width(0)
        , m_height(0)
        , m_channels(4)
    {
        FILE* f = fopen(name, "rb");
        assert(f);

        unsigned char magic[8];
        auto read = fread(magic, 1, 8, f);
        assert(read == 8);

        assert(png_sig_cmp(magic, 0, 8) == 0);

        png_structp pngPtr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        png_infop infoPtr = png_create_info_struct(pngPtr);
        setjmp(png_jmpbuf(pngPtr));

        int colorType, interlaceType;
        png_init_io(pngPtr, f);
        png_set_sig_bytes(pngPtr, read);

        int bitDepth = 0;
        png_read_info(pngPtr, infoPtr);
        png_get_IHDR(pngPtr, infoPtr, &m_width, &m_height, &bitDepth, &colorType, &interlaceType, NULL, NULL);

        //strip 16bit channel to 8bit channel
        png_set_strip_16(pngPtr);
        //expand paletted images to RGB (chunky) format
        if (colorType == PNG_COLOR_TYPE_PALETTE)
        {
            png_set_palette_to_rgb(pngPtr);
        }
        //expand less-than-8-bit images to 8-bit per channel
        else if (colorType == PNG_COLOR_TYPE_GRAY && bitDepth < 8)
        {
            png_set_expand_gray_1_2_4_to_8(pngPtr);
        }
        //if transparency chunks bit is valid, expand transparency chunks to alpha channel
        if (png_get_valid(pngPtr, infoPtr, PNG_INFO_tRNS))
        {
            png_set_tRNS_to_alpha(pngPtr);
        }
        //expand grayscale images to 24-bit RGB
        if (colorType == PNG_COLOR_TYPE_GRAY_ALPHA)
        {
            png_set_gray_to_rgb(pngPtr);
        }
        //set pixel order to BGR
        /*if (bgr)
        {
            png_set_bgr(png_ptr);
        }*/

        switch (colorType)
        {
        case PNG_COLOR_TYPE_PALETTE:
            if (!png_get_valid(pngPtr, infoPtr, PNG_INFO_tRNS))
            {
                png_set_filler(pngPtr, 0xff, PNG_FILLER_AFTER);
                //m_alpha = false;
            }
            break;
        case PNG_COLOR_TYPE_GRAY_ALPHA:
            png_set_gray_to_rgb(pngPtr);
            break;
        case PNG_COLOR_TYPE_RGB:
            png_set_filler(pngPtr, 0xff, PNG_FILLER_AFTER);
            //m_alpha = false;
            break;
        default:
            break;
        }

        // assume 4 channels
        uint8_t* ptr = m_data = new uint8_t[m_width * m_height * m_channels];
        for (uint32_t i = 0; i < m_height; i++)
        {
            png_read_rows(pngPtr, (png_bytepp)&ptr, nullptr, 1);
            ptr += (m_width*4);
        }
        png_read_end(pngPtr, infoPtr);
        png_destroy_read_struct(&pngPtr, &infoPtr, nullptr);
        fclose(f);
    }

    PngHelper::PngHelper(uint8_t* data, uint32_t width, uint32_t height, int channels)
        : m_data(data)
        , m_width(width)
        , m_height(height)
        , m_channels(channels)
    {
        auto size = m_width * m_height * m_channels;
        m_data = new uint8_t[size];
        std::copy(data, data + size, m_data);
    }

    PngHelper::~PngHelper()
    {
        delete[] m_data;
    }

    void PngHelper::Write(const char* filename)
    {
        FILE* f = fopen(filename, "wb");
        assert(f);

        png_structp pngPtr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        assert(pngPtr);

        png_infop infoPtr = png_create_info_struct(pngPtr);
        assert(infoPtr);

        setjmp(png_jmpbuf(pngPtr));
        png_init_io(pngPtr, f);
        png_set_IHDR(pngPtr, infoPtr, m_width, m_height, 8, PNG_COLOR_TYPE_RGB_ALPHA /*colorType*/,
            PNG_INTERLACE_NONE /*interlace*/, PNG_COMPRESSION_TYPE_BASE /*compression*/,
            PNG_FILTER_TYPE_BASE /*filter*/);

        png_write_info(pngPtr, infoPtr);

        uint32_t* ptr = (uint32_t*)m_data;
        for (int i = 0; i < m_height; i++)
        {
            png_write_rows(pngPtr, (png_bytepp)(&ptr), 1);
            ptr += m_width;
        }


        png_write_end(pngPtr, nullptr);

        png_destroy_write_struct(&pngPtr, &infoPtr);
        fclose(f);
    }

    const uint8_t* PngHelper::Data() const
    {
        return m_data;
    }

    uint32_t PngHelper::Height() const
    {
        return m_height;
    }

    uint32_t PngHelper::Width() const
    {
        return m_width;
    }
}