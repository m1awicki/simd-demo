#pragma once
#include <cstdint>

namespace simd_demo
{
    class PngHelper
    {
    public:
        PngHelper(const char* name);
        PngHelper(uint8_t* data, uint32_t width, uint32_t height, int channels);
        ~PngHelper();

        void Write(const char* filename);

        const uint8_t* Data() const;
        uint32_t Width() const;
        uint32_t Height() const;

    private:
        uint8_t* m_data;
        uint32_t m_width;
        uint32_t m_height;
        int m_channels;
    };
}
