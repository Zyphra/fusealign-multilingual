// RIFF/WAVE reader with libsndfile's (soundfile.read(..., dtype="float32")) sample scaling.
#include "wav.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace fusealign {

namespace {

constexpr uint16_t FORMAT_PCM = 0x0001;
constexpr uint16_t FORMAT_IEEE_FLOAT = 0x0003;
constexpr uint16_t FORMAT_EXTENSIBLE = 0xFFFE;

uint16_t le16(const unsigned char * p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
uint32_t le32(const unsigned char * p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

}  // namespace

Audio read_wav(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const auto fail = [&](const std::string & why) { return std::runtime_error(path + ": " + why); };
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        throw fail("not a RIFF/WAVE file");
    }

    bool have_fmt = false;
    uint16_t format = 0, channels = 0, block_align = 0, bits = 0;
    uint32_t sample_rate = 0;
    const unsigned char * data = nullptr;
    size_t data_size = 0;
    size_t pos = 12;
    while (pos + 8 <= bytes.size()) {
        const unsigned char * header = bytes.data() + pos;
        const size_t size = le32(header + 4);
        const size_t body = pos + 8;
        const size_t available = bytes.size() - body;
        if (std::memcmp(header, "fmt ", 4) == 0) {
            if (size < 16 || size > available) {
                throw fail("truncated fmt chunk");
            }
            const unsigned char * f = bytes.data() + body;
            format = le16(f);
            channels = le16(f + 2);
            sample_rate = le32(f + 4);
            block_align = le16(f + 12);
            bits = le16(f + 14);
            if (format == FORMAT_EXTENSIBLE) {
                if (size < 40) {
                    throw fail("truncated WAVE_FORMAT_EXTENSIBLE fmt chunk");
                }
                format = le16(f + 24);  // first two bytes of the sub-format GUID
            }
            have_fmt = true;
        } else if (std::memcmp(header, "data", 4) == 0) {
            data = bytes.data() + body;
            // Streaming writers leave the size at 0 or 0xFFFFFFFF; like libsndfile, read what the file holds.
            data_size = size == 0 || size > available ? available : size;
            if (have_fmt) {
                break;
            }
        }
        if (size > available) {
            break;
        }
        pos = body + size + (size & 1);  // chunks are padded to an even size
    }
    if (!have_fmt) {
        throw fail("no fmt chunk");
    }
    if (!data) {
        throw fail("no data chunk");
    }
    if (channels == 0 || sample_rate == 0) {
        throw fail("invalid fmt chunk (zero channels or sample rate)");
    }
    if (format != FORMAT_PCM && format != FORMAT_IEEE_FLOAT) {
        throw fail("unsupported WAVE format 0x" + [&] {
            char hex[8];
            std::snprintf(hex, sizeof(hex), "%04X", format);
            return std::string(hex);
        }() + " (compressed); only PCM and IEEE float are supported");
    }
    const size_t width = block_align / channels;  // bytes per sample
    if (width == 0 || block_align % channels != 0) {
        throw fail("invalid block alignment");
    }
    if (format == FORMAT_PCM ? (width < 1 || width > 4) : (width != 4 && width != 8)) {
        throw fail("unsupported sample width of " + std::to_string(bits) + " bits");
    }

    const size_t frames = data_size / block_align;
    Audio audio;
    audio.sample_rate = static_cast<int>(sample_rate);
    audio.channels = channels;
    audio.samples.resize(frames * channels);
    float * out = audio.samples.data();
    const size_t n = audio.samples.size();
    const unsigned char * p = data;
    if (format == FORMAT_IEEE_FLOAT && width == 4) {
        for (size_t i = 0; i < n; ++i, p += 4) {
            const uint32_t u = le32(p);
            std::memcpy(out + i, &u, 4);
        }
    } else if (format == FORMAT_IEEE_FLOAT) {
        for (size_t i = 0; i < n; ++i, p += 8) {
            const uint64_t u = le32(p) | (static_cast<uint64_t>(le32(p + 4)) << 32);
            double d;
            std::memcpy(&d, &u, 8);
            out[i] = static_cast<float>(d);
        }
    } else if (width == 1) {  // unsigned
        for (size_t i = 0; i < n; ++i, ++p) {
            out[i] = static_cast<float>(static_cast<int>(*p) - 128) * (1.0f / 128.0f);
        }
    } else if (width == 2) {
        for (size_t i = 0; i < n; ++i, p += 2) {
            out[i] = static_cast<float>(static_cast<int16_t>(le16(p))) * (1.0f / 32768.0f);
        }
    } else if (width == 3) {
        for (size_t i = 0; i < n; ++i, p += 3) {
            const int32_t x = static_cast<int32_t>((static_cast<uint32_t>(p[0]) << 8) | (static_cast<uint32_t>(p[1]) << 16) |
                                                   (static_cast<uint32_t>(p[2]) << 24));
            out[i] = static_cast<float>(x) * (1.0f / 2147483648.0f);
        }
    } else {
        for (size_t i = 0; i < n; ++i, p += 4) {
            out[i] = static_cast<float>(static_cast<int32_t>(le32(p))) * (1.0f / 2147483648.0f);
        }
    }
    return audio;
}

}  // namespace fusealign
