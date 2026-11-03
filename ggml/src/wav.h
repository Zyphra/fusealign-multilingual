// RIFF/WAVE reader: PCM 8/16/24/32-bit and IEEE float 32/64, WAVE_FORMAT_EXTENSIBLE included.
// Samples are scaled as soundfile.read(..., dtype="float32") scales them (PCM16: x / 32768).
#pragma once

#include <string>
#include <vector>

namespace fusealign {

struct Audio {
    std::vector<float> samples;  // interleaved, samples.size() == frames * channels
    int sample_rate = 0;
    int channels = 0;
};

Audio read_wav(const std::string & path);  // throws std::runtime_error

}  // namespace fusealign
