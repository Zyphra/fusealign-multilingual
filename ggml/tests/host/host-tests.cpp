// Unit tests of the host-side pieces: ports of tests/test_longform.py (FillUnplacedTests,
// UnroutableTests) plus text, WAV and resampling edge cases.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "longform.h"
#include "resample.h"
#include "text.h"
#include "wav.h"

using namespace fusealign;

namespace {

int failures = 0;

void check(bool ok, const std::string & what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << "\n";
    }
}

template <typename F>
void check_throws(F && f, const std::string & what) {
    try {
        f();
    } catch (const std::invalid_argument &) {
        return;
    } catch (const std::runtime_error &) {
        return;
    }
    check(false, what + " (no exception)");
}

using Opt = std::optional<Interval>;

void fill_unplaced_tests() {
    {
        const auto got = fill_unplaced({std::nullopt, Interval{1, 2}, std::nullopt, std::nullopt, Interval{5, 6}, std::nullopt}, 8);
        const std::vector<Interval> want = {{0, 1}, {1, 2}, {2, 3.5}, {3.5, 5}, {5, 6}, {6, 8}};
        check(got == want, "fill_unplaced: null runs split their gap and placed stay verbatim");
    }
    {
        const auto got = fill_unplaced(std::vector<Opt>(3), 3);
        const std::vector<Interval> want = {{0, 1}, {1, 2}, {2, 3}};
        check(got == want, "fill_unplaced: all unplaced divides the clip equally");
    }
    {
        // Overlapping placed neighbours are kept verbatim; the fill between them may be inverted.
        const auto got = fill_unplaced({Interval{0, 3}, std::nullopt, Interval{2, 4}}, 5);
        const std::vector<Interval> want = {{0, 3}, {3, 2}, {2, 4}};
        check(got == want, "fill_unplaced: no clamping between overlapping neighbours");
    }
    check(fill_unplaced({}, 1.0).empty(), "fill_unplaced: empty");
}

void unroutable_tests() {
    {
        const std::string big(1023, 'x');
        std::vector<std::string> words = {"a", big, "b"};
        std::vector<Interval> approx = {{0.0, 1.0}, {1.0, 2.0}, {2.0, 3.0}};
        for (int i = 0; i < 11; ++i) {
            words.push_back(std::string(100, 'y'));
            approx.push_back({5.0, 6.0});
        }
        words.push_back("c");
        approx.push_back({7.0, 8.0});
        // 11 x 100 bytes + 10 spaces = 1,110 bytes share one start: no time cut separates them.
        std::set<size_t> want = {1};
        for (size_t i = 3; i < 14; ++i) {
            want.insert(i);
        }
        check(unroutable(words, approx) == want, "unroutable: oversized word and oversized equal-start run only");
    }
    {
        const std::vector<std::string> words(10, std::string(100, 'y'));
        check(unroutable(words, std::vector<Interval>(10, {5.0, 6.0})).empty(), "unroutable: equal-start run that fits is kept");
    }
    {
        // Counts bytes, not code points: 341 x 3-byte characters = 1,023 bytes.
        std::string thai;
        for (int i = 0; i < 341; ++i) {
            thai += "\xE0\xB8\xAB";
        }
        check(unroutable({"a", thai}, {{0.0, 1.0}, {1.0, 2.0}}) == std::set<size_t>{1}, "unroutable: UTF-8 bytes");
    }
}

void plan_chunks_tests() {
    // 60 words over 35 s with no speech regions: two cores, every word owned once.
    std::vector<std::string> words;
    for (int i = 0; i < 60; ++i) {
        words.push_back("word" + std::to_string(i));
    }
    const auto approx = approximate_times(words, {}, {{0.0, 35.0}}, 35.0);
    check(approx.size() == words.size(), "approximate_times: one interval per word");
    const auto chunks = plan_chunks(words, approx, {{0.0, 35.0}}, 35.0);
    check(chunks.size() == 2, "plan_chunks: 35 s gives two cores");
    check(!chunks.empty() && chunks.front().core.first == 0.0 && chunks.back().core.second == 35.0, "plan_chunks: cores cover the clip");
    check_throws([&] { plan_chunks(words, {}, {}, 35.0); }, "plan_chunks: count mismatch raises");
    check_throws([&] { plan_chunks({std::string(1023, 'x')}, {{0.0, 1.0}}, {}, 35.0); }, "plan_chunks: oversized word raises");

    // A word predicted with its centre inside a core beats one outside it.
    std::vector<Chunk> two = {{{0.0, 10.0}, {0.0, 12.0}, {0}}, {{10.0, 20.0}, {8.0, 20.0}, {0}}};
    const auto stitched = stitch_predictions(1, two, {{Interval{11.0, 11.5}}, {Interval{10.5, 11.0}}});
    check(stitched.size() == 1 && stitched[0] && *stitched[0] == Interval(10.5, 11.0), "stitch_predictions: in-core centre wins");
    check_throws([&] { stitch_predictions(1, two, {{Interval{1.0, 2.0}}}); }, "stitch_predictions: count mismatch raises");
}

void text_tests() {
    using V = std::vector<std::string>;
    check(split_whitespace("") == V{}, "split: empty");
    check(split_whitespace(" \t\n ") == V{}, "split: only whitespace");
    check(split_whitespace("  a\tb\xE3\x80\x80" "c\xC2\x85" "d \x1C" "e  ") == V{"a", "b", "c", "d", "e"},
          "split: ASCII, U+3000, U+0085, U+001C separators");
    check(split_whitespace("a\xE2\x80\x8B" "b") == V{"a\xE2\x80\x8B" "b"}, "split: U+200B is not whitespace");
    check(split_whitespace("\xE4\xB8\xAD\xC2\xA0\xE6\x96\x87") == V{"\xE4\xB8\xAD", "\xE6\x96\x87"}, "split: U+00A0");

    check(has_whitespace("a b"), "has_whitespace: space");
    check(has_whitespace("a\xC2\xA0" "b"), "has_whitespace: U+00A0");
    check(!has_whitespace("ab"), "has_whitespace: none");
    check(!has_whitespace("a\xE2\x80\x8B" "b"), "has_whitespace: U+200B is not whitespace");
    check(!has_whitespace(""), "has_whitespace: empty");

    check(is_wordy("hello"), "is_wordy: ASCII letters");
    check(is_wordy("\xD9\xA3"), "is_wordy: U+0663 (Nd)");
    check(is_wordy("\xC2\xB2"), "is_wordy: U+00B2 (No)");
    check(is_wordy("\xE0\xB8\xAB"), "is_wordy: Thai letter");
    check(is_wordy(",a"), "is_wordy: any wordy code point");
    check(!is_wordy(","), "is_wordy: comma");
    check(!is_wordy("\xE2\x80\x94"), "is_wordy: em dash");
    check(!is_wordy("\xCC\x81"), "is_wordy: combining acute (Mn)");
    check(!is_wordy("\xF0\x9F\x99\x82"), "is_wordy: emoji");
    check(!is_wordy(""), "is_wordy: empty");

    for (const char * bad : {"\xFF", "\xC0\x80", "\xC1\xBF", "\xE0\x80\x80", "\xED\xA0\x80", "\xF4\x90\x80\x80",
                             "\xF5\x80\x80\x80", "\xE4\xB8", "a\x80", "\xE4\x41\xAD"}) {
        check_throws([&] { check_utf8(bad); }, std::string("check_utf8 rejects ") + bad);
    }
    check_throws([&] { is_wordy("a\xFF"); }, "is_wordy: invalid UTF-8 raises");
    check_throws([&] { split_whitespace("a \xC0\x80"); }, "split: invalid UTF-8 raises");
    try {
        check_utf8("\xE4\xB8\xAD\xF0\x9F\x99\x82\xED\x9F\xBF\xEE\x80\x80\xF4\x8F\xBF\xBF");
    } catch (const std::exception & e) {
        check(false, std::string("check_utf8 accepts valid UTF-8: ") + e.what());
    }
    check(decode_utf8("a\xC4\xB0\xE2\x84\xAA") == std::u32string{U'a', 0x130, 0x212A}, "decode_utf8");
    check(ascii_key(U'A') == 'a' && ascii_key(U'7') == '7' && ascii_key(0x130) == 'i' && ascii_key(0x212A) == 'k' &&
              ascii_key(U'-') == '\0' && ascii_key(0xFF21) == '\0',
          "ascii_key");
}

void write_le(std::ofstream & out, uint64_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) {
        out.put(static_cast<char>((value >> (8 * i)) & 0xFF));
    }
}

void wav_tests() {
    const std::string path = "test-host-tmp.wav";
    {
        std::ofstream out(path, std::ios::binary);
        const int16_t samples[] = {0, 1, -1, 32767, -32768, 12345};
        out.write("RIFF", 4);
        write_le(out, 4 + (8 + 3 + 1) + (8 + 16) + (8 + sizeof(samples)), 4);
        out.write("WAVE", 4);
        out.write("LIST", 4);  // unknown chunk with an odd size and its pad byte
        write_le(out, 3, 4);
        out.write("abc\0", 4);
        out.write("fmt ", 4);
        write_le(out, 16, 4);
        write_le(out, 1, 2);       // PCM
        write_le(out, 2, 2);       // channels
        write_le(out, 22050, 4);   // rate
        write_le(out, 22050 * 4, 4);
        write_le(out, 4, 2);       // block align
        write_le(out, 16, 2);
        out.write("data", 4);
        write_le(out, sizeof(samples), 4);
        for (const int16_t s : samples) {
            write_le(out, static_cast<uint16_t>(s), 2);
        }
    }
    const Audio audio = read_wav(path);
    check(audio.sample_rate == 22050 && audio.channels == 2 && audio.samples.size() == 6, "read_wav: header");
    const std::vector<float> want = {0.0f, 1.0f / 32768, -1.0f / 32768, 32767.0f / 32768, -1.0f, 12345.0f / 32768};
    check(audio.samples == want, "read_wav: PCM16 scaling");
    {
        std::ofstream out(path, std::ios::binary);
        out.write("RIFF", 4);
        write_le(out, 4 + 8 + 16 + 8, 4);
        out.write("WAVE", 4);
        out.write("fmt ", 4);
        write_le(out, 16, 4);
        write_le(out, 2, 2);  // MS ADPCM
        write_le(out, 1, 2);
        write_le(out, 16000, 4);
        write_le(out, 16000, 4);
        write_le(out, 1, 2);
        write_le(out, 4, 2);
        out.write("data", 4);
        write_le(out, 0, 4);
    }
    check_throws([&] { read_wav(path); }, "read_wav: compressed format rejected");
    check_throws([&] { read_wav("does-not-exist.wav"); }, "read_wav: missing file");
    std::remove(path.c_str());
}

void resample_tests() {
    const std::vector<float> x = {0.5f, -0.25f, 1.0f};
    check(resample(x.data(), x.size(), 16000, 16000) == x, "resample: same rate is the identity");
    std::vector<float> tone(48000);
    for (size_t i = 0; i < tone.size(); ++i) {
        tone[i] = static_cast<float>(std::sin(2 * 3.141592653589793 * 440.0 * static_cast<double>(i) / 48000.0));
    }
    const auto y = resample(tone.data(), tone.size(), 48000, 16000);
    check(y.size() == 16000, "resample: 48 kHz -> 16 kHz length");
    double err = 0.0;
    for (size_t i = 100; i + 100 < y.size(); ++i) {
        err = std::max(err, std::fabs(y[i] - std::sin(2 * 3.141592653589793 * 440.0 * static_cast<double>(i) / 16000.0)));
    }
    check(err < 1e-2, "resample: 440 Hz tone survives");
    check(resample(tone.data(), 1, 44100, 16000).size() == 1, "resample: ceil(new * len / orig)");
}

}  // namespace

int main() {
    fill_unplaced_tests();
    unroutable_tests();
    plan_chunks_tests();
    text_tests();
    wav_tests();
    resample_tests();
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "test-host: all passed\n";
    return 0;
}
