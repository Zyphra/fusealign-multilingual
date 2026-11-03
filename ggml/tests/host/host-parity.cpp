// JSON-lines driver for tests/host/host_parity.py: one request per stdin line, one response per stdout line.
// Large float arrays travel as raw little-endian float32 files named in the request.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"
#include "longform.h"
#include "resample.h"
#include "text.h"
#include "vad.h"
#include "wav.h"

using json = nlohmann::json;
using namespace fusealign;

namespace {

std::vector<float> read_f32(const std::string & path, size_t n) {
    std::vector<float> data(n);
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(n * sizeof(float)));
    if (!in) {
        throw std::runtime_error("cannot read " + path);
    }
    return data;
}

void write_f32(const std::string & path, const std::vector<float> & data) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
}

std::vector<Interval> intervals_of(const json & j) {
    std::vector<Interval> out;
    for (const auto & pair : j) {
        out.emplace_back(pair.at(0).get<double>(), pair.at(1).get<double>());
    }
    return out;
}

std::vector<AsrWord> asr_of(const json & j) {
    std::vector<AsrWord> out;
    for (const auto & w : j) {
        AsrWord word;
        word.word = w.at("word").get<std::string>();
        if (w.contains("start") && !w["start"].is_null()) {
            word.start = w["start"].get<double>();
        }
        if (w.contains("end") && !w["end"].is_null()) {
            word.end = w["end"].get<double>();
        }
        out.push_back(std::move(word));
    }
    return out;
}

json intervals_json(const std::vector<Interval> & v) {
    json out = json::array();
    for (const auto & [a, b] : v) {
        out.push_back({a, b});
    }
    return out;
}

json optional_intervals_json(const std::vector<std::optional<Interval>> & v) {
    json out = json::array();
    for (const auto & iv : v) {
        out.push_back(iv ? json{iv->first, iv->second} : json(nullptr));
    }
    return out;
}

std::vector<std::optional<Interval>> optional_intervals_of(const json & j) {
    std::vector<std::optional<Interval>> out;
    for (const auto & iv : j) {
        if (iv.is_null()) {
            out.emplace_back();
        } else {
            out.emplace_back(Interval{iv.at(0).get<double>(), iv.at(1).get<double>()});
        }
    }
    return out;
}

json chunks_json(const std::vector<Chunk> & chunks) {
    json out = json::array();
    for (const auto & c : chunks) {
        out.push_back({{"core", {c.core.first, c.core.second}}, {"audio", {c.audio.first, c.audio.second}}, {"indices", c.indices}});
    }
    return out;
}

std::string utf8_of(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

std::map<std::string, std::unique_ptr<Vad>> vads;

const Vad & vad_for(const std::string & path) {
    auto & slot = vads[path];
    if (!slot) {
        slot = std::make_unique<Vad>(path);
    }
    return *slot;
}

json handle(const json & req) {
    const std::string op = req.at("op").get<std::string>();
    if (op == "approximate_times") {
        return intervals_json(approximate_times(req.at("words").get<std::vector<std::string>>(), asr_of(req.at("asr_words")),
                                                intervals_of(req.at("speech_regions")), req.at("duration").get<double>()));
    }
    if (op == "unroutable") {
        const auto skipped = unroutable(req.at("words").get<std::vector<std::string>>(), intervals_of(req.at("approx")));
        return json(std::vector<size_t>(skipped.begin(), skipped.end()));
    }
    if (op == "plan_chunks") {
        return chunks_json(plan_chunks(req.at("words").get<std::vector<std::string>>(), intervals_of(req.at("approx")),
                                       intervals_of(req.at("speech_regions")), req.at("duration").get<double>()));
    }
    if (op == "long_plan") {
        // FuseAlignAligner._long up to the forward: approximate_times -> unroutable -> plan_chunks on the rest.
        const auto words = req.at("words").get<std::vector<std::string>>();
        const auto regions = intervals_of(req.at("speech_regions"));
        const double duration = req.at("duration").get<double>();
        const auto approx = approximate_times(words, asr_of(req.at("asr_words")), regions, duration);
        const auto skipped = unroutable(words, approx);
        std::vector<size_t> keep;
        std::vector<std::string> kept_words;
        std::vector<Interval> kept_approx;
        for (size_t i = 0; i < words.size(); ++i) {
            if (!skipped.count(i)) {
                keep.push_back(i);
                kept_words.push_back(words[i]);
                kept_approx.push_back(approx[i]);
            }
        }
        auto chunks = plan_chunks(kept_words, kept_approx, regions, duration);
        for (auto & chunk : chunks) {
            for (auto & index : chunk.indices) {
                index = keep[index];
            }
        }
        return {{"approx", intervals_json(approx)},
                {"skipped", std::vector<size_t>(skipped.begin(), skipped.end())},
                {"chunks", chunks_json(chunks)}};
    }
    if (op == "stitch_predictions") {
        std::vector<Chunk> chunks;
        for (const auto & c : req.at("chunks")) {
            chunks.push_back({{c.at("core").at(0).get<double>(), c.at("core").at(1).get<double>()},
                              {c.at("audio").at(0).get<double>(), c.at("audio").at(1).get<double>()},
                              c.at("indices").get<std::vector<size_t>>()});
        }
        std::vector<std::vector<std::optional<Interval>>> predictions;
        for (const auto & p : req.at("predictions")) {
            predictions.push_back(optional_intervals_of(p));
        }
        return optional_intervals_json(stitch_predictions(req.at("n_words").get<size_t>(), chunks, predictions));
    }
    if (op == "fill_unplaced") {
        return intervals_json(fill_unplaced(optional_intervals_of(req.at("intervals")), req.at("duration").get<double>()));
    }
    if (op == "split") {
        return split_whitespace(req.at("text").get<std::string>());
    }
    if (op == "is_wordy") {
        json wordy = json::array(), space = json::array();
        for (const auto & token : req.at("tokens")) {
            wordy.push_back(is_wordy(token.get<std::string>()));
            space.push_back(has_whitespace(token.get<std::string>()));
        }
        return {{"is_wordy", wordy}, {"has_whitespace", space}};
    }
    if (op == "codepoints") {
        // One character per code point 0..0x10FFFF ('-' for surrogates, which UTF-8 cannot carry).
        std::string wordy, space, key;
        for (uint32_t cp = 0; cp <= 0x10FFFF; ++cp) {
            if (cp >= 0xD800 && cp <= 0xDFFF) {
                wordy += '-';
                space += '-';
                key += '-';
                continue;
            }
            const std::string s = utf8_of(cp);
            wordy += is_wordy(s) ? '1' : '0';
            space += has_whitespace(s) ? '1' : '0';
            const char k = ascii_key(decode_utf8(s).at(0));
            key += k ? k : '.';
        }
        return {{"is_wordy", wordy}, {"isspace", space}, {"key", key}};
    }
    if (op == "resample") {
        const auto x = read_f32(req.at("in").get<std::string>(), req.at("n").get<size_t>());
        const auto y = resample(x.data(), x.size(), req.at("orig").get<int>(), req.at("new").get<int>());
        write_f32(req.at("out").get<std::string>(), y);
        return {{"n", y.size()}};
    }
    if (op == "vad_probs" || op == "speech_regions") {
        const Vad & vad = vad_for(req.at("gguf").get<std::string>());
        const auto x = read_f32(req.at("in").get<std::string>(), req.at("n").get<size_t>());
        const auto t0 = std::chrono::steady_clock::now();
        json out;
        if (op == "vad_probs") {
            const auto p = vad.probs(x.data(), x.size());
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            write_f32(req.at("out").get<std::string>(), p);
            out = {{"n", p.size()}, {"seconds", seconds}};
        } else {
            out = intervals_json(vad.speech_regions(x.data(), x.size()));
        }
        return out;
    }
    if (op == "read_wav") {
        const Audio audio = read_wav(req.at("path").get<std::string>());
        write_f32(req.at("out").get<std::string>(), audio.samples);
        return {{"sample_rate", audio.sample_rate}, {"channels", audio.channels}, {"n", audio.samples.size()}};
    }
    throw std::runtime_error("unknown op " + op);
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
        json response;
        try {
            response = {{"ok", handle(json::parse(line))}};
        } catch (const std::invalid_argument & e) {
            response = {{"error", "ValueError"}, {"message", e.what()}};
        } catch (const std::exception & e) {
            response = {{"error", "RuntimeError"}, {"message", e.what()}};
        }
        std::cout << response.dump(-1, ' ', false, json::error_handler_t::strict) << "\n" << std::flush;
    }
    return 0;
}
