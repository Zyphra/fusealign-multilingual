// fusealign-cli: word-level forced alignment of WAV files with a fusealign GGUF.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "fusealign.h"
#include "json.hpp"
#include "wav.h"

using json = nlohmann::json;

namespace {

const char * USAGE = R"(usage:
  fusealign-cli -m MODEL.gguf -a AUDIO.wav (--text "TRANSCRIPT" | --tokens '["tok", ...]') [options]
  fusealign-cli -m MODEL.gguf --batch INPUTS.jsonl [-o OUT.jsonl] [options]

options:
  -m, --model PATH        fusealign-gguf/1 file (f32 or q8_0)
  -d, --device DEVICE     auto (default: CUDA if available, else CPU) | cpu | cuda | cuda:N
  -t, --threads N         CPU threads (default: all)
  --anchors PATH          approximate word times for long-form audio: a JSON list of {"word","start","end"}
                          (single mode), or JSON lines {"clip_id","words":[...]} keyed by clip_id (batch mode)

Single mode prints a JSON list of {"word","start","end","presence","placed"}, one per token (times in seconds).
The transcript is split on whitespace; --tokens takes pre-segmented tokens (Chinese, Japanese, Thai, ...).

Batch mode reads JSON lines {"clip_id","audio", and one of "text" | "tokens" | "words", optional "anchors"}:
  "text"/"tokens" -> {"clip_id","words":[{"word","start","end","presence","placed"}, ...]}
  "words"         -> {"clip_id","intervals":[[start,end] | null, ...],"presence":[...]}  (raw model output;
                     16 kHz audio, no punctuation handling or fallback fill)
A row that fails, including a line that is not valid JSON, is written as {"clip_id","error"} and the exit status is 1.

Recordings over 30 s or transcripts over 1022 UTF-8 bytes are aligned long-form. Without anchors the words are
routed by the Silero VAD character-time map.
)";

[[noreturn]] void usage_error(const std::string & message) {
    std::fprintf(stderr, "fusealign-cli: %s\n\n%s", message.c_str(), USAGE);
    std::exit(2);
}

std::vector<fusealign::AsrWord> parse_anchors(const json & list) {
    std::vector<fusealign::AsrWord> anchors;
    for (const json & item : list) {
        fusealign::AsrWord word;
        word.word = item.at("word").get<std::string>();
        if (item.contains("start") && !item["start"].is_null()) word.start = item["start"].get<double>();
        if (item.contains("end") && !item["end"].is_null()) word.end = item["end"].get<double>();
        anchors.push_back(std::move(word));
    }
    return anchors;
}

std::vector<float> load_mono(const std::string & path, int & sample_rate) {
    fusealign::Audio audio = fusealign::read_wav(path);
    if (audio.channels != 1) {
        throw std::runtime_error(path + ": expected a mono recording, got " + std::to_string(audio.channels) +
                                 " channels");
    }
    sample_rate = audio.sample_rate;
    return std::move(audio.samples);
}

json words_json(const std::vector<fusealign::WordAlignment> & words) {
    json out = json::array();
    for (const auto & w : words) {
        out.push_back({{"word", w.word}, {"start", w.start}, {"end", w.end},
                       {"presence", static_cast<double>(w.presence)}, {"placed", w.placed}});
    }
    return out;
}

json placement_json(const fusealign::Placement & placed) {
    json intervals = json::array();
    for (const auto & iv : placed.intervals) {
        if (iv) intervals.push_back({iv->first, iv->second});
        else intervals.push_back(nullptr);
    }
    json presence = json::array();
    for (float p : placed.presence) presence.push_back(static_cast<double>(p));
    return {{"intervals", intervals}, {"presence", presence}};
}

// Map key of a clip_id: the string itself, or the JSON text of a non-string id.
std::string clip_key(const json & clip_id) {
    return clip_id.is_string() ? clip_id.get<std::string>() : clip_id.dump();
}

json align_row(fusealign::Aligner & aligner, const json & row, const std::vector<fusealign::AsrWord> * anchors) {
    int sample_rate = 0;
    const std::vector<float> wav = load_mono(row.at("audio").get<std::string>(), sample_rate);
    json out;
    if (row.contains("words")) {
        if (sample_rate != fusealign::SAMPLE_RATE) {
            throw std::runtime_error("\"words\" rows need 16 kHz audio; use \"tokens\" for other sample rates");
        }
        out = placement_json(aligner.intervals(wav, row["words"].get<std::vector<std::string>>(), anchors));
    } else if (row.contains("tokens")) {
        out["words"] = words_json(aligner.align(wav, sample_rate, row["tokens"].get<std::vector<std::string>>(), anchors));
    } else if (row.contains("text")) {
        out["words"] = words_json(aligner.align(wav, sample_rate, row["text"].get<std::string>(), anchors));
    } else {
        throw std::runtime_error("row needs \"text\", \"tokens\" or \"words\"");
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    fusealign::Options options;
    std::string audio, text, tokens, batch, out_path, anchors_path;
    bool have_text = false, have_tokens = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) usage_error("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "-m" || arg == "--model") options.model_path = value();
        else if (arg == "-d" || arg == "--device") options.device = value();
        else if (arg == "-t" || arg == "--threads") {
            const std::string v = value();
            size_t used = 0;
            int n = -1;
            try { n = std::stoi(v, &used); } catch (const std::exception &) { used = 0; }
            if (used == 0 || used != v.size() || n < 0) usage_error("--threads needs a non-negative integer, got " + v);
            options.n_threads = n;
        }
        else if (arg == "-a" || arg == "--audio") audio = value();
        else if (arg == "--text") { text = value(); have_text = true; }
        else if (arg == "--tokens") { tokens = value(); have_tokens = true; }
        else if (arg == "--batch") batch = value();
        else if (arg == "-o" || arg == "--out") out_path = value();
        else if (arg == "--anchors") anchors_path = value();
        else if (arg == "-h" || arg == "--help") { std::fputs(USAGE, stdout); return 0; }
        else usage_error("unknown argument " + arg);
    }
    if (options.model_path.empty()) usage_error("--model is required");
    if (batch.empty() == audio.empty()) usage_error("give exactly one of --audio and --batch");
    if (!audio.empty() && have_text == have_tokens) usage_error("give exactly one of --text and --tokens");

    try {
        fusealign::Aligner aligner(options);
        std::fprintf(stderr, "fusealign-cli: %s on %s\n", options.model_path.c_str(),
                     aligner.model().backend_name().c_str());

        if (!audio.empty()) {
            std::vector<fusealign::AsrWord> anchors;
            if (!anchors_path.empty()) {
                std::ifstream in(anchors_path);
                if (!in) throw std::runtime_error("cannot open " + anchors_path);
                anchors = parse_anchors(json::parse(in));
            }
            int sample_rate = 0;
            const std::vector<float> wav = load_mono(audio, sample_rate);
            const auto * anchor_ptr = anchors_path.empty() ? nullptr : &anchors;
            const auto words = have_text ? aligner.align(wav, sample_rate, text, anchor_ptr)
                                         : aligner.align(wav, sample_rate,
                                                         json::parse(tokens).get<std::vector<std::string>>(), anchor_ptr);
            std::cout << words_json(words).dump(2) << "\n";
            return 0;
        }

        std::map<std::string, std::vector<fusealign::AsrWord>> anchors_by_clip;
        if (!anchors_path.empty()) {
            std::ifstream in(anchors_path);
            if (!in) throw std::runtime_error("cannot open " + anchors_path);
            for (std::string line; std::getline(in, line);) {
                if (line.empty()) continue;
                const json row = json::parse(line);
                anchors_by_clip[clip_key(row.at("clip_id"))] = parse_anchors(row.at("words"));
            }
        }
        std::ifstream in(batch);
        if (!in) throw std::runtime_error("cannot open " + batch);
        std::ofstream file;
        if (!out_path.empty()) {
            file.open(out_path);
            if (!file) throw std::runtime_error("cannot write " + out_path);
        }
        std::ostream & out = out_path.empty() ? std::cout : file;
        size_t rows = 0, failed = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::string line; std::getline(in, line);) {
            if (line.empty()) continue;
            json result = {{"clip_id", nullptr}};
            try {
                const json row = json::parse(line);
                result["clip_id"] = row.value("clip_id", json());
                std::vector<fusealign::AsrWord> row_anchors;
                const std::vector<fusealign::AsrWord> * anchor_ptr = nullptr;
                if (row.contains("anchors")) {
                    row_anchors = parse_anchors(row["anchors"]);
                    anchor_ptr = &row_anchors;
                } else if (row.contains("clip_id")) {
                    const auto it = anchors_by_clip.find(clip_key(row["clip_id"]));
                    if (it != anchors_by_clip.end()) anchor_ptr = &it->second;
                }
                result.update(align_row(aligner, row, anchor_ptr));
            } catch (const std::exception & e) {
                result["error"] = e.what();
                ++failed;
                std::fprintf(stderr, "fusealign-cli: %s: %s\n", result["clip_id"].dump().c_str(), e.what());
            }
            out << result.dump() << "\n";
            ++rows;
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "fusealign-cli: %zu rows (%zu failed) in %.1f s\n", rows, failed, seconds);
        return failed ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fusealign-cli: %s\n", e.what());
        return 1;
    }
}
