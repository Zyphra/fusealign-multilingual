// test-model-forward: one Model::forward per clip, for parity checks against the Python package.
//
//   test-model-forward --model GGUF [--device D] [--threads N] [--limit N] --inputs JSONL --out JSONL
//
// Input rows: {"clip_id", "audio" (16 kHz mono WAV path), "words"}. Output rows:
// {"clip_id", "intervals": [[start, end] | null], "presence": [...]}. Clips over max_seconds, over max_text_bytes
// or under one patch are skipped (reported on stderr), as they take the long-form or fallback paths.

#include "model.h"
#include "wav.h"

#include <json.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

std::vector<float> read_wav_16k_mono(const std::string & path) {
    fusealign::Audio audio = fusealign::read_wav(path);
    if (audio.channels != 1 || audio.sample_rate != 16000) throw std::runtime_error(path + ": expected 16 kHz mono");
    return std::move(audio.samples);
}

[[noreturn]] void usage() {
    std::cerr << "usage: test-model-forward --model GGUF [--device auto|cpu|cuda|cuda:N] [--threads N] [--limit N] "
                 "--inputs JSONL --out JSONL\n";
    std::exit(2);
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model_path, inputs_path, out_path;
    fusealign::BackendOptions options;
    long limit = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) usage();
        if (arg == "--model") model_path = argv[++i];
        else if (arg == "--device") options.device = argv[++i];
        else if (arg == "--threads") options.n_threads = std::stoi(argv[++i]);
        else if (arg == "--limit") limit = std::stol(argv[++i]);
        else if (arg == "--inputs") inputs_path = argv[++i];
        else if (arg == "--out") out_path = argv[++i];
        else usage();
    }
    if (model_path.empty() || inputs_path.empty() || out_path.empty()) usage();

    try {
        fusealign::Model model(model_path, options);
        const fusealign::HParams & hp = model.hparams();
        std::cerr << "model " << model_path << " on " << model.backend_name() << "\n";
        std::ifstream inputs(inputs_path);
        if (!inputs) throw std::runtime_error("cannot open " + inputs_path);
        std::ofstream out(out_path);
        if (!out) throw std::runtime_error("cannot write " + out_path);

        long rows = 0, done = 0, skipped = 0;
        double forward_seconds = 0.0;
        std::string line;
        while ((limit < 0 || rows < limit) && std::getline(inputs, line)) {
            if (line.empty()) continue;
            ++rows;
            const json row = json::parse(line);
            const std::string clip_id = row.at("clip_id").get<std::string>();
            const std::vector<std::string> words = row.at("words").get<std::vector<std::string>>();
            const std::vector<float> wav = read_wav_16k_mono(row.at("audio").get<std::string>());
            size_t text_bytes = words.empty() ? 0 : words.size() - 1;
            for (const std::string & w : words) text_bytes += w.size();
            const char * reason = nullptr;
            if (wav.size() < (size_t) hp.patch_window) reason = "under one patch";
            else if (wav.size() > (size_t) (hp.max_seconds * 16000)) reason = "over max_seconds";
            else if (text_bytes > (size_t) hp.max_text_bytes) reason = "over max_text_bytes";
            else if (words.empty()) reason = "no words";
            if (reason) {
                std::cerr << "skipped " << clip_id << ": " << reason << "\n";
                ++skipped;
                continue;
            }
            const auto t0 = std::chrono::steady_clock::now();
            const fusealign::Placement placement = model.forward(wav.data(), wav.size(), words);
            forward_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            json intervals = json::array();
            for (const auto & interval : placement.intervals) {
                intervals.push_back(interval ? json::array({interval->first, interval->second}) : json(nullptr));
            }
            out << json{{"clip_id", clip_id}, {"intervals", intervals}, {"presence", placement.presence}}.dump()
                << "\n";
            ++done;
        }
        std::fprintf(stderr, "%ld clips aligned, %ld skipped; forward %.3f s (%.2f clips/s)\n", done, skipped,
                     forward_seconds, done / (forward_seconds > 0 ? forward_seconds : 1.0));
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
