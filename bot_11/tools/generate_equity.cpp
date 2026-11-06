// Original offline Hold'em equity-label generator for bot_11.
// Build: g++ -O3 -std=c++17 -fopenmp tools/generate_equity.cpp -o generate_equity
// No external libraries, network access, poker code, or runtime bot dependency.
// Postflop output records are exactly 48 bytes, little endian:
// uint8 boardlen; uint8 cards[7] (hole2, board up to5, unused255); float32 y[10].
// y: random HU, random2, random3, pf40, pf15, pf5, current35, current15,
//    pf40 & current35, pf15 & current15.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
constexpr const char* ranks = "23456789TJQKA";
constexpr const char* suits = "cdhs";
std::array<uint32_t, 8192> top5{};
std::array<uint8_t, 8192> straight_high{};
std::array<uint16_t, 52> rank_bit{};

void init_evaluator() {
    for (int c = 0; c < 52; ++c) rank_bit[c] = uint16_t(1u << (c / 4));
    for (unsigned m = 1; m < 8192; ++m) {
        unsigned rest = m;
        uint32_t packed = 0;
        for (int shift = 16; rest && shift >= 0; shift -= 4) {
            int r = 31 - __builtin_clz(rest);
            packed |= unsigned(r + 2) << shift;
            rest ^= 1u << r;
        }
        top5[m] = packed;
        if ((m & 0x100f) == 0x100f) straight_high[m] = 5;
        for (int high = 4; high <= 12; ++high)
            if (((m >> (high - 4)) & 31) == 31) straight_high[m] = uint8_t(high + 2);
    }
}

inline int high_rank(unsigned mask) { return 32 - __builtin_clz(mask) + 1; }

inline uint32_t evaluate(const int* cards, int n) {
    unsigned once = 0, twice = 0, trips = 0, quads = 0;
    unsigned sm[4] = {0, 0, 0, 0};
    for (int i = 0; i < n; ++i) {
        const int c = cards[i];
        const unsigned bit = rank_bit[c];
        quads |= trips & bit;
        trips |= twice & bit;
        twice |= once & bit;
        once |= bit;
        sm[c & 3] |= bit;
    }
    unsigned flush = 0;
    for (int s = 0; s < 4; ++s)
        if (__builtin_popcount(sm[s]) >= 5) { flush = sm[s]; break; }
    if (flush && straight_high[flush]) return (8u << 20) | (uint32_t(straight_high[flush]) << 16);
    if (quads) {
        int q = high_rank(quads);
        return (7u << 20) | (q << 16) | (high_rank(once ^ (1u << (q - 2))) << 12);
    }
    int trip = 0;
    if (trips) {
        trip = high_rank(trips);
        unsigned pair = twice ^ (1u << (trip - 2));
        if (pair) return (6u << 20) | (trip << 16) | (high_rank(pair) << 12);
    }
    if (flush) return (5u << 20) | top5[flush];
    if (straight_high[once]) return (4u << 20) | (uint32_t(straight_high[once]) << 16);
    if (trips) return (3u << 20) | (trip << 16) | ((top5[once ^ (1u << (trip - 2))] >> 4) & 0xff00);
    if (twice) {
        int pair = high_rank(twice);
        unsigned rest = twice ^ (1u << (pair - 2));
        if (rest) {
            int second = high_rank(rest);
            int kicker = high_rank(once ^ (1u << (pair - 2)) ^ (1u << (second - 2)));
            return (2u << 20) | (pair << 16) | (second << 12) | (kicker << 8);
        }
        return (1u << 20) | (pair << 16) | ((top5[once ^ (1u << (pair - 2))] >> 4) & 0xfff0);
    }
    return top5[once];
}

// Independent readable reference for --mode check.
uint32_t reference5(const int* cards) {
    std::array<int, 15> counts{};
    std::vector<std::pair<int, int>> groups;
    bool flush = true;
    for (int i = 0; i < 5; ++i) {
        ++counts[cards[i] / 4 + 2];
        if (cards[i] % 4 != cards[0] % 4) flush = false;
    }
    for (int r = 14; r >= 2; --r) if (counts[r]) groups.emplace_back(counts[r], r);
    std::sort(groups.begin(), groups.end(), std::greater<std::pair<int, int>>());
    int straight = 0;
    if (groups.size() == 5) {
        int high = groups[0].second, low = groups.back().second;
        if (high - low == 4) straight = high;
        else if (counts[14] && counts[5] && counts[4] && counts[3] && counts[2]) straight = 5;
    }
    int category = 0;
    std::vector<int> kickers;
    if (flush && straight) { category = 8; kickers = {straight}; }
    else if (groups[0].first == 4) { category = 7; kickers = {groups[0].second, groups[1].second}; }
    else if (groups[0].first == 3 && groups[1].first == 2) { category = 6; kickers = {groups[0].second, groups[1].second}; }
    else if (flush) { category = 5; for (int r = 14; r >= 2; --r) if (counts[r]) kickers.push_back(r); }
    else if (straight) { category = 4; kickers = {straight}; }
    else {
        if (groups[0].first == 3) category = 3;
        else if (groups[0].first == 2 && groups[1].first == 2) category = 2;
        else if (groups[0].first == 2) category = 1;
        for (auto g : groups) kickers.push_back(g.second);
    }
    uint32_t score = uint32_t(category) << 20;
    for (size_t i = 0; i < kickers.size(); ++i) score |= uint32_t(kickers[i]) << (16 - 4 * i);
    return score;
}

struct RNG {
    uint64_t state;
    explicit RNG(uint64_t seed): state(seed) {}
    uint64_t next() {
        uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }
    unsigned bounded(unsigned n) {
        // Multiplication with a 32-bit random variate, with exact rejection.
        uint64_t product = uint64_t(uint32_t(next())) * n;
        uint32_t low = uint32_t(product);
        if (low < n) {
            const uint32_t threshold = uint32_t(-n) % n;
            while (low < threshold) {
                product = uint64_t(uint32_t(next())) * n;
                low = uint32_t(product);
            }
        }
        return unsigned(product >> 32);
    }
};

uint64_t mix_seed(uint64_t seed, uint64_t index, uint64_t domain) {
    RNG rng(seed ^ (index * UINT64_C(0xd1342543de82ef95)) ^ domain);
    return rng.next();
}

int class_index(int a, int b) {
    int ra = a / 4, rb = b / 4;
    int high = std::max(ra, rb), low = std::min(ra, rb);
    if (high == low || a % 4 == b % 4) return high * 13 + low;
    return low * 13 + high;
}

std::array<int, 2> class_cards(int index) {
    int row = index / 13, col = index % 13;
    if (row >= col) return {row * 4, col * 4 + (row == col ? 1 : 0)};
    return {col * 4, row * 4 + 1};
}

std::string class_key(int index) {
    int row = index / 13, col = index % 13;
    std::string key;
    key += ranks[std::max(row, col)]; key += ranks[std::min(row, col)];
    if (row != col) key += row > col ? 's' : 'o';
    return key;
}

int class_combos(int index) {
    int row = index / 13, col = index % 13;
    return row == col ? 6 : row > col ? 4 : 12;
}

int make_deck(const int* known, int n, int* deck) {
    uint64_t occupied = 0;
    for (int i = 0; i < n; ++i) occupied |= UINT64_C(1) << known[i];
    int count = 0;
    for (int c = 0; c < 52; ++c) if (!(occupied & (UINT64_C(1) << c))) deck[count++] = c;
    return count;
}

// Partial Fisher-Yates, then restore all swaps so every rollout starts with the
// same deck. Draws within a rollout are disjoint; rollouts are independent.
void draw(RNG& rng, int* deck, int deck_n, int count, int* chosen) {
    int positions[11];
    for (int i = 0; i < count; ++i) {
        int j = i + int(rng.bounded(unsigned(deck_n - i)));
        positions[i] = j;
        std::swap(deck[i], deck[j]);
        chosen[i] = deck[i];
    }
    for (int i = count - 1; i >= 0; --i) std::swap(deck[i], deck[positions[i]]);
}

inline double share(uint32_t hero, const uint32_t* opponents, int count) {
    int ties = 1;
    for (int i = 0; i < count; ++i) {
        if (opponents[i] > hero) return 0.0;
        if (opponents[i] == hero) ++ties;
    }
    return 1.0 / ties;
}

struct Options {
    std::string mode = "postflop", out = "", cards = "";
    uint64_t seed = 1100401, ranking_seed = 1100400;
    int states = 10000, rollouts = 1024, ranking_rollouts = 15000, threads = 1;
    bool exhaustive = false;
};

Options parse_options(int argc, char** argv) {
    Options options;
    bool rollouts_given = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--exhaustive") { options.exhaustive = true; continue; }
        if (arg == "--help") {
            std::cout << "--mode postflop|preflop|evaluate|check --states N --rollouts M --seed S --out PATH --threads 1..8\n"
                      << "--ranking-rollouts N (default15000) --ranking-seed S (fixed default1100400); evaluate: --cards 'As Ks Qs Js Ts' or integer-ID lines on stdin\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("missing argument for " + arg);
        std::string value = argv[++i];
        if (arg == "--mode") options.mode = value;
        else if (arg == "--out") options.out = value;
        else if (arg == "--cards") options.cards = value;
        else if (arg == "--seed") options.seed = std::stoull(value);
        else if (arg == "--ranking-seed") options.ranking_seed = std::stoull(value);
        else if (arg == "--states") options.states = std::stoi(value);
        else if (arg == "--rollouts") { options.rollouts = std::stoi(value); rollouts_given = true; }
        else if (arg == "--ranking-rollouts") options.ranking_rollouts = std::stoi(value);
        else if (arg == "--threads") options.threads = std::stoi(value);
        else throw std::runtime_error("unknown option " + arg);
    }
    if (options.mode == "preflop" && !rollouts_given) options.rollouts = 200000;
    if (options.states < 1 || options.rollouts < 1 || options.ranking_rollouts < 1)
        throw std::runtime_error("state/rollout counts must be positive");
    if (options.threads < 1 || options.threads > 8) throw std::runtime_error("threads must be in [1,8]");
#ifndef _OPENMP
    if (options.threads != 1) throw std::runtime_error("rebuild with -fopenmp for multithreading");
#endif
    if ((options.mode == "postflop" || options.mode == "preflop") && options.out.empty())
        throw std::runtime_error("--out is required for generators");
    return options;
}

struct Ranking {
    std::array<double, 169> uniform{}, percentile{}, mass_before{};
    double weight(int cls, double fraction) const {
        // Fractional membership for the boundary class preserves requested
        // combo mass exactly (6 pair, 4 suited, 12 offsuit combinations).
        return std::max(0.0, std::min(1.0, (fraction * 1326.0 - mass_before[cls]) / class_combos(cls)));
    }
};

Ranking build_ranking(const Options& options) {
    Ranking result;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.threads)
#endif
    for (int cls = 0; cls < 169; ++cls) {
        RNG rng(mix_seed(options.ranking_seed, cls, UINT64_C(0x72616e6b696e6721)));
        auto hole = class_cards(cls);
        int deck[52], chosen[11], hero[7], opponent[7];
        int deck_n = make_deck(hole.data(), 2, deck);
        hero[0] = hole[0]; hero[1] = hole[1];
        double total = 0;
        for (int m = 0; m < options.ranking_rollouts; ++m) {
            draw(rng, deck, deck_n, 7, chosen);
            for (int i = 0; i < 5; ++i) hero[i + 2] = opponent[i + 2] = chosen[i];
            opponent[0] = chosen[5]; opponent[1] = chosen[6];
            const uint32_t h = evaluate(hero, 7), o = evaluate(opponent, 7);
            total += h > o ? 1.0 : h == o ? 0.5 : 0.0;
        }
        result.uniform[cls] = total / options.ranking_rollouts;
    }
    std::array<int, 169> order{};
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return result.uniform[a] == result.uniform[b] ? a < b : result.uniform[a] > result.uniform[b];
    });
    double cumulative = 0;
    for (int cls : order) {
        result.mass_before[cls] = cumulative;
        result.percentile[cls] = (cumulative + class_combos(cls) * 0.5) / 1326.0;
        cumulative += class_combos(cls);
    }
    std::cerr << "ranking ready (169 classes x " << options.ranking_rollouts << " independent rollouts)\n";
    return result;
}

struct Record {
    uint8_t boardlen = 0;
    std::array<uint8_t, 7> cards{};
    std::array<float, 10> labels{};
};

Record postflop_state(const Options& options, const Ranking& ranking, int state) {
    RNG rng(mix_seed(options.seed, state, UINT64_C(0x706f7374666c6f70)));
    Record record;
    record.boardlen = uint8_t(3 + state % 3); // exactly balanced street counts
    record.cards.fill(255);
    int all_deck[52]; std::iota(all_deck, all_deck + 52, 0);
    int selected[11]; draw(rng, all_deck, 52, record.boardlen + 2, selected);
    int known[7];
    for (int i = 0; i < record.boardlen + 2; ++i) known[i] = record.cards[i] = uint8_t(selected[i]);
    int deck[52], hero[7], opponent[7];
    const int known_n = record.boardlen + 2;
    const int deck_n = make_deck(known, known_n, deck), missing = 5 - record.boardlen;
    std::copy(known, known + known_n, hero);
    for (int i = 0; i < record.boardlen; ++i) opponent[i + 2] = known[i + 2];
    std::vector<uint32_t> current(options.rollouts);
    std::vector<uint8_t> opponent_class(options.rollouts);
    std::vector<float> outcome(options.rollouts);
    double totals[3] = {0, 0, 0};
    for (int m = 0; m < options.rollouts; ++m) {
        draw(rng, deck, deck_n, missing + 6, selected);
        for (int i = 0; i < missing; ++i) hero[known_n + i] = opponent[known_n + i] = selected[i];
        const uint32_t h = evaluate(hero, 7);
        uint32_t opponents[3];
        for (int k = 0; k < 3; ++k) {
            opponent[0] = selected[missing + 2 * k];
            opponent[1] = selected[missing + 2 * k + 1];
            opponents[k] = evaluate(opponent, 7);
            if (k == 0) {
                current[m] = evaluate(opponent, known_n);
                opponent_class[m] = uint8_t(class_index(opponent[0], opponent[1]));
            }
        }
        outcome[m] = float(share(h, opponents, 1));
        totals[0] += outcome[m];
        totals[1] += share(h, opponents, 2);
        totals[2] += share(h, opponents, 3);
    }
    for (int i = 0; i < 3; ++i) record.labels[i] = float(totals[i] / options.rollouts);
    auto sorted = current;
    std::sort(sorted.begin(), sorted.end());
    const uint32_t cutoff35 = sorted[std::min(options.rollouts - 1, int(options.rollouts * 0.65))];
    const uint32_t cutoff15 = sorted[std::min(options.rollouts - 1, int(options.rollouts * 0.85))];
    double numerator[7]{}, denominator[7]{};
    for (int m = 0; m < options.rollouts; ++m) {
        const int cls = opponent_class[m];
        const double pf40 = ranking.weight(cls, .40), pf15 = ranking.weight(cls, .15), pf5 = ranking.weight(cls, .05);
        const double current35 = current[m] >= cutoff35 ? 1.0 : 0.0;
        const double current15 = current[m] >= cutoff15 ? 1.0 : 0.0;
        const double weight[7] = {pf40, pf15, pf5, current35, current15, pf40 * current35, pf15 * current15};
        for (int j = 0; j < 7; ++j) { numerator[j] += weight[j] * outcome[m]; denominator[j] += weight[j]; }
    }
    for (int j = 0; j < 7; ++j)
        record.labels[j + 3] = float(denominator[j] > 0 ? numerator[j] / denominator[j] : record.labels[0]);
    return record;
}

void write_float_le(std::ofstream& output, float value) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "requires IEEE754 float32");
    uint32_t bits; std::memcpy(&bits, &value, 4);
    const char bytes[4] = {char(bits & 255), char((bits >> 8) & 255), char((bits >> 16) & 255), char(bits >> 24)};
    output.write(bytes, 4);
}

void generate_postflop(const Options& options, const Ranking& ranking) {
    const auto start = std::chrono::steady_clock::now();
    std::vector<Record> records(options.states);
    std::atomic<int> completed{0};
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) num_threads(options.threads)
#endif
    for (int state = 0; state < options.states; ++state) {
        records[state] = postflop_state(options, ranking, state);
        int done = ++completed;
        if (done % std::max(100, options.states / 20) == 0) {
#ifdef _OPENMP
#pragma omp critical(progress)
#endif
            std::cerr << "postflop " << done << "/" << options.states << "\n";
        }
    }
    std::ofstream output(options.out, std::ios::binary);
    if (!output) throw std::runtime_error("cannot open output " + options.out);
    for (const auto& record : records) {
        output.put(char(record.boardlen));
        output.write(reinterpret_cast<const char*>(record.cards.data()), 7);
        for (float value : record.labels) write_float_le(output, value);
    }
    output.close();
    if (!output) throw std::runtime_error("failed writing output " + options.out);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "{\"mode\":\"postflop\",\"states\":" << options.states << ",\"rollouts\":" << options.rollouts
              << ",\"seed\":" << options.seed << ",\"ranking_seed\":" << options.ranking_seed
              << ",\"ranking_rollouts\":" << options.ranking_rollouts << ",\"threads\":" << options.threads
              << ",\"record_bytes\":48,\"seconds\":" << seconds << "}\n";
}

struct PreflopResult {
    std::array<double, 7> equity{};
    double uniform2 = 0, uniform3 = 0;
};

PreflopResult preflop_class(const Options& options, const Ranking& ranking, int cls) {
    RNG rng(mix_seed(options.seed, cls, UINT64_C(0x707265666c6f7021)));
    const auto hole = class_cards(cls);
    int deck[52], hero[7], opponent[7], selected[11];
    const int deck_n = make_deck(hole.data(), 2, deck);
    hero[0] = hole[0]; hero[1] = hole[1];
    const double fractions[7] = {1, .60, .40, .25, .15, .08, .04};
    double numerator[7]{}, denominator[7]{}, multi2 = 0, multi3 = 0;
    for (int m = 0; m < options.rollouts; ++m) {
        draw(rng, deck, deck_n, 11, selected);
        for (int i = 0; i < 5; ++i) hero[i + 2] = opponent[i + 2] = selected[i];
        const uint32_t h = evaluate(hero, 7);
        uint32_t opponents[3];
        int opp_class = 0;
        for (int k = 0; k < 3; ++k) {
            opponent[0] = selected[5 + k * 2]; opponent[1] = selected[6 + k * 2];
            opponents[k] = evaluate(opponent, 7);
            if (k == 0) opp_class = class_index(opponent[0], opponent[1]);
        }
        const double result = share(h, opponents, 1);
        for (int j = 0; j < 7; ++j) {
            const double weight = j == 0 ? 1.0 : ranking.weight(opp_class, fractions[j]);
            numerator[j] += weight * result; denominator[j] += weight;
        }
        multi2 += share(h, opponents, 2); multi3 += share(h, opponents, 3);
    }
    PreflopResult result;
    for (int j = 0; j < 7; ++j) result.equity[j] = denominator[j] > 0 ? numerator[j] / denominator[j] : numerator[0] / options.rollouts;
    result.uniform2 = multi2 / options.rollouts; result.uniform3 = multi3 / options.rollouts;
    return result;
}

void generate_preflop(const Options& options, const Ranking& ranking) {
    const auto start = std::chrono::steady_clock::now();
    std::array<PreflopResult, 169> results;
    std::atomic<int> completed{0};
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(options.threads)
#endif
    for (int cls = 0; cls < 169; ++cls) {
        results[cls] = preflop_class(options, ranking, cls);
        int done = ++completed;
        if (done % 13 == 0) {
#ifdef _OPENMP
#pragma omp critical(progress)
#endif
            std::cerr << "preflop " << done << "/169\n";
        }
    }
    std::ofstream output(options.out);
    if (!output) throw std::runtime_error("cannot open output " + options.out);
    output << std::setprecision(10) << "{\n";
    for (int cls = 0; cls < 169; ++cls) {
        output << "  \"" << class_key(cls) << "\":{\"percentile\":" << ranking.percentile[cls] << ",\"equity\":[";
        for (int j = 0; j < 7; ++j) output << (j ? "," : "") << results[cls].equity[j];
        output << "],\"uniform2\":" << results[cls].uniform2 << ",\"uniform3\":" << results[cls].uniform3 << "}" << (cls < 168 ? "," : "") << "\n";
    }
    output << "}\n"; output.close();
    if (!output) throw std::runtime_error("failed writing output " + options.out);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "{\"mode\":\"preflop\",\"classes\":169,\"rollouts\":" << options.rollouts
              << ",\"ranking_rollouts\":" << options.ranking_rollouts << ",\"ranking_seed\":" << options.ranking_seed << ",\"seed\":" << options.seed
              << ",\"threads\":" << options.threads << ",\"seconds\":" << seconds << "}\n";
}

std::vector<int> parse_cards(std::string text) {
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream input(text);
    std::vector<int> cards;
    std::string token;
    uint64_t occupied = 0;
    while (input >> token) {
        int c = -1;
        if (token.size() == 2 && std::strchr(ranks, std::toupper(token[0])) && std::strchr(suits, std::tolower(token[1])))
            c = int(std::strchr(ranks, std::toupper(token[0])) - ranks) * 4 + int(std::strchr(suits, std::tolower(token[1])) - suits);
        else { size_t end; c = std::stoi(token, &end); if (end != token.size()) throw std::runtime_error("invalid card " + token); }
        if (c < 0 || c >= 52 || (occupied & (UINT64_C(1) << c))) throw std::runtime_error("out-of-range or duplicate card");
        occupied |= UINT64_C(1) << c; cards.push_back(c);
    }
    if (cards.size() < 5 || cards.size() > 7) throw std::runtime_error("evaluate needs 5..7 cards");
    return cards;
}

void check_evaluator(const Options& options) {
    RNG rng(options.seed);
    int deck[52]; std::iota(deck, deck + 52, 0);
    int chosen[11], subset[5];
    for (int state = 0; state < options.states; ++state) {
        draw(rng, deck, 52, 7, chosen);
        uint32_t reference = 0;
        for (int a = 0; a < 3; ++a) for (int b = a + 1; b < 4; ++b)
        for (int c = b + 1; c < 5; ++c) for (int d = c + 1; d < 6; ++d)
        for (int e = d + 1; e < 7; ++e) {
            subset[0] = chosen[a]; subset[1] = chosen[b]; subset[2] = chosen[c]; subset[3] = chosen[d]; subset[4] = chosen[e];
            reference = std::max(reference, reference5(subset));
        }
        if (evaluate(chosen, 7) != reference) throw std::runtime_error("random 7-card reference mismatch");
    }
    if (options.exhaustive) {
        uint64_t counts[9]{};
        for (int a = 0; a < 48; ++a) for (int b = a + 1; b < 49; ++b)
        for (int c = b + 1; c < 50; ++c) for (int d = c + 1; d < 51; ++d)
        for (int e = d + 1; e < 52; ++e) {
            subset[0] = a; subset[1] = b; subset[2] = c; subset[3] = d; subset[4] = e;
            const uint32_t fast = evaluate(subset, 5);
            if (fast != reference5(subset)) throw std::runtime_error("exhaustive 5-card reference mismatch");
            ++counts[fast >> 20];
        }
        const uint64_t expected[9] = {1302540,1098240,123552,54912,10200,5108,3744,624,40};
        for (int i = 0; i < 9; ++i) if (counts[i] != expected[i]) throw std::runtime_error("category count mismatch");
    }
    std::cout << "{\"random_reference_checked\":" << options.states << ",\"exhaustive_5card\":" << (options.exhaustive ? "true" : "false") << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        init_evaluator();
        if (options.mode == "evaluate") {
            if (!options.cards.empty()) {
                auto cards = parse_cards(options.cards);
                std::cout << evaluate(cards.data(), int(cards.size())) << "\n";
            } else {
                std::string line;
                while (std::getline(std::cin, line)) if (!line.empty()) {
                    auto cards = parse_cards(line);
                    std::cout << evaluate(cards.data(), int(cards.size())) << "\n";
                }
            }
        } else if (options.mode == "check") check_evaluator(options);
        else if (options.mode == "postflop" || options.mode == "preflop") {
            const Ranking ranking = build_ranking(options);
            if (options.mode == "postflop") generate_postflop(options, ranking);
            else generate_preflop(options, ranking);
        } else throw std::runtime_error("unknown mode " + options.mode);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
