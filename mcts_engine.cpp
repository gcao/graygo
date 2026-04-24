/*
 * mcts_engine.cpp — Full C++ self-play engine for Gray Go v6.
 *
 * Includes:
 *   - State encoding (encode_player_relative) — 6-channel (9,9) tensor
 *   - MCTS with libtorch inference (no Python callbacks)
 *   - Self-play game runner returning training samples as numpy arrays
 *
 * Build (see Makefile target mcts_engine):
 *   g++ -O3 -std=c++17 -fPIC -shared \
 *       $(python3 -m pybind11 --includes) \
 *       -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include \
 *       mcts_engine.cpp \
 *       -L$(TORCH_DIR)/lib -ltorch -ltorch_cpu -lc10 -ltorch_cuda \
 *       -Wl,-rpath,$(TORCH_DIR)/lib \
 *       -o mcts_engine$(python3-config --extension-suffix)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <torch/script.h>
#include <torch/torch.h>

#include <vector>
#include <set>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <memory>
#include <deque>
#include <utility>
#include <limits>

namespace py = pybind11;

// ──────────────────────────────────────────────────────────────
// Constants (must match engine.cpp)
// ──────────────────────────────────────────────────────────────

static constexpr int EMPTY = 0;
static constexpr int BLACK = 1;
static constexpr int WHITE = 2;
static constexpr int GRAY  = 3;

static constexpr int BLACK_PLAYER = 0;
static constexpr int WHITE_PLAYER = 1;

static constexpr int MAX_SIZE = 19;

// ──────────────────────────────────────────────────────────────
// Board (copied from engine.cpp — self-contained)
// ──────────────────────────────────────────────────────────────

class Board {
public:
    int size;
    int8_t grid[MAX_SIZE * MAX_SIZE];

    Board(int sz = 9) : size(sz) {
        std::memset(grid, EMPTY, sizeof(grid));
    }

    Board(const Board& o) : size(o.size) {
        std::memcpy(grid, o.grid, sizeof(grid));
    }

    Board& operator=(const Board& o) {
        size = o.size;
        std::memcpy(grid, o.grid, sizeof(grid));
        return *this;
    }

    int get(int x, int y) const {
        int s = size;
        return grid[((y % s + s) % s) * s + ((x % s + s) % s)];
    }

    void set(int x, int y, int value) {
        int s = size;
        grid[((y % s + s) % s) * s + ((x % s + s) % s)] = (int8_t)value;
    }

    int get_action(int action) const {
        return grid[(action / size) * size + (action % size)];
    }

    void set_action(int action, int value) {
        grid[(action / size) * size + (action % size)] = (int8_t)value;
    }

    void neighbors(int x, int y, int out_x[4], int out_y[4]) const {
        int s = size;
        out_x[0] = (x - 1 + s) % s; out_y[0] = y;
        out_x[1] = (x + 1) % s;     out_y[1] = y;
        out_x[2] = x;               out_y[2] = (y - 1 + s) % s;
        out_x[3] = x;               out_y[3] = (y + 1) % s;
    }

    void group_and_liberties(
        int start_action,
        std::vector<bool>& visited,
        std::vector<int>& group,
        std::set<int>& liberties
    ) const {
        int color = get_action(start_action);
        if (color == EMPTY) return;

        group.clear();
        liberties.clear();

        std::vector<int> stack;
        stack.push_back(start_action);

        while (!stack.empty()) {
            int action = stack.back();
            stack.pop_back();

            if (visited[action]) continue;
            if (get_action(action) != color) continue;

            visited[action] = true;
            group.push_back(action);

            int ax = action % size;
            int ay = action / size;
            int nx[4], ny[4];
            neighbors(ax, ay, nx, ny);

            for (int i = 0; i < 4; i++) {
                int na = ny[i] * size + nx[i];
                int nc = grid[na];
                if (nc == EMPTY) {
                    liberties.insert(na);
                } else if (nc == color && !visited[na]) {
                    stack.push_back(na);
                }
            }
        }
    }

    struct GroupInfo {
        std::vector<int> stones;
        int color;
    };

    std::vector<GroupInfo> dead_groups() const {
        int board_points = size * size;
        std::vector<GroupInfo> dead;
        std::vector<bool> visited(board_points, false);
        std::vector<int> group;
        std::set<int> liberties;

        for (int action = 0; action < board_points; action++) {
            int color = get_action(action);
            if (color == EMPTY || visited[action]) continue;

            group.clear();
            liberties.clear();
            const_cast<Board*>(this)->group_and_liberties(action, visited, group, liberties);

            if (!group.empty() && liberties.empty()) {
                dead.push_back({group, color});
            }
        }
        return dead;
    }

    struct RemovedCounts {
        int black = 0;
        int white = 0;
        int gray = 0;
    };

    RemovedCounts remove_groups(const std::vector<GroupInfo>& groups) {
        RemovedCounts removed;
        for (auto& g : groups) {
            if (g.stones.empty()) continue;
            int color = g.color;
            int count = (int)g.stones.size();
            if (color == BLACK) removed.black += count;
            else if (color == WHITE) removed.white += count;
            else if (color == GRAY) removed.gray += count;
            for (int action : g.stones) {
                set_action(action, EMPTY);
            }
        }
        return removed;
    }

    std::pair<double, double> score() const {
        double black_score = 0.0;
        double white_score = 0.0;
        int board_points = size * size;

        for (int action = 0; action < board_points; action++) {
            int color = get_action(action);
            if (color == BLACK) black_score += 1.0;
            else if (color == WHITE) white_score += 1.0;
            else if (color == GRAY) {
                black_score += 0.5;
                white_score += 0.5;
            }
        }

        std::vector<bool> visited(board_points, false);
        for (int action = 0; action < board_points; action++) {
            if (visited[action] || get_action(action) != EMPTY) continue;

            std::vector<int> region;
            std::set<int> borders;
            std::vector<int> stack;
            stack.push_back(action);

            while (!stack.empty()) {
                int cur = stack.back();
                stack.pop_back();
                if (visited[cur]) continue;
                if (get_action(cur) != EMPTY) continue;

                visited[cur] = true;
                region.push_back(cur);

                int cx = cur % size;
                int cy = cur / size;
                int nx[4], ny[4];
                neighbors(cx, cy, nx, ny);

                for (int i = 0; i < 4; i++) {
                    int na = ny[i] * size + nx[i];
                    int nc = grid[na];
                    if (nc == EMPTY && !visited[na]) {
                        stack.push_back(na);
                    } else if (nc != EMPTY) {
                        borders.insert(nc);
                    }
                }
            }

            int n_factions = (int)borders.size();
            if (n_factions == 0) {
                black_score += (double)region.size() * 0.5;
                white_score += (double)region.size() * 0.5;
            } else {
                double weight = 1.0 / n_factions;
                double w_black = (borders.count(BLACK) ? weight : 0.0);
                double w_white = (borders.count(WHITE) ? weight : 0.0);
                double w_gray  = (borders.count(GRAY)  ? weight : 0.0);
                double bs = w_black + 0.5 * w_gray;
                double ws = w_white + 0.5 * w_gray;
                black_score += (double)region.size() * bs;
                white_score += (double)region.size() * ws;
            }
        }

        return {black_score, white_score};
    }

    bool equals(const Board& other) const {
        if (size != other.size) return false;
        return std::memcmp(grid, other.grid, size * size) == 0;
    }
};

// ──────────────────────────────────────────────────────────────
// Ko History Entry
// ──────────────────────────────────────────────────────────────

struct KoEntry {
    Board board;
    int black_action;
    int white_action;
};

// ──────────────────────────────────────────────────────────────
// GameState (copied from engine.cpp)
// ──────────────────────────────────────────────────────────────

class GameState {
public:
    int size;
    Board board;
    int turn_number;
    std::set<int> forbidden_black;
    std::set<int> forbidden_white;
    int consecutive_double_passes;
    bool game_over;
    std::vector<KoEntry> ko_history;

    GameState(int sz = 9)
        : size(sz), board(sz), turn_number(0),
          consecutive_double_passes(0), game_over(false) {}

    GameState copy() const {
        GameState result(size);
        result.board = board;
        result.turn_number = turn_number;
        result.forbidden_black = forbidden_black;
        result.forbidden_white = forbidden_white;
        result.consecutive_double_passes = consecutive_double_passes;
        result.game_over = game_over;
        result.ko_history.reserve(ko_history.size());
        for (auto& entry : ko_history) {
            result.ko_history.push_back({Board(entry.board), entry.black_action, entry.white_action});
        }
        return result;
    }

    int pass_action() const { return size * size; }
    bool is_pass(int action) const { return action == size * size; }

    void get_legal_actions(int player, std::vector<uint8_t>& out) const {
        int total = size * size + 1;
        out.resize(total);
        const auto& forbidden = (player == BLACK_PLAYER) ? forbidden_black : forbidden_white;
        for (int i = 0; i < size * size; i++) {
            out[i] = (board.get_action(i) == EMPTY && forbidden.count(i) == 0) ? 1 : 0;
        }
        out[size * size] = 1;  // pass always legal
    }

    bool is_legal_action(int player, int action) const {
        if (is_pass(action)) return true;
        if (action < 0 || action >= size * size) return false;
        if (board.get_action(action) != EMPTY) return false;
        const auto& forbidden = (player == BLACK_PLAYER) ? forbidden_black : forbidden_white;
        return forbidden.count(action) == 0;
    }

    void step(int black_action, int white_action) {
        if (game_over) return;

        bool both_pass = is_pass(black_action) && is_pass(white_action);
        if (both_pass) {
            consecutive_double_passes++;
        } else {
            consecutive_double_passes = 0;
        }

        Board board_before = board;
        std::set<int> newly_placed;

        if (!both_pass) {
            bool black_is_pass = is_pass(black_action);
            bool white_is_pass = is_pass(white_action);

            if (!black_is_pass && !white_is_pass && black_action == white_action) {
                int x = black_action % size;
                int y = black_action / size;
                board.set(x, y, GRAY);
                newly_placed.insert(black_action);
            } else {
                if (!black_is_pass) {
                    int x = black_action % size;
                    int y = black_action / size;
                    board.set(x, y, BLACK);
                    newly_placed.insert(black_action);
                }
                if (!white_is_pass) {
                    int x = white_action % size;
                    int y = white_action / size;
                    board.set(x, y, WHITE);
                    newly_placed.insert(white_action);
                }
            }

            auto dead = board.dead_groups();
            std::vector<Board::GroupInfo> first_pass;
            for (auto& g : dead) {
                bool contains_new = false;
                for (int s : g.stones) {
                    if (newly_placed.count(s)) { contains_new = true; break; }
                }
                if (!contains_new) first_pass.push_back(g);
            }
            board.remove_groups(first_pass);

            dead = board.dead_groups();
            board.remove_groups(dead);
        }

        bool board_changed = !board.equals(board_before);

        if (board_changed) {
            forbidden_black.clear();
            forbidden_white.clear();

            for (auto& entry : ko_history) {
                if (board.equals(entry.board)) {
                    if (!is_pass(entry.black_action)) {
                        forbidden_black.insert(entry.black_action);
                    }
                    if (!is_pass(entry.white_action)) {
                        forbidden_white.insert(entry.white_action);
                    }
                    break;
                }
            }
        } else {
            if (!is_pass(black_action)) {
                forbidden_black.insert(black_action);
            }
            if (!is_pass(white_action)) {
                forbidden_white.insert(white_action);
            }
        }

        ko_history.push_back({board_before, black_action, white_action});
        if (ko_history.size() > 2) {
            ko_history.erase(ko_history.begin());
        }

        turn_number++;
        if (both_pass && consecutive_double_passes >= 2) {
            game_over = true;
        }
    }

    int winner_player() const {
        auto [bs, ws] = board.score();
        if (bs > ws) return BLACK_PLAYER;
        if (ws > bs) return WHITE_PLAYER;
        return -1;
    }
};

// ──────────────────────────────────────────────────────────────
// State Encoding (Step 2)
// ──────────────────────────────────────────────────────────────

// Encode board from player's perspective into a (6, S, S) float tensor.
// Channels: [my_stones, opp_stones, gray, empty, my_forbidden, opp_forbidden]
static torch::Tensor encode_player_relative(const GameState& state, int player) {
    int s = state.size;
    auto tensor = torch::zeros({6, s, s}, torch::kFloat32);
    auto acc = tensor.accessor<float, 3>();

    int my_color, opp_color;
    if (player == BLACK_PLAYER) {
        my_color = BLACK;
        opp_color = WHITE;
    } else {
        my_color = WHITE;
        opp_color = BLACK;
    }

    const auto& my_forbidden = (player == BLACK_PLAYER) ? state.forbidden_black : state.forbidden_white;
    const auto& opp_forbidden = (player == BLACK_PLAYER) ? state.forbidden_white : state.forbidden_black;

    for (int y = 0; y < s; y++) {
        for (int x = 0; x < s; x++) {
            int cell = state.board.grid[y * s + x];
            if (cell == my_color)  acc[0][y][x] = 1.0f;
            if (cell == opp_color) acc[1][y][x] = 1.0f;
            if (cell == GRAY)      acc[2][y][x] = 1.0f;
            if (cell == EMPTY)     acc[3][y][x] = 1.0f;
        }
    }

    for (int action : my_forbidden) {
        acc[4][action / s][action % s] = 1.0f;
    }
    for (int action : opp_forbidden) {
        acc[5][action / s][action % s] = 1.0f;
    }

    return tensor;
}

// ──────────────────────────────────────────────────────────────
// Model wrapper for libtorch inference
// ──────────────────────────────────────────────────────────────

class ModelWrapper {
    torch::jit::script::Module model_;
    torch::Device device_;
public:
    ModelWrapper(const std::string& path, torch::Device device)
        : device_(device)
    {
        model_ = torch::jit::load(path, device);
        model_.eval();
    }

    // Forward pass: input (N, 6, S, S) -> (policy_logits, value, aux1..aux5)
    // Returns softmaxed policy (N, action_size) and value (N,)
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input) {
        torch::NoGradGuard no_grad;
        input = input.to(device_);
        auto outputs = model_.forward({input}).toTuple();
        auto policy_logits = outputs->elements()[0].toTensor();  // (N, action_size)
        auto value = outputs->elements()[1].toTensor();           // (N,)
        auto policy = torch::softmax(policy_logits, /*dim=*/1);
        return {policy.cpu(), value.cpu()};
    }

    // Color-flip inference: encode from both perspectives, run two forward passes
    // Returns (bp, wp, value) where value is from Black's perspective
    struct ColorFlipResult {
        std::vector<float> bp;  // black policy (n_actions,)
        std::vector<float> wp;  // white policy (n_actions,)
        float value;            // from Black's perspective
    };

    ColorFlipResult evaluate_position(const GameState& state) {
        auto black_enc = encode_player_relative(state, BLACK_PLAYER).unsqueeze(0);
        auto white_enc = encode_player_relative(state, WHITE_PLAYER).unsqueeze(0);

        // Batch both encodings together for a single forward pass
        auto batch = torch::cat({black_enc, white_enc}, 0);  // (2, 6, S, S)
        auto [policies, values] = forward(batch);

        int n_actions = policies.size(1);
        ColorFlipResult result;
        result.bp.resize(n_actions);
        result.wp.resize(n_actions);

        auto p_acc = policies.accessor<float, 2>();
        for (int i = 0; i < n_actions; i++) {
            result.bp[i] = p_acc[0][i];
            result.wp[i] = p_acc[1][i];
        }

        float bv = values[0].item<float>();
        float wv = values[1].item<float>();
        result.value = (bv - wv) / 2.0f;

        return result;
    }

    // Batched color-flip evaluation for multiple states
    std::vector<ColorFlipResult> evaluate_positions_batch(const std::vector<const GameState*>& states) {
        if (states.empty()) return {};

        int n = (int)states.size();
        int s = states[0]->size;

        // Encode all states from both perspectives
        std::vector<torch::Tensor> encodings;
        encodings.reserve(2 * n);
        for (int i = 0; i < n; i++) {
            encodings.push_back(encode_player_relative(*states[i], BLACK_PLAYER));
            encodings.push_back(encode_player_relative(*states[i], WHITE_PLAYER));
        }
        auto batch = torch::stack(encodings, 0);  // (2*N, 6, S, S)
        auto [policies, values] = forward(batch);

        int n_actions = policies.size(1);
        auto p_acc = policies.accessor<float, 2>();
        auto v_acc = values.accessor<float, 1>();

        std::vector<ColorFlipResult> results(n);
        for (int i = 0; i < n; i++) {
            results[i].bp.resize(n_actions);
            results[i].wp.resize(n_actions);
            for (int j = 0; j < n_actions; j++) {
                results[i].bp[j] = p_acc[2 * i][j];
                results[i].wp[j] = p_acc[2 * i + 1][j];
            }
            float bv = v_acc[2 * i];
            float wv = v_acc[2 * i + 1];
            results[i].value = (bv - wv) / 2.0f;
        }
        return results;
    }
};

// ──────────────────────────────────────────────────────────────
// MCTS (Step 3) — adapted from mcts_engine_callback.cpp
// ──────────────────────────────────────────────────────────────

struct JointAction {
    int black;
    int white;
};

struct Node {
    bool is_expanded = false;
    int visit_count = 0;
    double value_sum = 0.0;

    std::vector<float> policy_black;
    std::vector<float> policy_white;

    std::vector<JointAction> candidates;
    std::unordered_map<int, std::unique_ptr<Node>> children;
    std::vector<int> child_N;
    std::vector<double> child_W;

    GameState state;
    bool game_over = false;
    double terminal_value = 0.0;

    Node() = default;
};

static void normalize_policy(float* policy, const uint8_t* legal, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        if (!legal[i]) policy[i] = 0.0f;
        total += policy[i];
    }
    if (total <= 0.0) {
        std::memset(policy, 0, n * sizeof(float));
        policy[n - 1] = 1.0f;
    } else {
        float inv = 1.0f / (float)total;
        for (int i = 0; i < n; i++) policy[i] *= inv;
    }
}

static void build_candidates(
    const float* bp, const float* wp,
    const uint8_t* legal_b, const uint8_t* legal_w,
    int n_actions, float tau, int max_per_player,
    std::vector<JointAction>& out
) {
    std::vector<int> b_idx, w_idx;
    for (int i = 0; i < n_actions; i++) {
        if (bp[i] > tau && legal_b[i]) b_idx.push_back(i);
    }
    for (int i = 0; i < n_actions; i++) {
        if (wp[i] > tau && legal_w[i]) w_idx.push_back(i);
    }

    if (b_idx.empty()) {
        int best = -1; float bestv = -1.0f;
        for (int i = 0; i < n_actions; i++)
            if (legal_b[i] && bp[i] > bestv) { bestv = bp[i]; best = i; }
        if (best >= 0) b_idx.push_back(best);
        else b_idx.push_back(n_actions - 1);
    }
    if (w_idx.empty()) {
        int best = -1; float bestv = -1.0f;
        for (int i = 0; i < n_actions; i++)
            if (legal_w[i] && wp[i] > bestv) { bestv = wp[i]; best = i; }
        if (best >= 0) w_idx.push_back(best);
        else w_idx.push_back(n_actions - 1);
    }

    auto topk = [](std::vector<int>& idx, const float* p, int k) {
        if ((int)idx.size() <= k) return;
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
            [p](int a, int b) { return p[a] > p[b]; });
        idx.resize(k);
    };
    topk(b_idx, bp, max_per_player);
    topk(w_idx, wp, max_per_player);

    out.clear();
    out.reserve(b_idx.size() * w_idx.size());
    for (int b : b_idx)
        for (int w : w_idx)
            out.push_back({b, w});
}

struct LocalMatrixSolution {
    std::vector<int> black_actions;
    std::vector<int> white_actions;
    std::vector<double> black_prior;
    std::vector<double> white_prior;
    std::vector<double> avg_black;
    std::vector<double> avg_white;
    std::vector<int> black_visits;
    std::vector<int> white_visits;
    std::vector<std::vector<double>> q_matrix;
};

static void normalize_probs(std::vector<double>& probs) {
    double sum = 0.0;
    for (double v : probs) sum += v;
    if (sum <= 1e-12) {
        if (probs.empty()) return;
        double uniform = 1.0 / (double)probs.size();
        for (double& v : probs) v = uniform;
        return;
    }
    for (double& v : probs) v /= sum;
}

static std::vector<double> positive_regret_strategy(
    const std::vector<double>& regrets,
    const std::vector<double>& fallback_prior
) {
    std::vector<double> out(regrets.size(), 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < regrets.size(); i++) {
        out[i] = std::max(0.0, regrets[i]);
        sum += out[i];
    }
    if (sum <= 1e-12) {
        out = fallback_prior;
        normalize_probs(out);
        return out;
    }
    for (double& v : out) v /= sum;
    return out;
}

static LocalMatrixSolution solve_local_matrix_game(const Node& node, int iterations = 24) {
    LocalMatrixSolution sol;
    if (node.candidates.empty()) return sol;

    std::unordered_map<int, int> b_map;
    std::unordered_map<int, int> w_map;

    for (const auto& action : node.candidates) {
        if (b_map.find(action.black) == b_map.end()) {
            int idx = (int)sol.black_actions.size();
            b_map[action.black] = idx;
            sol.black_actions.push_back(action.black);
        }
        if (w_map.find(action.white) == w_map.end()) {
            int idx = (int)sol.white_actions.size();
            w_map[action.white] = idx;
            sol.white_actions.push_back(action.white);
        }
    }

    int nb = (int)sol.black_actions.size();
    int nw = (int)sol.white_actions.size();
    sol.black_prior.resize(nb, 0.0);
    sol.white_prior.resize(nw, 0.0);
    sol.avg_black.resize(nb, 0.0);
    sol.avg_white.resize(nw, 0.0);
    sol.black_visits.resize(nb, 0);
    sol.white_visits.resize(nw, 0);
    sol.q_matrix.assign(nb, std::vector<double>(nw, 0.0));

    for (int i = 0; i < nb; i++) sol.black_prior[i] = node.policy_black[sol.black_actions[i]];
    for (int j = 0; j < nw; j++) sol.white_prior[j] = node.policy_white[sol.white_actions[j]];
    normalize_probs(sol.black_prior);
    normalize_probs(sol.white_prior);

    for (int idx = 0; idx < (int)node.candidates.size(); idx++) {
        const auto& action = node.candidates[idx];
        int bi = b_map[action.black];
        int wj = w_map[action.white];
        int n = node.child_N[idx];
        double q = (n > 0) ? (node.child_W[idx] / n) : 0.0;
        sol.q_matrix[bi][wj] = q;
        sol.black_visits[bi] += n;
        sol.white_visits[wj] += n;
    }

    std::vector<double> sigma_b = sol.black_prior;
    std::vector<double> sigma_w = sol.white_prior;
    std::vector<double> regret_b(nb, 0.0), regret_w(nw, 0.0);

    for (int iter = 0; iter < iterations; iter++) {
        std::vector<double> black_payoff(nb, 0.0);
        for (int i = 0; i < nb; i++) {
            for (int j = 0; j < nw; j++) black_payoff[i] += sigma_w[j] * sol.q_matrix[i][j];
        }
        double exp_black = 0.0;
        for (int i = 0; i < nb; i++) exp_black += sigma_b[i] * black_payoff[i];
        for (int i = 0; i < nb; i++) regret_b[i] += black_payoff[i] - exp_black;

        std::vector<double> white_payoff(nw, 0.0);
        for (int j = 0; j < nw; j++) {
            for (int i = 0; i < nb; i++) white_payoff[j] += sigma_b[i] * (-sol.q_matrix[i][j]);
        }
        double exp_white = 0.0;
        for (int j = 0; j < nw; j++) exp_white += sigma_w[j] * white_payoff[j];
        for (int j = 0; j < nw; j++) regret_w[j] += white_payoff[j] - exp_white;

        sigma_b = positive_regret_strategy(regret_b, sol.black_prior);
        sigma_w = positive_regret_strategy(regret_w, sol.white_prior);

        for (int i = 0; i < nb; i++) sol.avg_black[i] += sigma_b[i];
        for (int j = 0; j < nw; j++) sol.avg_white[j] += sigma_w[j];
    }

    normalize_probs(sol.avg_black);
    normalize_probs(sol.avg_white);
    return sol;
}

static int select_puct(const Node& node, float c_puct) {
    if (node.candidates.empty()) return 0;
    if (node.candidates.size() == 1) return 0;

    LocalMatrixSolution sol = solve_local_matrix_game(node, 24);
    double sqrt_total = std::sqrt((double)(node.visit_count + 1));

    int best_b_idx = 0;
    double best_b_score = -1e18;
    for (int i = 0; i < (int)sol.black_actions.size(); i++) {
        double q_against_white = 0.0;
        for (int j = 0; j < (int)sol.white_actions.size(); j++) {
            q_against_white += sol.avg_white[j] * sol.q_matrix[i][j];
        }
        double u = c_puct * sol.black_prior[i] * sqrt_total / (1.0 + sol.black_visits[i]);
        double score = q_against_white + u;
        if (score > best_b_score) {
            best_b_score = score;
            best_b_idx = i;
        }
    }

    int best_w_idx = 0;
    double best_w_score = -1e18;
    for (int j = 0; j < (int)sol.white_actions.size(); j++) {
        double q_against_black = 0.0;
        for (int i = 0; i < (int)sol.black_actions.size(); i++) {
            q_against_black += sol.avg_black[i] * sol.q_matrix[i][j];
        }
        double u = c_puct * sol.white_prior[j] * sqrt_total / (1.0 + sol.white_visits[j]);
        double score = (-q_against_black) + u;
        if (score > best_w_score) {
            best_w_score = score;
            best_w_idx = j;
        }
    }

    int best_black = sol.black_actions[best_b_idx];
    int best_white = sol.white_actions[best_w_idx];
    for (int idx = 0; idx < (int)node.candidates.size(); idx++) {
        const auto& action = node.candidates[idx];
        if (action.black == best_black && action.white == best_white) return idx;
    }

    double best_joint_score = -1e18;
    int best_joint_idx = 0;
    for (int idx = 0; idx < (int)node.candidates.size(); idx++) {
        const auto& action = node.candidates[idx];
        int n = node.child_N[idx];
        double q = (n > 0) ? (node.child_W[idx] / n) : 0.0;
        double p = node.policy_black[action.black] * node.policy_white[action.white];
        double score = q + c_puct * p * sqrt_total / (1.0 + n);
        if (score > best_joint_score) {
            best_joint_score = score;
            best_joint_idx = idx;
        }
    }
    return best_joint_idx;
}

static void add_dirichlet_noise(
    float* policy, const uint8_t* legal, int n_actions, int board_size,
    float alpha, float epsilon, std::mt19937& rng
) {
    if (epsilon <= 0.0f) return;
    int board_pts = board_size * board_size;

    std::vector<int> legal_pts;
    for (int i = 0; i < board_pts && i < n_actions; i++)
        if (legal[i]) legal_pts.push_back(i);
    if (legal_pts.empty()) return;

    std::gamma_distribution<float> gamma(alpha, 1.0f);
    std::vector<float> noise(legal_pts.size());
    float noise_sum = 0.0f;
    for (size_t i = 0; i < legal_pts.size(); i++) {
        noise[i] = gamma(rng);
        noise_sum += noise[i];
    }
    if (noise_sum > 0) {
        for (auto& v : noise) v /= noise_sum;
    }

    for (size_t i = 0; i < legal_pts.size(); i++) {
        int idx = legal_pts[i];
        policy[idx] = (1.0f - epsilon) * policy[idx] + epsilon * noise[i];
    }
    float total = 0.0f;
    for (int i = 0; i < n_actions; i++) total += policy[i];
    if (total > 0) for (int i = 0; i < n_actions; i++) policy[i] /= total;
}

// Expand a node using model evaluation (pure C++ — no Python callback)
// Returns the value from Black's perspective.
static float expand_node(Node* node, ModelWrapper& model) {
    if (node->state.game_over) {
        node->is_expanded = true;
        node->game_over = true;
        int winner = node->state.winner_player();
        if (winner == BLACK_PLAYER) node->terminal_value = 1.0;
        else if (winner == WHITE_PLAYER) node->terminal_value = -1.0;
        else node->terminal_value = 0.0;
        return (float)node->terminal_value;
    }

    auto result = model.evaluate_position(node->state);
    int n_actions = (int)result.bp.size();

    node->policy_black = std::move(result.bp);
    node->policy_white = std::move(result.wp);

    std::vector<uint8_t> legal_b, legal_w;
    node->state.get_legal_actions(BLACK_PLAYER, legal_b);
    node->state.get_legal_actions(WHITE_PLAYER, legal_w);

    normalize_policy(node->policy_black.data(), legal_b.data(), n_actions);
    normalize_policy(node->policy_white.data(), legal_w.data(), n_actions);

    node->is_expanded = true;
    return result.value;
}

struct MCTSResult {
    std::unordered_map<int64_t, int> visit_counts;  // encoded (b*N+w) -> count
    int total_visits;
    int board_size;

    // Decode key to joint action
    std::pair<int, int> decode_key(int64_t key) const {
        int n = board_size * board_size + 1;
        return {(int)(key / n), (int)(key % n)};
    }

    static int64_t encode_key(int b, int w, int n_actions) {
        return (int64_t)b * n_actions + w;
    }
};

static MCTSResult run_mcts(
    const GameState& initial_state,
    ModelWrapper& model,
    int num_visits,
    float c_puct,
    float tau,
    float dirichlet_alpha,
    float dirichlet_epsilon,
    int max_candidates,
    std::mt19937& rng
) {
    int board_size = initial_state.size;
    int n_actions = board_size * board_size + 1;

    auto root = std::make_unique<Node>();
    root->state = initial_state.copy();

    // Evaluate root
    float root_value = expand_node(root.get(), model);

    if (root->game_over) {
        MCTSResult result;
        result.total_visits = 0;
        result.board_size = board_size;
        return result;
    }

    std::vector<uint8_t> legal_b, legal_w;
    root->state.get_legal_actions(BLACK_PLAYER, legal_b);
    root->state.get_legal_actions(WHITE_PLAYER, legal_w);

    build_candidates(root->policy_black.data(), root->policy_white.data(),
                     legal_b.data(), legal_w.data(), n_actions, tau,
                     max_candidates, root->candidates);
    root->child_N.resize(root->candidates.size(), 0);
    root->child_W.resize(root->candidates.size(), 0.0);

    // Dirichlet noise at root
    add_dirichlet_noise(root->policy_black.data(), legal_b.data(),
                        n_actions, board_size, dirichlet_alpha,
                        dirichlet_epsilon, rng);
    add_dirichlet_noise(root->policy_white.data(), legal_w.data(),
                        n_actions, board_size, dirichlet_alpha,
                        dirichlet_epsilon, rng);

    // Rebuild candidates after noise
    build_candidates(root->policy_black.data(), root->policy_white.data(),
                     legal_b.data(), legal_w.data(), n_actions, tau,
                     max_candidates, root->candidates);
    root->child_N.resize(root->candidates.size(), 0);
    root->child_W.resize(root->candidates.size(), 0.0);

    // MCTS visits
    for (int v = 0; v < num_visits; v++) {
        Node* node = root.get();
        std::vector<std::pair<Node*, int>> path;

        while (node->is_expanded) {
            if (node->game_over || node->candidates.empty()) {
                double val = node->terminal_value;
                node->visit_count++;
                for (auto it = path.rbegin(); it != path.rend(); ++it) {
                    it->first->visit_count++;
                    it->first->child_N[it->second]++;
                    it->first->child_W[it->second] += val;
                }
                goto next_visit;
            }

            {
                int cidx = select_puct(*node, c_puct);
                path.push_back({node, cidx});

                if (node->children.find(cidx) == node->children.end()) {
                    auto child = std::make_unique<Node>();
                    child->state = node->state.copy();
                    const JointAction& action = node->candidates[cidx];
                    child->state.step(action.black, action.white);
                    node->children[cidx] = std::move(child);
                }
                node = node->children[cidx].get();
            }
        }

        // Leaf node — expand
        {
            float value = expand_node(node, model);

            if (!node->game_over) {
                std::vector<uint8_t> lb, lw;
                node->state.get_legal_actions(BLACK_PLAYER, lb);
                node->state.get_legal_actions(WHITE_PLAYER, lw);

                build_candidates(node->policy_black.data(), node->policy_white.data(),
                                 lb.data(), lw.data(), n_actions, tau,
                                 max_candidates, node->candidates);
                node->child_N.resize(node->candidates.size(), 0);
                node->child_W.resize(node->candidates.size(), 0.0);
            }

            // Backup
            node->visit_count++;
            for (auto it = path.rbegin(); it != path.rend(); ++it) {
                it->first->visit_count++;
                it->first->child_N[it->second]++;
                it->first->child_W[it->second] += value;
            }
        }
        next_visit:;
    }

    // Collect results
    MCTSResult result;
    result.board_size = board_size;
    result.total_visits = 0;
    for (int i = 0; i < (int)root->candidates.size(); i++) {
        int n = root->child_N[i];
        if (n > 0) {
            int64_t key = MCTSResult::encode_key(root->candidates[i].black,
                                                  root->candidates[i].white, n_actions);
            result.visit_counts[key] = n;
            result.total_visits += n;
        }
    }
    return result;
}

// ──────────────────────────────────────────────────────────────
// Policy / move sampling helpers
// ──────────────────────────────────────────────────────────────

static void marginalize_policy(const MCTSResult& mcts, int player, int n_actions,
                                std::vector<float>& out) {
    out.assign(n_actions, 0.0f);
    float total = 0.0f;
    for (auto& [key, count] : mcts.visit_counts) {
        auto [b, w] = mcts.decode_key(key);
        if (player == BLACK_PLAYER) out[b] += count;
        else out[w] += count;
        total += count;
    }
    if (total > 0) {
        for (int i = 0; i < n_actions; i++) out[i] /= total;
    }
}

static std::pair<int, int> sample_joint_move(const MCTSResult& mcts, float temperature,
                                              std::mt19937& rng) {
    if (mcts.visit_counts.empty()) {
        int n = mcts.board_size * mcts.board_size;
        return {n, n};  // both pass
    }

    std::vector<int64_t> keys;
    std::vector<double> counts;
    for (auto& [key, count] : mcts.visit_counts) {
        keys.push_back(key);
        counts.push_back((double)count);
    }

    int idx;
    if (temperature < 1e-6) {
        idx = (int)(std::max_element(counts.begin(), counts.end()) - counts.begin());
    } else {
        std::vector<double> log_probs(counts.size());
        double max_log = -1e30;
        for (size_t i = 0; i < counts.size(); i++) {
            log_probs[i] = std::log(counts[i] + 1e-30) / temperature;
            if (log_probs[i] > max_log) max_log = log_probs[i];
        }
        double sum = 0.0;
        for (size_t i = 0; i < log_probs.size(); i++) {
            log_probs[i] = std::exp(log_probs[i] - max_log);
            sum += log_probs[i];
        }
        for (auto& p : log_probs) p /= sum;

        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);
        double cumulative = 0.0;
        idx = (int)log_probs.size() - 1;
        for (size_t i = 0; i < log_probs.size(); i++) {
            cumulative += log_probs[i];
            if (r <= cumulative) { idx = (int)i; break; }
        }
    }

    return mcts.decode_key(keys[idx]);
}

// ──────────────────────────────────────────────────────────────
// Auxiliary target computation
// ──────────────────────────────────────────────────────────────

static void compute_aux1(const GameState& before, const GameState& after,
                          int player, int board_size, float* out) {
    int my_color = (player == BLACK_PLAYER) ? BLACK : WHITE;
    for (int y = 0; y < board_size; y++) {
        for (int x = 0; x < board_size; x++) {
            int idx = y * board_size + x;
            int b = before.board.grid[idx];
            int a = after.board.grid[idx];
            if (b != a) {
                if (a == my_color && b != my_color) out[idx] = 1.0f;
                else if (b == my_color && a != my_color) out[idx] = -1.0f;
                else if (a == GRAY) out[idx] = 0.5f;
                else out[idx] = 0.0f;
            } else {
                out[idx] = 0.0f;
            }
        }
    }
}

static float compute_aux2(const GameState& final_state, int player) {
    auto [bs, ws] = final_state.board.score();
    int total = final_state.size * final_state.size;
    if (player == BLACK_PLAYER) return (float)((bs - ws) / total);
    else return (float)((ws - bs) / total);
}

static void compute_aux3(int opponent_action, int board_size, float* out) {
    int n_actions = board_size * board_size + 1;
    std::memset(out, 0, n_actions * sizeof(float));
    if (opponent_action >= 0 && opponent_action < n_actions) {
        out[opponent_action] = 1.0f;
    }
}

static float compute_policy_entropy(const std::vector<float>& policy) {
    double entropy = 0.0;
    for (float p : policy) {
        if (p > 1e-10f) {
            entropy -= p * std::log(p);
        }
    }
    double max_entropy = std::log((double)policy.size());
    if (max_entropy > 0) return (float)(entropy / max_entropy);
    return 0.0f;
}

static float compute_aux4(const MCTSResult& mcts, int player, int board_size) {
    int n_actions = board_size * board_size + 1;
    std::vector<float> policy;
    marginalize_policy(mcts, player, n_actions, policy);
    return compute_policy_entropy(policy);
}

static void compute_aux5(const GameState& state, int player, int board_size, float* out) {
    int s = board_size;
    int my_color = (player == BLACK_PLAYER) ? BLACK : WHITE;
    int opp_color = (player == BLACK_PLAYER) ? WHITE : BLACK;

    std::vector<float> dist_my(s * s, std::numeric_limits<float>::infinity());
    std::vector<float> dist_opp(s * s, std::numeric_limits<float>::infinity());

    std::deque<int> q_my, q_opp;

    for (int y = 0; y < s; y++) {
        for (int x = 0; x < s; x++) {
            int idx = y * s + x;
            int cell = state.board.grid[idx];
            if (cell == my_color) {
                dist_my[idx] = 0.0f;
                q_my.push_back(idx);
            } else if (cell == opp_color) {
                dist_opp[idx] = 0.0f;
                q_opp.push_back(idx);
            }
        }
    }

    // BFS for my stones
    while (!q_my.empty()) {
        int cur = q_my.front();
        q_my.pop_front();
        int cy = cur / s, cx = cur % s;
        int dx[4] = {-1, 1, 0, 0};
        int dy[4] = {0, 0, -1, 1};
        for (int d = 0; d < 4; d++) {
            int ny = (cy + dy[d] + s) % s;
            int nx = (cx + dx[d] + s) % s;
            int ni = ny * s + nx;
            int cell = state.board.grid[ni];
            if (cell == EMPTY || cell == GRAY) {
                float new_dist = dist_my[cur] + 1.0f;
                if (new_dist < dist_my[ni]) {
                    dist_my[ni] = new_dist;
                    q_my.push_back(ni);
                }
            }
        }
    }

    // BFS for opponent stones
    while (!q_opp.empty()) {
        int cur = q_opp.front();
        q_opp.pop_front();
        int cy = cur / s, cx = cur % s;
        int dx[4] = {-1, 1, 0, 0};
        int dy[4] = {0, 0, -1, 1};
        for (int d = 0; d < 4; d++) {
            int ny = (cy + dy[d] + s) % s;
            int nx = (cx + dx[d] + s) % s;
            int ni = ny * s + nx;
            int cell = state.board.grid[ni];
            if (cell == EMPTY || cell == GRAY) {
                float new_dist = dist_opp[cur] + 1.0f;
                if (new_dist < dist_opp[ni]) {
                    dist_opp[ni] = new_dist;
                    q_opp.push_back(ni);
                }
            }
        }
    }

    // Compute influence
    for (int i = 0; i < s * s; i++) {
        int cell = state.board.grid[i];
        if (cell == my_color) { out[i] = 1.0f; continue; }
        if (cell == opp_color) { out[i] = -1.0f; continue; }
        if (cell == GRAY) { out[i] = 0.0f; continue; }
        bool finite_my = std::isfinite(dist_my[i]);
        bool finite_opp = std::isfinite(dist_opp[i]);
        if (finite_my && finite_opp) {
            float total = dist_my[i] + dist_opp[i];
            if (total > 1e-6f) {
                out[i] = (dist_opp[i] - dist_my[i]) / total;
            } else {
                out[i] = 0.0f;
            }
        } else if (finite_my) {
            out[i] = 1.0f;
        } else if (finite_opp) {
            out[i] = -1.0f;
        } else {
            out[i] = 0.0f;
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Self-play game runner (Step 4)
// ──────────────────────────────────────────────────────────────

struct SelfPlayConfig {
    int board_size = 9;
    int num_games = 300;
    int max_turns = 100;
    int num_visits = 400;
    float c_puct = 1.5f;
    float tau = 0.01f;
    float dirichlet_alpha = 0.15f;
    float dirichlet_epsilon = 0.30f;
    float temp_high = 1.0f;
    float temp_low = 0.3f;
    int temp_threshold = 15;
    int randomize_first_n = 4;
    int max_candidates = 20;
    int seed = 42;
};

struct GameSamples {
    // Per-sample arrays: each game produces 2*turns samples (black + white perspectives)
    std::vector<std::vector<float>> states;     // each (6*S*S,)
    std::vector<std::vector<float>> policies;   // each (S*S+1,)
    std::vector<float> values;
    std::vector<std::vector<float>> aux1;       // each (S*S,)
    std::vector<float> aux2;
    std::vector<std::vector<float>> aux3;       // each (S*S+1,)
    std::vector<float> aux4;
    std::vector<std::vector<float>> aux5;       // each (S*S,)
};

static GameSamples play_one_game(ModelWrapper& model, const SelfPlayConfig& cfg,
                                  std::mt19937& rng) {
    int s = cfg.board_size;
    int n_actions = s * s + 1;
    int board_pts = s * s;

    GameState state(s);
    GameSamples result;

    // Per-turn records
    struct TurnRecord {
        GameState state_before;
        std::vector<float> bp;  // black policy
        std::vector<float> wp;  // white policy
        GameState state_after;
        int b_move, w_move;
        float aux4_b, aux4_w;
    };
    std::vector<TurnRecord> records;

    int turn = 0;
    while (!state.game_over && turn < cfg.max_turns) {
        GameState state_before = state.copy();

        int b_move, w_move;
        std::vector<float> bp(n_actions, 0.0f), wp(n_actions, 0.0f);
        float aux4_b = 0.0f, aux4_w = 0.0f;

        if (turn < cfg.randomize_first_n) {
            // Random moves
            std::vector<uint8_t> legal_b, legal_w;
            state.get_legal_actions(BLACK_PLAYER, legal_b);
            state.get_legal_actions(WHITE_PLAYER, legal_w);

            std::vector<int> b_legal, w_legal;
            for (int i = 0; i < n_actions; i++) {
                if (legal_b[i]) b_legal.push_back(i);
                if (legal_w[i]) w_legal.push_back(i);
            }

            if (b_legal.empty()) b_move = n_actions - 1;
            else {
                std::uniform_int_distribution<int> dist(0, (int)b_legal.size() - 1);
                b_move = b_legal[dist(rng)];
                for (int i : b_legal) bp[i] = 1.0f / b_legal.size();
            }

            if (w_legal.empty()) w_move = n_actions - 1;
            else {
                std::uniform_int_distribution<int> dist(0, (int)w_legal.size() - 1);
                w_move = w_legal[dist(rng)];
                for (int i : w_legal) wp[i] = 1.0f / w_legal.size();
            }

            aux4_b = compute_policy_entropy(bp);
            aux4_w = compute_policy_entropy(wp);
        } else {
            // MCTS search
            MCTSResult mcts = run_mcts(
                state, model, cfg.num_visits, cfg.c_puct, cfg.tau,
                cfg.dirichlet_alpha, cfg.dirichlet_epsilon,
                cfg.max_candidates, rng
            );

            marginalize_policy(mcts, BLACK_PLAYER, n_actions, bp);
            marginalize_policy(mcts, WHITE_PLAYER, n_actions, wp);

            float temp = (turn < cfg.temp_threshold) ? cfg.temp_high : cfg.temp_low;
            auto [bm, wm] = sample_joint_move(mcts, temp, rng);
            b_move = bm;
            w_move = wm;

            aux4_b = compute_aux4(mcts, BLACK_PLAYER, s);
            aux4_w = compute_aux4(mcts, WHITE_PLAYER, s);
        }

        state.step(b_move, w_move);

        TurnRecord rec;
        rec.state_before = std::move(state_before);
        rec.bp = std::move(bp);
        rec.wp = std::move(wp);
        rec.state_after = state.copy();
        rec.b_move = b_move;
        rec.w_move = w_move;
        rec.aux4_b = aux4_b;
        rec.aux4_w = aux4_w;
        records.push_back(std::move(rec));

        turn++;
    }

    // Determine game outcome
    int winner = state.winner_player();
    float black_outcome = 0.0f;
    if (winner == BLACK_PLAYER) black_outcome = 1.0f;
    else if (winner == WHITE_PLAYER) black_outcome = -1.0f;

    float aux2_black = compute_aux2(state, BLACK_PLAYER);
    float aux2_white = compute_aux2(state, WHITE_PLAYER);

    // Generate training samples (both perspectives per turn)
    for (auto& rec : records) {
        // Aux targets
        std::vector<float> aux1_b(board_pts), aux1_w(board_pts);
        std::vector<float> aux3_b(n_actions), aux3_w(n_actions);
        std::vector<float> aux5_b(board_pts), aux5_w(board_pts);

        compute_aux1(rec.state_before, rec.state_after, BLACK_PLAYER, s, aux1_b.data());
        compute_aux1(rec.state_before, rec.state_after, WHITE_PLAYER, s, aux1_w.data());

        compute_aux3(rec.w_move, s, aux3_b.data());  // Black's view: opponent=White
        compute_aux3(rec.b_move, s, aux3_w.data());  // White's view: opponent=Black

        compute_aux5(rec.state_before, BLACK_PLAYER, s, aux5_b.data());
        compute_aux5(rec.state_before, WHITE_PLAYER, s, aux5_w.data());

        // Black perspective sample
        auto black_enc = encode_player_relative(rec.state_before, BLACK_PLAYER);
        std::vector<float> black_state(black_enc.data_ptr<float>(),
                                        black_enc.data_ptr<float>() + 6 * board_pts);
        result.states.push_back(std::move(black_state));
        result.policies.push_back(rec.bp);
        result.values.push_back(black_outcome);
        result.aux1.push_back(std::move(aux1_b));
        result.aux2.push_back(aux2_black);
        result.aux3.push_back(std::move(aux3_b));
        result.aux4.push_back(rec.aux4_b);
        result.aux5.push_back(std::move(aux5_b));

        // White perspective sample
        auto white_enc = encode_player_relative(rec.state_before, WHITE_PLAYER);
        std::vector<float> white_state(white_enc.data_ptr<float>(),
                                        white_enc.data_ptr<float>() + 6 * board_pts);
        result.states.push_back(std::move(white_state));
        result.policies.push_back(rec.wp);
        result.values.push_back(-black_outcome);
        result.aux1.push_back(std::move(aux1_w));
        result.aux2.push_back(aux2_white);
        result.aux3.push_back(std::move(aux3_w));
        result.aux4.push_back(rec.aux4_w);
        result.aux5.push_back(std::move(aux5_w));
    }

    return result;
}

// Main entry point: run multiple self-play games, return numpy arrays
static py::dict run_selfplay_games(
    const std::string& model_path,
    int num_games,
    int board_size,
    int max_turns,
    int num_visits,
    float c_puct,
    float tau,
    float dirichlet_alpha,
    float dirichlet_epsilon,
    float temp_high,
    float temp_low,
    int temp_threshold,
    int randomize_first_n,
    int max_candidates,
    int seed,
    const std::string& device_str
) {
    torch::Device device(device_str);
    ModelWrapper model(model_path, device);

    SelfPlayConfig cfg;
    cfg.board_size = board_size;
    cfg.num_games = num_games;
    cfg.max_turns = max_turns;
    cfg.num_visits = num_visits;
    cfg.c_puct = c_puct;
    cfg.tau = tau;
    cfg.dirichlet_alpha = dirichlet_alpha;
    cfg.dirichlet_epsilon = dirichlet_epsilon;
    cfg.temp_high = temp_high;
    cfg.temp_low = temp_low;
    cfg.temp_threshold = temp_threshold;
    cfg.randomize_first_n = randomize_first_n;
    cfg.max_candidates = max_candidates;

    std::mt19937 rng(seed);

    int s = board_size;
    int n_actions = s * s + 1;
    int board_pts = s * s;

    // Collect all samples across games
    std::vector<std::vector<float>> all_states;
    std::vector<std::vector<float>> all_policies;
    std::vector<float> all_values;
    std::vector<std::vector<float>> all_aux1;
    std::vector<float> all_aux2;
    std::vector<std::vector<float>> all_aux3;
    std::vector<float> all_aux4;
    std::vector<std::vector<float>> all_aux5;

    int wins_black = 0, wins_white = 0, draws = 0;

    // Release GIL for the entire self-play loop
    py::gil_scoped_release release;

    for (int g = 0; g < num_games; g++) {
        auto samples = play_one_game(model, cfg, rng);

        all_states.insert(all_states.end(),
                          std::make_move_iterator(samples.states.begin()),
                          std::make_move_iterator(samples.states.end()));
        all_policies.insert(all_policies.end(),
                            std::make_move_iterator(samples.policies.begin()),
                            std::make_move_iterator(samples.policies.end()));
        all_values.insert(all_values.end(), samples.values.begin(), samples.values.end());
        all_aux1.insert(all_aux1.end(),
                        std::make_move_iterator(samples.aux1.begin()),
                        std::make_move_iterator(samples.aux1.end()));
        all_aux2.insert(all_aux2.end(), samples.aux2.begin(), samples.aux2.end());
        all_aux3.insert(all_aux3.end(),
                        std::make_move_iterator(samples.aux3.begin()),
                        std::make_move_iterator(samples.aux3.end()));
        all_aux4.insert(all_aux4.end(), samples.aux4.begin(), samples.aux4.end());
        all_aux5.insert(all_aux5.end(),
                        std::make_move_iterator(samples.aux5.begin()),
                        std::make_move_iterator(samples.aux5.end()));

        // Count wins from first sample's value
        if (!samples.values.empty()) {
            float v = samples.values[0];  // black perspective
            if (v > 0) wins_black++;
            else if (v < 0) wins_white++;
            else draws++;
        }
    }

    // Re-acquire GIL to create numpy arrays
    py::gil_scoped_acquire acquire;

    int n_samples = (int)all_states.size();
    if (n_samples == 0) {
        py::dict result;
        result["n_samples"] = 0;
        return result;
    }

    // Convert to numpy arrays
    auto states_np = py::array_t<float>({n_samples, 6, s, s});
    auto policies_np = py::array_t<float>({n_samples, n_actions});
    auto values_np = py::array_t<float>(n_samples);
    auto aux1_np = py::array_t<float>({n_samples, s, s});
    auto aux2_np = py::array_t<float>(n_samples);
    auto aux3_np = py::array_t<float>({n_samples, s * s + 1});
    auto aux4_np = py::array_t<float>(n_samples);
    auto aux5_np = py::array_t<float>({n_samples, s, s});

    float* s_ptr = (float*)states_np.request().ptr;
    float* p_ptr = (float*)policies_np.request().ptr;
    float* v_ptr = (float*)values_np.request().ptr;
    float* a1_ptr = (float*)aux1_np.request().ptr;
    float* a2_ptr = (float*)aux2_np.request().ptr;
    float* a3_ptr = (float*)aux3_np.request().ptr;
    float* a4_ptr = (float*)aux4_np.request().ptr;
    float* a5_ptr = (float*)aux5_np.request().ptr;

    for (int i = 0; i < n_samples; i++) {
        std::memcpy(s_ptr + i * 6 * board_pts, all_states[i].data(), 6 * board_pts * sizeof(float));
        std::memcpy(p_ptr + i * n_actions, all_policies[i].data(), n_actions * sizeof(float));
        v_ptr[i] = all_values[i];
        std::memcpy(a1_ptr + i * board_pts, all_aux1[i].data(), board_pts * sizeof(float));
        a2_ptr[i] = all_aux2[i];
        std::memcpy(a3_ptr + i * (board_pts + 1), all_aux3[i].data(), (board_pts + 1) * sizeof(float));
        a4_ptr[i] = all_aux4[i];
        std::memcpy(a5_ptr + i * board_pts, all_aux5[i].data(), board_pts * sizeof(float));
    }

    py::dict result;
    result["states"] = states_np;
    result["policies"] = policies_np;
    result["values"] = values_np;
    result["aux1"] = aux1_np;
    result["aux2"] = aux2_np;
    result["aux3"] = aux3_np;
    result["aux4"] = aux4_np;
    result["aux5"] = aux5_np;
    result["n_samples"] = n_samples;
    result["wins_black"] = wins_black;
    result["wins_white"] = wins_white;
    result["draws"] = draws;

    return result;
}

// ──────────────────────────────────────────────────────────────
// Python module
// ──────────────────────────────────────────────────────────────

PYBIND11_MODULE(mcts_engine, m) {
    m.doc() = "C++ MCTS engine v6 with libtorch inference for Gray Go";

    m.def("run_selfplay_games", &run_selfplay_games,
          py::arg("model_path"),
          py::arg("num_games") = 300,
          py::arg("board_size") = 9,
          py::arg("max_turns") = 100,
          py::arg("num_visits") = 400,
          py::arg("c_puct") = 1.5f,
          py::arg("tau") = 0.01f,
          py::arg("dirichlet_alpha") = 0.15f,
          py::arg("dirichlet_epsilon") = 0.30f,
          py::arg("temp_high") = 1.0f,
          py::arg("temp_low") = 0.3f,
          py::arg("temp_threshold") = 15,
          py::arg("randomize_first_n") = 4,
          py::arg("max_candidates") = 20,
          py::arg("seed") = 42,
          py::arg("device") = "cuda",
          "Run self-play games with pure C++ MCTS and libtorch inference.\n"
          "\n"
          "Returns dict with numpy arrays:\n"
          "  states: (N, 6, S, S), policies: (N, S*S+1), values: (N,),\n"
          "  aux1: (N, S, S), aux2: (N,), aux3: (N, S*S+1), aux4: (N,), aux5: (N, S, S)\n"
          "  n_samples: int, wins_black: int, wins_white: int, draws: int"
    );
}
