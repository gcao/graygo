/*
 * mcts_engine_callback.cpp — C++ MCTS engine for Gray Go joint-action search.
 *
 * Uses a Python eval callback for color-flip inference while keeping the
 * joint-action tree traversal in C++.
 *
 * eval_fn(state) -> (bp, wp, value, legal_b, legal_w, game_over, terminal_value)
 *
 * The Python wrapper handles:
 *   1. Encode state from Black perspective -> model forward pass -> bp, bv
 *   2. Encode state from White perspective -> model forward pass -> wp, wv
 *   3. Combine: value = (bv - wv) / 2.0 (from Black's perspective)
 *   4. Return (bp, wp, value, legal_b, legal_w, game_over, terminal_value)
 *
 * Build:
 *   g++ -O3 -shared -std=c++17 -fPIC \
 *       $(python3 -m pybind11 --includes) \
 *       mcts_engine_callback.cpp -o mcts_engine_callback$(python3-config --extension-suffix)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace py = pybind11;

// ──────────────────────────────────────────────────────────────
// Node
// ──────────────────────────────────────────────────────────────

struct JointAction {
    int black;
    int white;
    bool operator==(const JointAction& o) const { return black == o.black && white == o.white; }
};

struct JointActionHash {
    size_t operator()(const JointAction& a) const {
        return std::hash<int>()(a.black) ^ (std::hash<int>()(a.white) << 16);
    }
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

    py::object py_state;

    bool game_over = false;
    double terminal_value = 0.0;

    Node() = default;
};

// ──────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────

static void normalize_policy(float* policy, const uint8_t* legal, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        if (!legal[i]) policy[i] = 0.0f;
        total += policy[i];
    }
    if (total <= 0.0) {
        std::memset(policy, 0, n * sizeof(float));
        policy[n - 1] = 1.0f;  // pass
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

    // Fallback
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

    // Cap to top-K
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

// ──────────────────────────────────────────────────────────────
// PUCT selection
// ──────────────────────────────────────────────────────────────

static int select_puct(const Node& node, float c_puct) {
    double sqrt_total = std::sqrt((double)node.visit_count);
    double best_score = -1e18;
    int best_idx = 0;

    const int nc = (int)node.candidates.size();
    for (int i = 0; i < nc; i++) {
        int n = node.child_N[i];
        double q = (n > 0) ? node.child_W[i] / n : 0.0;
        float p = node.policy_black[node.candidates[i].black]
                * node.policy_white[node.candidates[i].white];
        double score = q + c_puct * p * sqrt_total / (1.0 + n);
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }
    return best_idx;
}

// ──────────────────────────────────────────────────────────────
// Dirichlet noise
// ──────────────────────────────────────────────────────────────

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
    // Re-normalize
    float total = 0.0f;
    for (int i = 0; i < n_actions; i++) total += policy[i];
    if (total > 0) for (int i = 0; i < n_actions; i++) policy[i] /= total;
}

// ──────────────────────────────────────────────────────────────
// Main MCTS function
// ──────────────────────────────────────────────────────────────

static py::dict run_mcts_callback_cpp(
    py::object initial_state,
    py::function eval_fn,
    py::function copy_state_fn,
    py::function step_fn,
    int num_visits,
    float c_puct,
    float tau,
    float dirichlet_alpha,
    float dirichlet_epsilon,
    int board_size,
    int max_candidates,
    int seed
) {
    int n_actions = board_size * board_size + 1;
    std::mt19937 rng(seed);

    auto root = std::make_unique<Node>();
    root->py_state = initial_state;

    // Evaluate root
    py::tuple eval_result = eval_fn(initial_state);
    auto bp_arr = eval_result[0].cast<py::array_t<float>>();
    auto wp_arr = eval_result[1].cast<py::array_t<float>>();
    float root_value = eval_result[2].cast<float>();
    auto lb_arr = eval_result[3].cast<py::array_t<uint8_t>>();
    auto lw_arr = eval_result[4].cast<py::array_t<uint8_t>>();
    root->game_over = eval_result[5].cast<bool>();
    root->terminal_value = eval_result[6].cast<double>();

    root->policy_black.assign(bp_arr.data(), bp_arr.data() + n_actions);
    root->policy_white.assign(wp_arr.data(), wp_arr.data() + n_actions);

    const uint8_t* legal_b = lb_arr.data();
    const uint8_t* legal_w = lw_arr.data();
    normalize_policy(root->policy_black.data(), legal_b, n_actions);
    normalize_policy(root->policy_white.data(), legal_w, n_actions);

    root->is_expanded = true;

    if (root->game_over) {
        py::dict result;
        result["visit_counts"] = py::dict();
        result["total_visits"] = 0;
        return result;
    }

    build_candidates(root->policy_black.data(), root->policy_white.data(),
                     legal_b, legal_w, n_actions, tau,
                     max_candidates, root->candidates);
    root->child_N.resize(root->candidates.size(), 0);
    root->child_W.resize(root->candidates.size(), 0.0);

    // Add Dirichlet noise to root
    add_dirichlet_noise(root->policy_black.data(), legal_b,
                        n_actions, board_size, dirichlet_alpha,
                        dirichlet_epsilon, rng);
    add_dirichlet_noise(root->policy_white.data(), legal_w,
                        n_actions, board_size, dirichlet_alpha,
                        dirichlet_epsilon, rng);

    // Rebuild candidates after noise
    build_candidates(root->policy_black.data(), root->policy_white.data(),
                     legal_b, legal_w, n_actions, tau,
                     max_candidates, root->candidates);
    root->child_N.resize(root->candidates.size(), 0);
    root->child_W.resize(root->candidates.size(), 0.0);

    // ── MCTS visits ──
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

            int cidx = select_puct(*node, c_puct);
            path.push_back({node, cidx});

            if (node->children.find(cidx) == node->children.end()) {
                auto child = std::make_unique<Node>();
                py::object child_state = copy_state_fn(node->py_state);
                const JointAction& action = node->candidates[cidx];
                step_fn(child_state, action.black, action.white);
                child->py_state = child_state;
                node->children[cidx] = std::move(child);
            }
            node = node->children[cidx].get();
        }

        // Leaf node — evaluate via Python
        {
            py::tuple eval_result = eval_fn(node->py_state);
            auto bp = eval_result[0].cast<py::array_t<float>>();
            auto wp = eval_result[1].cast<py::array_t<float>>();
            float value = eval_result[2].cast<float>();
            auto lb = eval_result[3].cast<py::array_t<uint8_t>>();
            auto lw = eval_result[4].cast<py::array_t<uint8_t>>();
            node->game_over = eval_result[5].cast<bool>();
            node->terminal_value = eval_result[6].cast<double>();

            if (node->game_over) {
                node->is_expanded = true;
                value = (float)node->terminal_value;
            } else {
                node->policy_black.assign(bp.data(), bp.data() + n_actions);
                node->policy_white.assign(wp.data(), wp.data() + n_actions);
                normalize_policy(node->policy_black.data(), lb.data(), n_actions);
                normalize_policy(node->policy_white.data(), lw.data(), n_actions);
                node->is_expanded = true;

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

    // ── Collect results ──
    py::dict visit_counts;
    int total = 0;
    for (int i = 0; i < (int)root->candidates.size(); i++) {
        int n = root->child_N[i];
        if (n > 0) {
            auto key = py::make_tuple(root->candidates[i].black,
                                       root->candidates[i].white);
            visit_counts[key] = n;
            total += n;
        }
    }

    py::dict result;
    result["visit_counts"] = visit_counts;
    result["total_visits"] = total;
    return result;
}

// ──────────────────────────────────────────────────────────────
// Python module
// ──────────────────────────────────────────────────────────────

PYBIND11_MODULE(mcts_engine_callback, m) {
    m.doc() = "C++ MCTS engine for Gray Go joint-action search";
    m.def("run_mcts_cpp", &run_mcts_callback_cpp,
          py::arg("initial_state"),
          py::arg("eval_fn"),
          py::arg("copy_state_fn"),
          py::arg("step_fn"),
          py::arg("num_visits") = 400,
          py::arg("c_puct") = 1.5f,
          py::arg("tau") = 0.01f,
          py::arg("dirichlet_alpha") = 0.15f,
          py::arg("dirichlet_epsilon") = 0.30f,
          py::arg("board_size") = 9,
          py::arg("max_candidates") = 20,
          py::arg("seed") = 42,
          "Run MCTS with C++ tree traversal and Python leaf evaluation.\n"
          "\n"
          "eval_fn(state) -> (bp, wp, value, legal_b, legal_w, game_over, terminal_value)\n"
          "copy_state_fn(state) -> new_state\n"
          "step_fn(state, black_action, white_action) -> None (mutates state)\n"
          "\n"
          "Returns dict with 'visit_counts' and 'total_visits'."
    );
}
