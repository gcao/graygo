/*
 * engine.cpp — C++ Gray Go game engine with pybind11 bindings.
 *
 * Faithful port of engine.py: toroidal board, simultaneous moves,
 * collision-to-gray, two-stage capture, ko detection, forbidden points,
 * Chinese-style scoring with gray-faction split.
 *
 * Build:
 *   g++ -O3 -shared -std=c++17 -fPIC \
 *       $(python3 -m pybind11 --includes) \
 *       engine.cpp -o graygo_engine$(python3-config --extension-suffix)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <set>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstdint>

namespace py = pybind11;

// ──────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────

static constexpr int EMPTY = 0;
static constexpr int BLACK = 1;
static constexpr int WHITE = 2;
static constexpr int GRAY  = 3;

static constexpr int BLACK_PLAYER = 0;
static constexpr int WHITE_PLAYER = 1;

static constexpr int MAX_SIZE = 19;

// ──────────────────────────────────────────────────────────────
// Board
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
        int x = action % size;
        int y = action / size;
        return grid[y * size + x];
    }

    void set_action(int action, int value) {
        int x = action % size;
        int y = action / size;
        grid[y * size + x] = (int8_t)value;
    }

    // Toroidal neighbors
    void neighbors(int x, int y, int out_x[4], int out_y[4]) const {
        int s = size;
        out_x[0] = (x - 1 + s) % s; out_y[0] = y;
        out_x[1] = (x + 1) % s;     out_y[1] = y;
        out_x[2] = x;               out_y[2] = (y - 1 + s) % s;
        out_x[3] = x;               out_y[3] = (y + 1) % s;
    }

    // Flood-fill to find group and liberties
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

    // Find all dead groups (zero liberties)
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

    // Remove groups and return counts per color
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

    // Scoring: Chinese style with gray faction split
    std::pair<double, double> score() const {
        double black_score = 0.0;
        double white_score = 0.0;
        int board_points = size * size;

        // Count stones
        for (int action = 0; action < board_points; action++) {
            int color = get_action(action);
            if (color == BLACK) black_score += 1.0;
            else if (color == WHITE) white_score += 1.0;
            else if (color == GRAY) {
                black_score += 0.5;
                white_score += 0.5;
            }
        }

        // Territory: flood-fill empty regions
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

            // territory_player_shares
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
// GameState
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

    py::array_t<uint8_t> legal_actions(int player) const {
        int total_actions = size * size + 1;
        auto result = py::array_t<uint8_t>(total_actions);
        auto buf = result.mutable_unchecked<1>();

        const auto& forbidden = (player == BLACK_PLAYER) ? forbidden_black : forbidden_white;

        for (int i = 0; i < size * size; i++) {
            buf(i) = (board.get_action(i) == EMPTY && forbidden.count(i) == 0) ? 1 : 0;
        }
        buf(size * size) = 1;  // pass always legal
        return result;
    }

    bool is_legal_action(int player, int action) const {
        if (is_pass(action)) return true;
        if (action < 0 || action >= size * size) return false;
        if (board.get_action(action) != EMPTY) return false;
        const auto& forbidden = (player == BLACK_PLAYER) ? forbidden_black : forbidden_white;
        return forbidden.count(action) == 0;
    }

    void step(int black_action, int white_action) {
        if (game_over) {
            throw std::runtime_error("Game is already over.");
        }
        if (!is_legal_action(BLACK_PLAYER, black_action)) {
            throw std::invalid_argument("Illegal black action: " + std::to_string(black_action));
        }
        if (!is_legal_action(WHITE_PLAYER, white_action)) {
            throw std::invalid_argument("Illegal white action: " + std::to_string(white_action));
        }

        bool both_pass = is_pass(black_action) && is_pass(white_action);
        if (both_pass) {
            consecutive_double_passes++;
        } else {
            consecutive_double_passes = 0;
        }

        // Save board state before modifications
        Board board_before = board;

        std::set<int> newly_placed;

        if (!both_pass) {
            bool black_is_pass = is_pass(black_action);
            bool white_is_pass = is_pass(white_action);

            if (!black_is_pass && !white_is_pass && black_action == white_action) {
                // Collision -> gray
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

            // Two-stage capture resolution
            // Step 1: remove dead groups NOT containing newly placed stones
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

            // Step 2: recheck and remove remaining dead groups
            dead = board.dead_groups();
            board.remove_groups(dead);
        }

        bool board_changed = !board.equals(board_before);

        if (board_changed) {
            forbidden_black.clear();
            forbidden_white.clear();

            // Ko detection
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
            // Board unchanged -> forbid the moves that caused no change (suicide prevention)
            if (!is_pass(black_action)) {
                forbidden_black.insert(black_action);
            }
            if (!is_pass(white_action)) {
                forbidden_white.insert(white_action);
            }
        }

        // Update ko history (keep last 2)
        ko_history.push_back({board_before, black_action, white_action});
        if (ko_history.size() > 2) {
            ko_history.erase(ko_history.begin());
        }

        turn_number++;
        if (both_pass && consecutive_double_passes >= 2) {
            game_over = true;
        }
    }

    std::pair<double, double> score() const {
        return board.score();
    }

    int winner_color() const {
        auto [bs, ws] = board.score();
        if (bs > ws) return BLACK;
        if (ws > bs) return WHITE;
        return EMPTY;
    }

    // Returns 0=BLACK_PLAYER, 1=WHITE_PLAYER, -1=draw
    int winner_player() const {
        int wc = winner_color();
        if (wc == BLACK) return BLACK_PLAYER;
        if (wc == WHITE) return WHITE_PLAYER;
        return -1;
    }

    // Get board as numpy array for Python interop
    py::array_t<int8_t> get_board_numpy() const {
        auto result = py::array_t<int8_t>({size, size});
        auto buf = result.mutable_unchecked<2>();
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
                buf(y, x) = board.grid[y * size + x];
        return result;
    }

    // Get forbidden points as Python sets
    py::list get_forbidden_points() const {
        py::list result;
        py::set fb, fw;
        for (int a : forbidden_black) fb.add(py::int_(a));
        for (int a : forbidden_white) fw.add(py::int_(a));
        result.append(fb);
        result.append(fw);
        return result;
    }
};

// ──────────────────────────────────────────────────────────────
// Python module
// ──────────────────────────────────────────────────────────────

PYBIND11_MODULE(graygo_engine, m) {
    m.doc() = "C++ Gray Go game engine";

    m.attr("EMPTY") = EMPTY;
    m.attr("BLACK") = BLACK;
    m.attr("WHITE") = WHITE;
    m.attr("GRAY") = GRAY;
    m.attr("BLACK_PLAYER") = BLACK_PLAYER;
    m.attr("WHITE_PLAYER") = WHITE_PLAYER;

    py::class_<GameState>(m, "GameState")
        .def(py::init<int>(), py::arg("size") = 9)
        .def("copy", &GameState::copy)
        .def("step", &GameState::step, py::arg("black_action"), py::arg("white_action"))
        .def("legal_actions", &GameState::legal_actions, py::arg("player"))
        .def("is_legal_action", &GameState::is_legal_action, py::arg("player"), py::arg("action"))
        .def("score", &GameState::score)
        .def("winner_color", &GameState::winner_color)
        .def("winner_player", &GameState::winner_player)
        .def("get_board_numpy", &GameState::get_board_numpy)
        .def("get_forbidden_points", &GameState::get_forbidden_points)
        .def_readonly("size", &GameState::size)
        .def_readonly("turn_number", &GameState::turn_number)
        .def_readonly("game_over", &GameState::game_over)
        .def_readonly("consecutive_double_passes", &GameState::consecutive_double_passes)
        .def("pass_action", &GameState::pass_action)
        .def("is_pass", &GameState::is_pass, py::arg("action"));
}
