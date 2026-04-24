# Gray Go

Clean migration of the current Gray Go self-play training implementation.

## Layout

- `engine.py`, `engine.cpp`: Python and pybind11 game engines.
- `model.py`: current neural network with policy, value, and five auxiliary heads.
- `selfplay.py`: Python-driven MCTS self-play using the callback C++ MCTS engine.
- `selfplay_cpp.py`: optional pure C++/libtorch self-play runner.
- `train.py`, `gate.py`, `run.py`: training, promotion gate, and orchestration entry point.
- `mcts_engine_callback.cpp`: C++ joint-action MCTS with a Python eval callback.
- `mcts_engine.cpp`: pure C++ self-play engine with libtorch inference.

## Build

```bash
make all
```

For the pure C++ self-play path:

```bash
make cpp-selfplay
python run.py --use-cpp-selfplay --device cuda
```

Generated checkpoints are written to `checkpoints/` by default and are ignored by git.
