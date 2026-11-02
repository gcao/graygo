"""Candidate-independent, symmetry-separated gray-exposure start families.

These are explicitly synthetic curriculum roots with fresh public metadata.
They are not collision-probe certificates or claims of empty-board reachability.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import numpy as np
from .board_augmentation import permutations
from .confidence import canonical_sha
from .engine import BLACK, WHITE, GRAY, Rules, State, groups, neighbors, transform
from .evaluation import seed_for

CATALOG_VERSION = 'graygo/gray-exposure-catalog/1'
STARTS_VERSION = 'graygo/gray-exposure-starts/1'
GENERATOR_VERSION = 'graygo/small-gray-balanced-pairs/1'


@lru_cache(maxsize=8192)
def _canonical(board):
    _, inverse = permutations(9)
    views = np.asarray(board, dtype=np.uint8)[inverse]
    swapped = np.array([0, 2, 1, 3], dtype=np.uint8)[views]
    return tuple(min(row.tobytes() for row in np.concatenate((views, swapped))))


def canonical_board(board):
    board = tuple(board)
    if len(board) != 81 or any(type(c) is not int or c not in range(4) for c in board):
        raise ValueError('81 exact board colors required')
    return _canonical(board)


def family_id(board):
    return hashlib.sha256(b'graygo/gray-exposure-family/1/9\0' + bytes(canonical_board(board))).hexdigest()


def _validate_board(board):
    if canonical_board(board) != board:
        raise ValueError('catalog boards must use their canonical family representative')
    State(Rules(), board)  # Includes the complete zero-liberty-group check.
    if not 1 <= board.count(GRAY) <= 6 or any(len(g.stones) > 3 for g in groups(board, 9) if g.color == GRAY):
        raise ValueError('one to six gray stones in groups of at most three required')
    if board.count(BLACK) != board.count(WHITE) or board.count(BLACK) > 12:
        raise ValueError('generator v1 uses zero to twelve balanced black/white pairs')


@dataclass(frozen=True)
class ExposureCatalog:
    seed: int
    train: tuple
    heldout: tuple

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('nonnegative independent catalog seed required')
        object.__setattr__(self, 'train', tuple(tuple(board) for board in self.train))
        object.__setattr__(self, 'heldout', tuple(tuple(board) for board in self.heldout))
        if not self.train or not self.heldout or len(self.train) + len(self.heldout) > 4096:
            raise ValueError('nonempty bounded training and held-out family splits required')
        seen = set()
        for board in (*self.train, *self.heldout):
            _validate_board(board)
            identity = family_id(board)
            if identity in seen:
                raise ValueError('symmetry/color-equivalent families cannot cross or repeat within splits')
            seen.add(identity)

    def to_dict(self):
        return {'version': CATALOG_VERSION, 'generator': GENERATOR_VERSION, 'seed': self.seed,
            'size': 9, 'gray_stones': [1, 6], 'max_gray_group': 3, 'balanced_pairs': [0, 12],
            'symmetry_equivalence': 'D4 x torus translations x black/white exchange',
            'root_metadata': 'empty history/forbidden sets, zero pass streak/rounds, full 200-round allowance',
            'splits': {name: [{'family_id': family_id(board), 'board': list(board)} for board in boards]
                       for name, boards in (('train', self.train), ('heldout', self.heldout))}}

    @property
    def sha256(self):return canonical_sha(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get('version') != CATALOG_VERSION or value.get('generator') != GENERATOR_VERSION:
            raise ValueError('unknown gray-exposure catalog/generator version')
        expected = {'version','generator','seed','size','gray_stones','max_gray_group','balanced_pairs','symmetry_equivalence','root_metadata','splits'}
        if set(value) != expected or not isinstance(value['splits'],dict) or set(value['splits']) != {'train','heldout'}:
            raise ValueError('catalog schema fields differ')
        for families in value['splits'].values():
            if not isinstance(families,list) or not families or len(families)>4096:
                raise ValueError('bounded nonempty family lists required')
            if any(not isinstance(f,dict) or set(f)!={'family_id','board'} or not isinstance(f['board'],list) for f in families):
                raise ValueError('catalog family schema differs')
        result = cls(value['seed'], tuple(tuple(f['board']) for f in value['splits']['train']),
                     tuple(tuple(f['board']) for f in value['splits']['heldout']))
        if result.to_dict() != value:
            raise ValueError('catalog fields, family identities or reset protocol differ')
        regenerated = create_catalog(result.seed, train_families=len(result.train), heldout_families=len(result.heldout))
        if regenerated != result:
            raise ValueError('catalog does not match its independent generator seed/split')
        return result


def create_catalog(seed, *, train_families=64, heldout_families=32, max_attempts=100000):
    if type(seed) is not int or seed < 0 or any(type(v) is not int or v < 1 for v in (train_families, heldout_families, max_attempts)):
        raise ValueError('independent seed and positive catalog generation bounds required')
    total = train_families + heldout_families
    if total > 4096 or max_attempts > 1_000_000:
        raise ValueError('catalog family budget exceeded')
    rng = np.random.default_rng(seed_for(seed, 'exposure-catalog', GENERATOR_VERSION))
    adjacent = neighbors(9); accepted = []; seen = set()
    for _ in range(max_attempts):
        index = len(accepted); gray = 1 + index % 6
        # Vary the first connected cluster as well as the total gray count.
        cluster_size = 1 + (index // 6) % min(3, gray)
        stones = {int(rng.integers(81))}
        while len(stones) < cluster_size:
            frontier = sorted({q for p in stones for q in adjacent[p]} - stones)
            stones.add(frontier[int(rng.integers(len(frontier)))])
        if len(stones) < gray:
            remaining = np.array([p for p in range(81) if p not in stones])
            stones.update(map(int, rng.choice(remaining, gray - len(stones), replace=False)))
        board = [0] * 81
        for point in stones:board[point] = GRAY
        pairs = int(rng.integers(13)); free = [p for p in range(81) if board[p] == 0]
        occupied = rng.choice(free, 2 * pairs, replace=False)
        for point in occupied[:pairs]:board[int(point)] = BLACK
        for point in occupied[pairs:]:board[int(point)] = WHITE
        board = canonical_board(board)
        try:_validate_board(board)
        except ValueError:continue
        identity = family_id(board)
        if identity in seen:continue
        accepted.append(board); seen.add(identity)
        if len(accepted) == total:
            split_rng = np.random.default_rng(seed_for(seed, 'exposure-catalog', 'split', GENERATOR_VERSION))
            order = split_rng.permutation(total)
            return ExposureCatalog(seed, tuple(accepted[int(i)] for i in order[:train_families]),
                                   tuple(accepted[int(i)] for i in order[train_families:]))
    raise RuntimeError('gray-exposure family generation budget exhausted; no partial catalog')


@dataclass(frozen=True)
class ExposureStarts:
    catalog: ExposureCatalog
    split: str = 'train'
    mode: str = 'mixed'

    def __post_init__(self):
        if not isinstance(self.catalog, ExposureCatalog) or self.split not in ('train', 'heldout') or self.mode not in ('mixed', 'gray-only'):
            raise ValueError('frozen catalog, declared split and mixed/gray-only mode required')

    @property
    def identity(self):
        return {'kind': 'independent-gray-exposure', 'version': STARTS_VERSION, 'catalog_sha256': self.catalog.sha256,
                'split': self.split, 'mode': self.mode, 'gray_probability': .5 if self.mode == 'mixed' else 1.,
                'augmentation': 'uniform D4/torus/color exchange', 'round_allowance': 200}

    @property
    def sha256(self):return canonical_sha(self.identity)

    def sample_with_family(self, seed, rules):
        if not isinstance(rules, Rules) or rules.size != 9 or rules.max_rounds != 200:
            raise ValueError('official 9x9/200-round curriculum rules required')
        rng = np.random.default_rng(seed); coin = rng.random()
        if self.mode == 'mixed' and coin >= .5:return State(rules), None
        boards = self.catalog.train if self.split == 'train' else self.catalog.heldout
        board = boards[int(rng.integers(len(boards)))]
        rotation = int(rng.integers(4)); reflect = bool(rng.integers(2)); shift = tuple(map(int, rng.integers(9, size=2)))
        swap = bool(rng.integers(2))
        state = transform(State(rules, board), rotation=rotation, reflect=reflect, shift=shift, swap=swap)
        return state, family_id(board)

    def sample(self, seed, rules):return self.sample_with_family(seed, rules)[0]
