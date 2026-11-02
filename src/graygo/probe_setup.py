"""Actual default-game setup traces for the controlled last-liberty probes."""
from collections import deque
from functools import lru_cache
from .engine import BLACK,WHITE,GRAY,Rules,State,groups,neighbors,resolve


def _bfs(board,color,anchor,excluded):
    adjacent=neighbors(9);seeds=sorted(p for p in adjacent[anchor] if board[p]==color and p!=excluded)
    if not seeds:return None
    order=[];seen=set(seeds);queue=deque(seeds)
    while queue:
        p=queue.popleft();order.append(p)
        for q in sorted(adjacent[p]):
            if q!=excluded and board[q]==color and q not in seen:seen.add(q);queue.append(q)
    expected={p for p,c in enumerate(board) if c==color and p!=excluded}
    return order if seen==expected else None


def _ring_plan(dense,sparse,anchor):
    friendly=BLACK if dense.count(BLACK)>dense.count(WHITE) else WHITE;adjacent=neighbors(9)
    candidates=[]
    k=min(sum(bool(c) for c in board)-1 for board in (dense,sparse));extra=196-k
    for point,color in enumerate(dense):
        if color!=friendly or not all(dense[q]==friendly for q in adjacent[point]):continue
        order=_bfs(dense,friendly,anchor,point)
        if order is None:continue
        ready=max(order.index(q) for q in adjacent[point])+1
        if 2*(k-ready)>=extra:candidates.append((ready,point,tuple(order)))
    if not candidates:raise ValueError('no early protected ring for a 199-round setup')
    ready,point,order=min(candidates)
    return friendly,ready,point,order


@lru_cache(maxsize=1024)
def replay_setup(moves):
    state=State()
    for pair in moves:
        if len(pair)!=2 or any(type(action) is not int for action in pair):raise ValueError('exact setup action pairs required')
        state=resolve(state,*pair).state
    return state


def reachable_pair(before,after,changed_point):
    dense,sparse=(before,after) if before.count(0)==1 else (after,before)
    if dense.count(0)!=1 or sparse.count(0)!=2:raise ValueError('one/two-liberty-hole pair required')
    anchor=dense.index(0);friendly,ready,ring,friend_order=_ring_plan(dense,sparse,anchor)
    results=[]
    for board in (before,after):
        opponent=3-friendly;other_order=_bfs(board,opponent,anchor,ring)
        if other_order is None:raise ValueError('opposing setup region is disconnected')
        order=[(friendly,p) for p in friend_order]+[(opponent,p) for p in other_order]
        k=len(order);extra=196-k;state=State();moves=[]
        def step(black,white):
            nonlocal state
            state=resolve(state,black,white).state;moves.append((black,white))
        def place(color,point):step(point,81) if color==BLACK else step(81,point)
        for index,(color,point) in enumerate(order):
            if index>=ready and extra:
                previous=state.board;place(opponent,ring)
                if state.board!=previous or ring not in state.forbidden(opponent):raise ValueError('setup suicide did not preserve its protected ring')
                extra-=1
                if extra:step(81,81);extra-=1
            place(color,point)
        if extra or len(moves)!=196:raise ValueError('setup padding did not reach the exact round')
        previous=state.board;place(opponent,ring)
        if state.board!=previous:raise ValueError('final setup suicide changed the board')
        step(81,81);place(friendly,ring)
        if state.rules!=Rules() or state.rounds_played!=199 or state.done or state.board!=tuple(board) or state.forbidden_black or state.forbidden_white or state.pass_streak:
            raise ValueError('setup does not produce the complete default-game probe root')
        trace=tuple(moves)
        if replay_setup(trace)!=state:raise ValueError('setup trace replay differs')
        results.append((state,trace))
    return tuple(results)


def _anchored_order(board,color,excluded,targets=()):
    """Grow every component from a permanent empty point, prioritizing targets."""
    excluded={excluded} if isinstance(excluded,int) else set(excluded)
    adjacent=neighbors(9);points={p for p,c in enumerate(board) if c==color and p not in excluded}
    seeds=sorted(p for p in points if any(board[q]==0 for q in adjacent[p]))
    parent={p:None for p in seeds};queue=deque(seeds);breadth=[]
    while queue:
        point=queue.popleft();breadth.append(point)
        for other in sorted(adjacent[point]):
            if other in points and other not in parent:parent[other]=point;queue.append(other)
    if set(parent)!=points:return None
    early=[];seen=set()
    for target in sorted(targets):
        if target not in parent:return None
        path=[];point=target
        while point is not None and point not in seen:path.append(point);point=parent[point]
        for point in reversed(path):early.append(point);seen.add(point)
    return tuple(early+[p for p in breadth if p not in seen]),len(early)


def _multigroup_plan(board,ring,tail=None):
    color=board[ring];adjacent=neighbors(9)
    excluded={ring} if tail is None else {ring,tail}
    friendly=_anchored_order(board,color,excluded,adjacent[ring])
    if friendly is None:return None
    order,ready=friendly
    result=[(color,p) for p in order]
    for other in (BLACK,WHITE,GRAY):
        if other==color:continue
        part=_anchored_order(board,other,excluded)
        if part is None:return None
        result.extend((other,p) for p in part[0])
    extra=196-len(result)
    if not 0<=extra<=2*(len(result)-ready):return None
    return color,ready,tuple(result),extra,tail


def reachable_multigroup_pair(before,after):
    """Reach dense multi-faction roots by 199 actual default-rule rounds.

    All growth stays connected to permanent liberties. A shared protected ring
    permits bounded suicide/pass padding between board changes. Gray growth is
    produced by collisions. No history/counter is assigned synthetically.
    """
    boards=tuple(State(Rules(),board).board for board in (before,after))
    if any(not g.liberties for board in boards for g in groups(board,9)):
        raise ValueError('setup requires stable groups with permanent liberties')
    adjacent=neighbors(9);candidates=[]
    for ring,color in enumerate(boards[0]):
        if not color or any(board[ring]!=color or any(board[p]!=color for p in adjacent[ring]) for board in boards):continue
        # A gray final placement needs both players. The padding suicide forbids
        # one of them, so reserve another actual board-changing placement for
        # round 198 instead of leaving the restriction in place with a pass.
        tails=([p for p,c in enumerate(boards[0]) if c and p!=ring and p not in adjacent[ring] and boards[1][p]==c]
               if color==GRAY else [None])
        for tail in tails:
            plans=tuple(_multigroup_plan(board,ring,tail) for board in boards)
            if any(plan is None for plan in plans):continue
            candidates.append((max(plan[1] for plan in plans),sum(plan[1] for plan in plans),ring,-1 if tail is None else tail,plans))
    if not candidates:raise ValueError('no shared protected ring with enough legal padding rounds')
    _,_,ring,_,plans=min(candidates,key=lambda item:item[:4]);results=[]
    for board,(color,ready,order,extra,tail) in zip(boards,plans):
        state=State();moves=[]
        def step(black,white):
            nonlocal state
            state=resolve(state,black,white).state;moves.append((black,white))
        def place(faction,point):
            if faction==GRAY:step(point,point)
            elif faction==BLACK:step(point,81)
            else:step(81,point)
        enemy=WHITE if color==BLACK else BLACK
        def suicide():
            previous=state.board;place(enemy,ring)
            if state.board!=previous or ring not in state.forbidden(enemy):
                raise ValueError('protected-ring padding changed the board or missed its restriction')
        for index,(faction,point) in enumerate(order):
            if index>=ready and extra:
                suicide();extra-=1
                if extra:step(81,81);extra-=1
            previous=state.board;place(faction,point)
            expected=list(previous);expected[point]=faction
            if state.board!=tuple(expected):raise ValueError('anchored setup placement captured or lost a stone')
        if extra or len(moves)!=196:raise ValueError('multigroup setup did not reach its exact padding boundary')
        suicide()
        if tail is None:step(81,81)
        else:place(board[tail],tail)
        place(color,ring)
        if (state.rules!=Rules() or state.rounds_played!=199 or state.done or state.board!=board
                or state.forbidden_black or state.forbidden_white or state.pass_streak):
            raise ValueError('multigroup setup did not produce the complete default-game root')
        trace=tuple(moves)
        if replay_setup(trace)!=state:raise ValueError('multigroup setup trace replay differs')
        results.append((state,trace))
    return tuple(results)
