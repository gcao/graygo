"""Actual default-game setup traces for the controlled last-liberty probes."""
from collections import deque
from functools import lru_cache
from .engine import BLACK,WHITE,Rules,State,groups,neighbors,resolve


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
