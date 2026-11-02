// Complete graygo/rule-state/1 validation and graygo/encoding/1 expansion.
// The independent Python engine/encoder remains the semantic reference.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static std::uint64_t little(const std::uint8_t* p, int bytes) {
    std::uint64_t value=0;
    for(int i=0;i<bytes;++i) value|=std::uint64_t(p[i])<<(8*i);
    return value;
}
static int width(std::uint64_t value) {
    int bytes=1;
    while(value>=256) {++bytes;value>>=8;}
    return bytes;
}

extern "C" EXPORT unsigned graygo_rule_decode_abi() {return 1;}

// Return 0 on success, 1 for invalid configuration, 2 for invalid state,
// and 3 for allocation/internal failure. No C++ exception crosses the C ABI.
extern "C" EXPORT int graygo_decode_rule_states(
    const std::uint8_t* packets,std::size_t rows,std::size_t stride,
    const std::uint8_t* valid,int size,std::uint64_t max_rounds,
    const std::uint8_t* prefix,std::size_t prefix_bytes,
    int round_bytes,int action_bytes,
    std::uint8_t* spatial,float* globals,std::uint8_t* legal,
    std::size_t* bad_row) noexcept {
    if(bad_row) *bad_row=0;
    if(size<3 || size>19 || max_rounds<1 || max_rounds>std::numeric_limits<std::uint32_t>::max() ||
       prefix_bytes!=11 || !prefix || round_bytes!=width(max_rounds)) return 1;
    const int points=size*size,board_bytes=(2*points+7)/8,mask_bytes=(points+7)/8;
    const std::size_t header=prefix_bytes+3+round_bytes;
    if(action_bytes!=width(points) || stride!=header+3*board_bytes+2*mask_bytes+4*action_bytes) return 1;
    if(rows && (!packets || !valid || !spatial || !globals || !legal)) return 1;
    try {
        std::vector<int> neighbors(points*4),board(points),seen(points),liberty_mark(points),queue(points),stones(points);
        for(int p=0;p<points;++p) {
            const int r=p/size,c=p%size;
            neighbors[4*p]=((r+size-1)%size)*size+c;neighbors[4*p+1]=((r+1)%size)*size+c;
            neighbors[4*p+2]=r*size+(c+size-1)%size;neighbors[4*p+3]=r*size+(c+1)%size;
        }
        for(std::size_t row=0;row<rows;++row) {
            if(bad_row) *bad_row=row;
            auto* x=spatial+row*30*points;auto* g=globals+row*14;auto* mask=legal+row*(points+1);
            std::memset(x,0,30*points);std::memset(g,0,14*sizeof(float));std::memset(mask,0,points+1);
            if(valid[row]==0) continue;
            if(valid[row]!=1) return 2;
            const auto* data=packets+row*stride;
            if(std::memcmp(data,prefix,prefix_bytes)!=0) return 2;
            const int player=data[prefix_bytes],count=data[prefix_bytes+1],passes=data[prefix_bytes+2];
            const auto rounds=little(data+prefix_bytes+3,round_bytes);
            if((player!=1 && player!=2) || count>2 || passes>2 || rounds>max_rounds || count!=std::min<std::uint64_t>(rounds,2)) return 2;
            const auto* boards=data+header;const auto* restrictions=boards+3*board_bytes;
            const auto* moves=restrictions+2*mask_bytes;
            if(points%4) for(int slot=0;slot<3;++slot)
                if(boards[(slot+1)*board_bytes-1]>>(2*(points%4))) return 2;
            if(points%8) for(int p=0;p<2;++p)
                if(restrictions[(p+1)*mask_bytes-1]>>(points%8)) return 2;
            int black[2]={0,0},white[2]={0,0};
            for(int slot=0;slot<2;++slot) {
                const auto* hboard=boards+(slot+1)*board_bytes;
                const auto* hmove=moves+slot*2*action_bytes;
                if(slot<2-count) {
                    for(int i=0;i<board_bytes;++i) if(hboard[i]) return 2;
                    for(int i=0;i<2*action_bytes;++i) if(hmove[i]) return 2;
                } else {
                    black[slot]=static_cast<int>(little(hmove,action_bytes));
                    white[slot]=static_cast<int>(little(hmove+action_bytes,action_bytes));
                    if(black[slot]>points || white[slot]>points) return 2;
                }
            }
            int trailing=0;
            for(int slot=1;slot>=2-count;--slot) {
                if(black[slot]!=points || white[slot]!=points) break;
                ++trailing;
            }
            if(trailing!=passes) return 2;
            const int faction[4]={3,player==1?0:1,player==2?0:1,2};
            for(int p=0;p<points;++p) {
                board[p]=(boards[p/4]>>(2*(p%4)))&3;
                x[faction[board[p]]*points+p]=1;
            }
            std::fill(seen.begin(),seen.end(),0);std::fill(liberty_mark.begin(),liberty_mark.end(),0);
            int group=0;
            for(int start=0;start<points;++start) {
                const int color=board[start];
                if(color==0 || seen[start]) continue;
                ++group;int head=0,tail=1,nstones=0,nliberties=0;
                queue[0]=start;seen[start]=1;
                while(head<tail) {
                    const int point=queue[head++];stones[nstones++]=point;
                    for(int direction=0;direction<4;++direction) {
                        const int other=neighbors[point*4+direction];
                        if(board[other]==0) {
                            if(liberty_mark[other]!=group) {liberty_mark[other]=group;++nliberties;}
                        } else if(board[other]==color && !seen[other]) {
                            seen[other]=1;queue[tail++]=other;
                        }
                    }
                }
                // Resolved states must not contain dead groups, including terminals.
                if(nliberties==0) return 2;
                const int channel=4+4*faction[color]+std::min(nliberties,4)-1;
                for(int i=0;i<nstones;++i) x[channel*points+stones[i]]=1;
            }
            for(int slot=2-count;slot<2;++slot) {
                g[slot]=1;
                const auto* hboard=boards+(slot+1)*board_bytes;
                for(int p=0;p<points;++p) {
                    const int color=(hboard[p/4]>>(2*(p%4)))&3;
                    x[(16+4*slot+faction[color])*points+p]=1;
                }
                const int mine=player==1?black[slot]:white[slot],theirs=player==1?white[slot]:black[slot];
                const int actions[2]={mine,theirs};
                for(int who=0;who<2;++who) {
                    if(actions[who]==points) g[2+2*slot+who]=1;
                    else x[(24+2*slot+who)*points+actions[who]]=1;
                }
            }
            const auto* own=restrictions+(player==1?0:mask_bytes);
            const auto* other=restrictions+(player==1?mask_bytes:0);
            const bool done=passes==2 || rounds==max_rounds;
            for(int p=0;p<points;++p) {
                const int blocked=(own[p/8]>>(p%8))&1;
                x[28*points+p]=blocked;x[29*points+p]=(other[p/8]>>(p%8))&1;
                if(!done && board[p]==0 && !blocked) mask[p]=1;
            }
            if(!done) mask[points]=1;
            const auto remaining=max_rounds-rounds;
            g[6]=static_cast<float>(passes);g[7]=static_cast<float>(static_cast<double>(remaining)/static_cast<double>(max_rounds));
            const int thresholds[6]={1,2,3,5,10,25};
            for(int i=0;i<6;++i) g[8+i]=remaining<=static_cast<unsigned>(thresholds[i]);
        }
        return 0;
    } catch(...) {return 3;}
}
