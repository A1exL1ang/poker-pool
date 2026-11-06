// Independent standard Hold'em evaluator and postflop feature extraction.
// Cards: 4 * (rank-2) + suit. Scores compare directly; category = score >> 20.
// Keep feature_vector semantics synchronized with pokerbot/equity.py.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace holdem {

inline int highbit(unsigned x) { return 31 - __builtin_clz(x); }
inline int popcount(unsigned x) { return __builtin_popcount(x); }

struct Tables {
    std::array<int,8192> straight{}, top5{}, window_count{}, window_high{}, straight_outs{};
    Tables() {
        for (unsigned mask=0; mask<8192; ++mask) {
            unsigned rest=mask;
            int value=0;
            for (int shift=16; rest && shift>=0; shift-=4) {
                int high=highbit(rest);
                value|=(high+2)<<shift;
                rest^=1u<<high;
            }
            top5[mask]=value;
            unsigned runs=mask&(mask>>1)&(mask>>2)&(mask>>3)&(mask>>4);
            straight[mask]=runs ? highbit(runs)+6 : (mask&0x100f)==0x100f ? 5 : 0;
            unsigned outs=0;
            int best=0, top=0;
            for (int low=0; low<=9; ++low) {
                unsigned window=low==9 ? 0x100f : 0x1fu<<low;
                int high=low==9 ? 5 : low+6;
                int count=popcount(mask&window);
                if (count>best || (count==best && high>top)) { best=count; top=high; }
                if (count==4) outs|=window&~mask;
            }
            window_count[mask]=best;
            window_high[mask]=top;
            straight_outs[mask]=straight[mask] ? 0 : popcount(outs);
        }
    }
};
inline const Tables& tables() { static const Tables instance; return instance; }

template<class Card>
inline int evaluate(const Card* cards, int n) {
    const auto& t=tables();
    unsigned suits[4]={0,0,0,0};
    for (int i=0;i<n;++i) suits[cards[i]&3]|=1u<<(cards[i]>>2);
    unsigned a=suits[0],b=suits[1],c=suits[2],d=suits[3], ranks=a|b|c|d;
    unsigned flush=popcount(a)>=5?a:popcount(b)>=5?b:popcount(c)>=5?c:popcount(d)>=5?d:0;
    if (flush && t.straight[flush]) return (8<<20)|(t.straight[flush]<<16);
    unsigned quads=a&b&c&d;
    if (quads) {
        int high=highbit(quads);
        return (7<<20)|((high+2)<<16)|((highbit(ranks^(1u<<high))+2)<<12);
    }
    unsigned trips=(a&b&c)|(a&b&d)|(a&c&d)|(b&c&d);
    unsigned pairs=(a&b)|(a&c)|(a&d)|(b&c)|(b&d)|(c&d);
    if (trips) {
        int high=highbit(trips);
        unsigned other=pairs^(1u<<high);
        if (other) return (6<<20)|((high+2)<<16)|((highbit(other)+2)<<12);
    }
    if (flush) return (5<<20)|t.top5[flush];
    if (t.straight[ranks]) return (4<<20)|(t.straight[ranks]<<16);
    if (trips) {
        int high=highbit(trips);
        return (3<<20)|((high+2)<<16)|((t.top5[ranks^(1u<<high)]>>4)&0xff00);
    }
    if (pairs) {
        int high=highbit(pairs);
        unsigned other=pairs^(1u<<high);
        if (other) {
            int low=highbit(other);
            int kicker=highbit(ranks^(1u<<high)^(1u<<low))+2;
            return (2<<20)|((high+2)<<16)|((low+2)<<12)|(kicker<<8);
        }
        return (1<<20)|((high+2)<<16)|((t.top5[ranks^(1u<<high)]>>4)&0xfff0);
    }
    return t.top5[ranks];
}
template<class Card, std::size_t N>
inline int evaluate(const std::array<Card,N>& cards) { return evaluate(cards.data(),N); }
template<class Card>
inline int evaluate(const std::vector<Card>& cards) { return evaluate(cards.data(),cards.size()); }

constexpr int kFeatureCount=44;
constexpr std::array<const char*,kFeatureCount> kFeatureNames={
    "board_length", "category", "made_rank_1", "made_rank_2", "made_rank_3", "made_rank_4", "made_rank_5",
    "hole_high", "hole_low", "hole_pair", "hole_suited",
    "board_rank_1", "board_rank_2", "board_rank_3", "board_rank_4", "board_rank_5",
    "board_unique_ranks", "board_max_multiplicity", "board_pairs", "board_trips",
    "board_max_suit", "total_max_suit", "flush_hole_count", "flush_hole_high", "flush_hole_low", "flush_nut_gap",
    "straight_window_count", "board_straight_window_count", "straight_out_ranks", "straight_window_high",
    "hole_overcards", "hole_high_board_hits", "hole_low_board_hits",
    "board_category", "board_made_rank_1", "board_made_rank_2", "board_made_rank_3", "board_made_rank_4", "board_made_rank_5",
    "board_suit_of_high_hole", "board_suit_of_low_hole", "board_gap", "hole_gap", "board_ace"
};

inline std::array<int,kFeatureCount> feature_vector(const int* hole,const int* board,int n) {
    const auto& t=tables();
    int h0=hole[0],h1=hole[1];
    if (h0/4<h1/4) std::swap(h0,h1);
    int hi=h0/4+2,lo=h1/4+2,hs0=h0&3,hs1=h1&3;
    int branks[13]={},bsuits[4]={},tsuits[4]={},sorted[5]={},cards[7];
    unsigned bmasks[4]={},bmask=0;
    cards[0]=hole[0];cards[1]=hole[1];
    for (int i=0;i<n;++i) {
        int card=board[i],r=card>>2,s=card&3;
        ++branks[r];++bsuits[s];bmasks[s]|=1u<<r;bmask|=1u<<r;
        sorted[i]=r+2;cards[i+2]=card;
    }
    std::sort(sorted,sorted+n,std::greater<int>());
    for (int i=0;i<4;++i) tsuits[i]=bsuits[i];
    ++tsuits[hs0];++tsuits[hs1];
    int suit=0;
    auto holehigh=[&](int s) { return hs0==s?hi:hs1==s?lo:0; };
    for (int s=1;s<4;++s) if (tsuits[s]>tsuits[suit] || (tsuits[s]==tsuits[suit] && (holehigh(s)>holehigh(suit) || (holehigh(s)==holehigh(suit) && bmasks[s]>bmasks[suit])))) suit=s;
    int fhi=holehigh(suit),flo=(hs0==suit && hs1==suit)?lo:0;
    int fcount=(hs0==suit)+(hs1==suit);
    int nutgap=fhi?popcount(((~bmasks[suit])&8191)>>(fhi-1)):13;
    int score=evaluate(cards,n+2),bscore=n==5?evaluate(board,n):0;
    unsigned mask=bmask|(1u<<(hi-2))|(1u<<(lo-2));
    int maxmult=0,npairs=0,ntrips=0,maxsuit=0;
    for (int count:branks) { maxmult=std::max(maxmult,count);npairs+=count==2;ntrips+=count==3; }
    for (int count:bsuits) maxsuit=std::max(maxsuit,count);
    return {
        n, score>>20,(score>>16)&15,(score>>12)&15,(score>>8)&15,(score>>4)&15,score&15,
        hi,lo,int(hi==lo),int(hs0==hs1),
        sorted[0],sorted[1],sorted[2],sorted[3],sorted[4],
        popcount(bmask),maxmult,npairs,ntrips,maxsuit,tsuits[suit],fcount,fhi,flo,nutgap,
        t.window_count[mask],t.window_count[bmask],t.straight_outs[mask],t.window_high[mask],
        int(hi>sorted[0])+int(lo>sorted[0]),branks[hi-2],branks[lo-2],
        bscore>>20,(bscore>>16)&15,(bscore>>12)&15,(bscore>>8)&15,(bscore>>4)&15,bscore&15,
        hi==lo?std::max(bsuits[hs0],bsuits[hs1]):bsuits[hs0],
        hi==lo?std::min(bsuits[hs0],bsuits[hs1]):bsuits[hs1],
        sorted[0]-sorted[n-1],hi-lo,int(bool(bmask&4096))
    };
}

// Canonical 13x13 classes: AA,AKs,...,A2s,AKo,KK,KQs,...,22.
inline int class_id(int a,int b) {
    int r0=a/4,r1=b/4;
    int high=std::max(r0,r1),low=std::min(r0,r1);
    if (high==low) return (12-high)*13+(12-high);
    return (a&3)==(b&3) ? (12-high)*13+(12-low) : (12-low)*13+(12-high);
}

} // namespace holdem
