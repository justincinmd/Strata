// The Engine (engine.hpp) for Qwen3.8-Flash-Next: its layers, its embeddings and its expert arena; what both models'
// engines share is engine_common.cu's (engine_parts.hpp).
#include <cstdlib>
#include "engine.hpp"
#include "engine_parts.hpp"
#include "env.hpp"
#include "peer_probe.hpp"

#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#if FNX_HIP
#include "hip/peer.h"
#include "hip/rccl.h"
#include "hip/evict_watch.h"
#endif

namespace fnx {


Engine::Engine(Model& m, int max_ctx, int lanes) : m_(m), max_ctx_(max_ctx) {
    tk_start();   // generator: the kernel variants of this run (FNX_TK_DIR, FNX_TK_<FAMILY>); a wrong name fails here
    // the 27B build's settings: this Flash-Next engine has none of what they set, so one given is a mistake, not a
    // run of the default
    for (const char* k : {"FNX_PF_SKIP", "FNX_PF_SKIP_CONFIRM", "FNX_PF_SCORE_MIB", "FNX_PF_WINDOWS", "FNX_PF_SCORES",
                          "FNX_KV_PAGES", "FNX_KV_RESERVE_MIB"})
        if (std::getenv(k)) throw std::runtime_error(std::string(k) + " is the 27B's: Flash-Next has nothing it sets");
    // the rotary's scaling (choose_rope, src/rope.hpp): the model's own YaRN, FNX_ROPE_SCALE / FNX_YARN_ORIG_CTX, or
    // both when they agree; set once here, before any graph capture. Positions past what the rotary covers have angles
    // the model never saw, so a context past that is refused here rather than served
    rope_.base = m_.rope_base;
    long long env_orig = 0;
    if (std::getenv("FNX_YARN_ORIG_CTX")) {
        env_orig = env_int("FNX_YARN_ORIG_CTX", 0);
        if (env_orig <= 0 || env_orig > INT32_MAX)
            throw std::runtime_error("FNX_YARN_ORIG_CTX: want a trained context length from 1 to " + std::to_string(INT32_MAX) + ", got " +
                                     std::to_string(env_orig));
    }
    const RopeChoice rc = choose_rope(m_.yarn_factor, m_.yarn_orig, m_.ctx_train, std::getenv("FNX_ROPE_SCALE") ? rope_scale() : -1.0, env_orig);
    if (max_ctx > rc.limit)
        throw std::runtime_error("context " + std::to_string(max_ctx) + " is past the " + std::to_string(rc.limit) + " the rotary covers (" +
                                 (rc.S ? "trained " + std::to_string(rc.orig) + " x YaRN " + std::to_string(rc.S)
                                       : "the model's trained length; FNX_ROPE_SCALE=S (--rope-scale S) covers trained x S") +
                                 ")");
    if (rc.S) {
        rope_ = yarn_rope(m_.rope_base, rc.orig, rc.S, NROT);
        std::fprintf(stderr, "fnx: rope scaling YaRN x%.4f%s: n_ctx_orig %lld, pairs %.0f..%.0f of %d blended, mscale %.4f\n", rc.S,
                     rc.from_model ? " (the model's own)" : "", rc.orig, rope_.c_low, rope_.c_high, NROT / 2, rope_.mscale);
        if (!rc.from_model && max_ctx <= m_.ctx_train)
            std::fprintf(stderr, "fnx: note: context %d is within the trained %d: the scaling changes every position's rotary "
                         "and no length needs it\n", max_ctx, m_.ctx_train);
    }
    attn_i8_set_default(m_.g.iq_experts());   // the int8 prompt attention: on for the IQ3_S-type files, off for the others
    gdn_tc_set_default(!m_.g.iq_experts());   // the tensor-core GDN scan: off for the IQ3_S-type files (NVIDIA builds only compile it in)
    hc1_rows();   // FNX_HC1 checked now, not at the first hc read (inside a graph capture)
    const int extra_pages = (std::max(lanes, 1) - 1) * lane_block_pages();
    // the paged K/V pool: pages for the whole context (one more for the block a position past the last one needs)
    // and the extra lanes' blocks. Int8 rows: the context's pages as many as fit in their f16 bytes (the smallest ratio
    // of the cards': every card has the same pages)
    int ctx_pages = max_ctx / KVP_POS + 2;
    if (kv_q8()) {
        double ratio = 1e9;
        for (int cc = 0; cc < NCARD; ++cc) ratio = std::min(ratio, (double) kvp_page(cc, false) / kvp_page(cc, true));
        ctx_pages = (int) (ctx_pages * ratio);
    }
    int n_pages = ctx_pages + extra_pages;
    // prompt-path memory per card in MiB, measured (FNX_PF_TIMING "prefill buffers on card", the same for both models on 2 cards), and what must stay free beside the K/V pool: the decode buffers,
    // graphs and cuBLAS workspaces (UD-Q4_K_XL on 2 x 48 GB: 6.5 GB are free after the weights; 8192 x 2 fits but leaves 0.8 GB, 4096 x 4 leaves 2.3 GB)
    constexpr size_t PF_MARGIN_MIB = 2000;
    // the prompt path's chunk and pass. NVIDIA default 8192 rows and 4 chunks a pass (owner 11-03); a card whose free VRAM after the weights cannot hold that beside the K/V pool and
    // the rest (UD-Q4_K_XL on 2 x 48 GB keeps ~12 GB) gets the largest of 8192 x 4, 8192 x 2, 4096 x 4, 4096 x 2 that fits. FNX_PF_ROWS / FNX_PF_PASS set by hand are used as given.
    {
        long long rows = 4096;
        pf_pass_pick_ = 0;
#if !FNX_HIP
        rows = 8192;
        if (!std::getenv("FNX_PF_ROWS") && !std::getenv("FNX_PF_PASS")) {
            size_t free_min = SIZE_MAX, kv_max = 0;
            for (int c = 0; c < NCARD; ++c) {
                ck(cudaSetDevice(m.cards[c].dev), "cudaSetDevice");
                size_t fr = 0, tot = 0;
                ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
                free_min = std::min(free_min, fr);
                kv_max = std::max(kv_max, (size_t) n_pages * kvp_page(c) * sizeof(uint16_t));
            }
            const size_t margin = kv_max + (PF_MARGIN_MIB << 20);
            struct Cfg { int rows, pass; size_t mib; };
            const Cfg cfgs[] = {{8192, 4, 4148}, {8192, 2, 3028}, {4096, 4, 1673}, {4096, 2, 1112}};
            Cfg pick = cfgs[3];
            for (const Cfg& cf : cfgs)
                if ((cf.mib << 20) + margin <= free_min) {
                    pick = cf;
                    break;
                }
            rows = pick.rows;
            pf_pass_pick_ = pick.pass;
            if (pick.rows != 8192 || pick.pass != 4)
                std::fprintf(stderr, "fnx: prompt path: %d rows, %d chunks a pass (the cards hold %zu MiB free after the weights, the K/V pool %zu MiB)\n", pick.rows, pick.pass,
                             free_min >> 20, kv_max >> 20);
        }
#endif
        const long long pf_rows = env_int("FNX_PF_ROWS", rows);
        if (pf_rows < MAXT || pf_rows > 8192 || pf_rows % MAXT)
            throw std::runtime_error("FNX_PF_ROWS: a multiple of " + std::to_string(MAXT) + " from " + std::to_string(MAXT) + " to 8192, got " + std::to_string(pf_rows));
        pf_c_ = (int) pf_rows;
    }
    pf_jobs_ = pf_c_ * TOPK / PF_JOB + NEXP;   // jobs of up to PF_JOB pairs per expert, at most
    set_prompt_rows(pf_c_);
    // FNX_PEER (AMD only; hip/peer_policy.hpp has the rules: 0 host exchange, 1 try the peer push, unset the build's default): every directed pair is probed at start
    // (kernel stores into the peer, a full read-back compare on both sides, a flag round trip); any failure leaves the host exchange for the whole run with one line.
    // FNX_PEER_PARTS: 1 the window sums, 2 the attention merge, 3 both (the default)
    bool peer_on = false;
    int peer_parts = 0;
#if FNX_HIP
    {
        std::vector<int> devs;
        for (int c = 0; c < NCARD; ++c) devs.push_back(m.cards[c].dev);
        const PeerDecision pd = peer_decide(std::getenv("FNX_PEER"), PEER_DEFAULT_ON, [&](std::string& why) { return peer_enable_and_probe(devs, why); });
        if (!pd.line.empty()) std::fprintf(stderr, "fnx: %s\n", pd.line.c_str());
        peer_on = pd.use;
        peer_parts = (int) env_int("FNX_PEER_PARTS", 3);
        if (peer_on) std::fprintf(stderr, "fnx: peer push: %s%s pushed into the peers' memory\n", peer_parts & 1 ? "the window sums " : "", peer_parts & 2 ? "and the attention merge" : "");
        {   // RCCL for the prompt sums: constant RCCL_DEFAULT_ON (off until the screens say otherwise), FNX_PF_RCCL=0/1 overrides
            const RcclDecision rd = rccl_decide(std::getenv("FNX_PF_RCCL"), peer_on, RCCL_DEFAULT_ON, [&](std::string& why) {
                std::shared_ptr<RcclState> st;
                if (!rccl_open(devs, st, why)) return false;
                rccl_ = st;
                return true;
            });
            if (!rd.line.empty()) std::fprintf(stderr, "fnx: %s\n", rd.line.c_str());
            rccl_on_ = rd.use;
        }
        // the prompt chunk sums through the peers' device memory (needs the peer push; RCCL, when it is in use, does the sums instead): FNX_PF_PEER=0 keeps the host exchange
        pf_peer_ = peer_on && !rccl_on_ && env_int("FNX_PF_PEER", 1) != 0;
        if (pf_peer_) std::fprintf(stderr, "fnx: peer push: the prompt chunk sums are pushed into the peers' device memory\n");
    }
    auto pxalloc = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (hipExtMallocWithFlags(&p, bytes, hipDeviceMallocFinegrained) != hipSuccess) throw std::runtime_error("FNX_PEER: fine-grained device allocation of " + std::to_string(bytes >> 10) + " KiB failed");
        ck(FNX_ZERO_DEV(p, bytes), "peer mailbox clear");
        own_dev(p);
        return p;
    };
#endif
    for (int c = 0; c < NCARD; ++c) {
        const CardW& w = m.cards[c];
        CardBuf* b = new CardBuf();
        b->dev = w.dev;
        ck(cudaSetDevice(w.dev), "cudaSetDevice");
        // FNX_HOSTRING: no peer access is ever enabled; the cards exchange through host memory
        ck(cudaStreamCreateWithFlags(&b->st, cudaStreamNonBlocking), "stream");
        own_stream(b->st);
        ck(cudaStreamCreateWithFlags(&b->st2, cudaStreamNonBlocking), "stream");
        own_stream(b->st2);
#if FNX_MALL_PF
        if (mall_pf_built()) {   // the cache prefetch's side stream and events
            // run mode (the default): a stream of a priority class of its own, so that no other stream of the card shares the persistent
            // prefetcher's hardware queue (kernels.hpp mall_pf_stream; FNX_MALL_PF_PRIO: 1 high (default), -1 low, 0 a default stream)
            const char* pmd = std::getenv("FNX_MALL_PF_MODE");
            const long long pprio = env_int("FNX_MALL_PF_PRIO", 1);
            if (pprio < -1 || pprio > 1) throw std::runtime_error("FNX_MALL_PF_PRIO: -1, 0 or 1");
            b->st3 = (cudaStream_t) mall_pf_stream(!pmd || std::string(pmd) == "run" ? (int) pprio : 0, &b->pf_own);
            own_stream(b->st3);
            ck(cudaEventCreateWithFlags(&b->pf_fork, cudaEventDisableTiming), "event");
            own_event(b->pf_fork);
            ck(cudaEventCreateWithFlags(&b->pf_join, cudaEventDisableTiming), "event");
            own_event(b->pf_join);
            b->pf_sink = dalloc<unsigned>(16);
            b->pf_table = dalloc<PfSet>((size_t) NL * 4);   // run: a set an event, up to 4 events a layer
            b->pf_words = dalloc<unsigned>(MALL_PF_WORDS);
        }
#endif
        ck(cudaEventCreateWithFlags(&b->fork, cudaEventDisableTiming), "event");
        own_event(b->fork);
        ck(cudaEventCreateWithFlags(&b->join, cudaEventDisableTiming), "event");
        own_event(b->join);
        ck(cudaEventCreate(&b->ev0), "event");
        own_event(b->ev0);
        ck(cudaEventCreate(&b->ev1), "event");
        own_event(b->ev1);
        const int P = MAXT * TOPK, ffs = w.ff1 - w.ff0;
        b->R = dalloc<float>((size_t) MAXT * D);
        b->mixed = dalloc<float>((size_t) MAXT * N);
        b->bo_a = dalloc<float>((size_t) MAXT * N);
        b->bo_f = dalloc<float>((size_t) MAXT * N);
        b->inj_a = dalloc<float>(MAXT * HC);
        b->inj_f = dalloc<float>(MAXT * HC);
        b->hcs = dalloc<float>(hc_scratch_floats(MAXT));
        if (const int g = hc_prepare(b->hcs); g < hc_blocks())
            std::fprintf(stderr, "fnx: card %d: %d SMs: the hc read runs its %d blocks of work on %d\n", c, g, hc_blocks(), g);
        if (std::getenv("FNX_HC_BAR_WRAP")) {
            const long long n = env_int("FNX_HC_BAR_WRAP", 0);
            if (n <= 0 || n > 0xffffffffLL) throw std::runtime_error("FNX_HC_BAR_WRAP: want a barrier count > 0, got " + std::to_string(n));
            hc_preset_barrier(b->hcs, (unsigned) n);
        }
        b->xq = dalloc<uint8_t>((size_t) P * q8_1_bytes(N));
        b->qkvz = dalloc<float>((size_t) MAXT * (w.conv_ch() + w.nv() * HS));
        b->ab = dalloc<float>((size_t) MAXT * 2 * w.nv());
        b->gdn_scratch = dalloc<float>(gdn_scratch_floats(w.nv()));
        b->qg = dalloc<float>((size_t) MAXT * QG_W);
        b->k = dalloc<float>((size_t) MAXT * KV_W);
        b->v = dalloc<float>((size_t) MAXT * KV_W);
        b->att = dalloc<float>((size_t) MAXT * w.nq * HD);
        b->att_part = dalloc<float>(attn_scratch_floats(w.nq));
        b->logits = dalloc<float>((size_t) MAXT * (NEXP + 1));
        b->topw = dalloc<float>(P);
        b->topi = dalloc<int>(P);
        b->njobs = dalloc<int>(2);
        b->jobs_gu = dalloc<Job>(P + 1);
        b->jobs_dn = dalloc<Job>(P + 1);
        for (int T = 1; T <= MAXT; ++T) {
            Job j{};
            j.w = 0;
            j.n = T;
            for (int t = 0; t < T; ++t) { j.y[t] = t; j.out[t] = t; }
            b->dense[T] = dalloc<Job>(1);
            ck(cudaMemcpy(b->dense[T], &j, sizeof j, cudaMemcpyHostToDevice), "dense job");
        }
        if (w.ec.n_slots) b->cache_list = dalloc<int>(1 + P);
        b->gu = dalloc<float>((size_t) (P + MAXT) * 2 * ffs);
        b->part = dalloc<float>((size_t) (P + MAXT) * N);
        b->sh_gu = dalloc<float>((size_t) MAXT * 2 * ffs);
        b->sh_hq = dalloc<uint8_t>((size_t) MAXT * q8_1_bytes(ffs));
        if (!std::getenv("FNX_NO_TWIN")) {
            // the split twins: xq's rows of n <= N columns sit at r q8_1_bytes(n), so r n < P N
            q8_twin(b->xq, (size_t) P * q8_1_bytes(N), dalloc<int8_t>((size_t) P * N), dalloc<float>((size_t) P * N / 32));
            q8_twin(b->sh_hq, (size_t) MAXT * q8_1_bytes(ffs), dalloc<int8_t>((size_t) MAXT * ffs), dalloc<float>((size_t) MAXT * ffs / 32));
            for (const void* q : {(const void*) b->xq, (const void*) b->sh_hq}) own([q] { q8_untwin(q); });
        }
        b->ple_emb = dalloc<float>((size_t) MAXT * N);
        b->ple_key = dalloc<float>((size_t) 3 * MAXT * D);   // key, then room for gated and normalized rows
        b->ple_val = dalloc<float>((size_t) MAXT * N);
        b->head_logits = dalloc<float>((size_t) MAXT * (w.v1 - w.v0));
        b->best_v = dalloc<float>(MAXT);
        b->best_i = dalloc<int>(MAXT);
        b->am_keys = dalloc<unsigned long long>(MAXT);
        if (c == 0 && env_flag("FNX_PREFETCH_PROBE")) {
            b->probe_logits = dalloc<float>((size_t) MAXT * NEXP);
            b->probe_guess = dalloc<int>((size_t) NL * MAXT * TOPK);
            b->probe_cnt = dalloc<unsigned long long>(7);
        }
        if (c == 0 && env_flag("FNX_ROUTE_COUNT")) {
            b->route_cnt = dalloc<unsigned>((size_t) 2 * NL * NEXP);
            ck(cudaMemset(b->route_cnt, 0, (size_t) 2 * NL * NEXP * sizeof(unsigned)), "route counts");
        }
        if (const long long tr = env_int("FNX_ROUTE_TRACE", 0); c == 0 && tr > 0) {
            b->route_trace_cap = (unsigned long long) tr * 1000000ull;
            void* hp = nullptr;   // in mapped host memory: no VRAM for it
            ck(cudaHostAlloc(&hp, b->route_trace_cap * sizeof(unsigned), cudaHostAllocMapped), "route trace");
            own_pinned(hp);
            b->route_trace = (unsigned*) hp;
            b->route_trace_len = dalloc<unsigned long long>(1);
            ck(cudaMemset(b->route_trace_len, 0, sizeof(unsigned long long)), "route trace");
        }
        b->S.assign(NL, nullptr);
        b->chist.assign(NL, nullptr);
        b->snap.assign(NL, nullptr);
        b->hsnap.assign(NL, nullptr);
        b->kc.assign(NL, nullptr);
        b->vc.assign(NL, nullptr);
        b->qsa_ring.assign(NL, nullptr);
        b->qsa_pooled.assign(NL, nullptr);
        b->qsa_y = dalloc<float>((size_t) MAXT * qsa_rows());
        b->qsa_q = dalloc<float>((size_t) MAXT * 512);
        b->qsa_scores = dalloc<float>((size_t) MAXT * (max_ctx / 4));
        b->qsa_cells = dalloc<int>((size_t) MAXT * qsa_maxsel());
        b->qsa_ncells = dalloc<int>(MAXT);
        const size_t snap_floats = (size_t) (MAXT - 1) * w.nv() * HS * HS;
        int n_rec = 0;
        for (int l = 0; l < NL; ++l) n_rec += is_attn(l) ? 0 : 1;
        b->snap_block = dalloc<float>(snap_floats * n_rec);
        b->snap_block_bytes = snap_floats * n_rec * sizeof(float);
        for (int l = 0, k = 0; l < NL; ++l) {
            if (is_attn(l)) {
                b->qsa_ring[l] = dalloc<uint16_t>((size_t) qsa_ring_rows() * 128);
            } else {
                b->S[l] = dalloc<float>((size_t) w.nv() * HS * HS);
                b->chist[l] = dalloc<float>((size_t) 2 * 3 * w.conv_ch());
                b->snap[l] = b->snap_block + snap_floats * k++;
                b->hsnap[l] = dalloc<float>((size_t) MAXT * 3 * w.conv_ch());
            }
        }
        b->ple_hist = dalloc<float>((size_t) 9 * D);
        b->ple_hsnap = dalloc<float>((size_t) MAXT * 9 * D);
        b->ctl = dalloc<int>(2);
        b->ring.card = c;
        // sized for fp32 words (two per pair); fp16 pairs use the first half of each slot row's room
        b->ring.in = hxalloc<unsigned long long>(ring_in_words(MAXT));
        b->ring.ag = hxalloc<unsigned long long>(ring_ag_words(MAXT));
        b->ring.f32 = ring_f32();
        b->ring.err = dalloc<unsigned>(ERR_WORDS);   // the error word and the first time-out's record behind it
        diag_load(c, *b);
        b->att_ring.card = c;
#if FNX_HIP
        if (peer_on && (peer_parts & 1)) b->ring.pk = pxalloc((size_t) 2 * NCARD * MAXT * (N / 2) * sizeof(unsigned long long));
        b->att_ring.in = peer_on && (peer_parts & 2) ? (unsigned long long*) pxalloc(att_ring_words() * sizeof(unsigned long long))
                                                     : hxalloc<unsigned long long>(att_ring_words());
#else
        b->att_ring.in = hxalloc<unsigned long long>(att_ring_words());
#endif
        b->att_ring.err = b->ring.err;
        cb_.push_back(b);
    }
    // every link carries a probe's stores before the ring's (src/peer_probe.hpp): one that drops them stops the engine
    // here, named, instead of in the first window's ring; a slow one is said and the engine goes on
    std::fprintf(stderr, "fnx: HOSTRING build: the cards exchange through page-locked host memory; peer access is never enabled, "
                         "no peer probe, IOMMU and BAR1 checks skipped\n");
    mall_pf_plan();   // the cache prefetch's sets (nothing without the build option)
    for (int c = 0; c < NCARD; ++c) {
        cb_[c]->ring.next_in = cb_[(c + 1) % NCARD]->ring.in;
        cb_[c]->ring.next_ag = cb_[(c + 1) % NCARD]->ring.ag;
        cb_[c]->att_ring.next_in = cb_[(c + 1) % NCARD]->att_ring.in;
        for (int d = 0; d < NCARD; ++d) {
            cb_[c]->ring.peer_in[d] = cb_[d]->ring.in;
            cb_[c]->att_ring.peer_in[d] = cb_[d]->att_ring.in;
        }
        // one hop for the window sums and the attention merge on 3 or more cards (FNX_ONEHOP=0: the rings); the mailboxes live in the ring's buffers
        // FNX_ONEHOP=2 (screen): the one-hop sum on two cards too (each card adds the two fp16-rounded addends in fp32: one host round trip instead of two; the sums differ in the last bits)
        const bool onehop = (NCARD > 2 || peer_on || env_int("FNX_ONEHOP", 1) == 2) && env_int("FNX_ONEHOP", 1) != 0 && !cb_[c]->ring.f32;
        cb_[c]->ring.onehop = cb_[c]->att_ring.onehop = onehop;
        for (int d = 0; d < NCARD; ++d) cb_[c]->ring.peer_pk[d] = cb_[d]->ring.pk;
        cb_[c]->ring.peer = onehop && cb_[c]->ring.pk != nullptr;
    }
    if (NCARD > 2 && env_int("FNX_ONEHOP", 1) != 0 && cb_[0]->ring.f32)
        std::fprintf(stderr, "fnx: FNX_RING_F32: the one-hop sum carries fp16 words only, the rings stay\n");
    // the prompt path's buffers now, so the arena below is sized from what is left after them
    prefill_alloc();
    // FNX_ARENA: the pool's own pages are those, or as many as every card's free VRAM holds beside the arena pool's
    // slots and FNX_ARENA_RESERVE_MIB, or FNX_ARENA_BASE_PAGES; the slots follow the own pages in the same allocation,
    // and the pool grows into them (lent coldest first) up to where every card keeps 2 MAXT TOPK slots. The buffers
    // above and the draft layer (an Mtp is made before its Engine) are already allocated, so the reserve is only what
    // comes later: the window graphs captured at first use (with batched drafts a verify graph per row total), cuBLAS's
    // workspaces and the lanes' page tables. Default 512: they took ~270 MiB a card on three cards and ~340 on two,
    // through one to six lanes, batched drafts and a 114k prompt
    if (m.arena) {
        const long long reserve_mib = env_int("FNX_ARENA_RESERVE_MIB", 512);
        if (reserve_mib < 0) throw std::runtime_error("FNX_ARENA_RESERVE_MIB: want MiB >= 0, got " + std::to_string(reserve_mib));
        const size_t reserve = (size_t) reserve_mib << 20;
        // a card with less free VRAM than the slots, the reserve and the smallest own pool (two cards, another process
        // on it, a leak): the coldest experts that many slots' worth go to RAM, placed again, and the engine goes on.
        // Too few slots left for a window is still an error (Model::place_experts says so)
        for (int round = 0;; ++round) {
            int more = 0;
            size_t short_mib = 0;
            for (int c = 0; c < NCARD; ++c) {
                const ExpertCache& ec = m.cards[c].ec;
                if (ec.arena_pool < 0) throw std::runtime_error("FNX_ARENA: card " + std::to_string(c) + " has no arena pool");
                ck(cudaSetDevice(m.cards[c].dev), "cudaSetDevice");
                size_t fr = 0, tot = 0;
                ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
                const size_t sb = ec.slot_bytes[ec.arena_pool],
                             need = (size_t) ec.n[ec.arena_pool] * sb + reserve + (size_t) (extra_pages + 1) * kvp_page(c) * sizeof(uint16_t) +
                                    2 * pf_stage_bytes(c);
                if (fr < need) {   // 256 MiB more than short, so the next engine's few MiB of drift place nothing again
                    more = std::max(more, (int) ((need - fr + (256u << 20) + sb - 1) / sb));
                    short_mib = std::max(short_mib, (need - fr) >> 20);
                }
            }
            if (!more) break;
            if (round == 8) throw std::runtime_error("FNX_ARENA: VRAM still short after placing the experts again 8 times");
            // two 32 GB cards are short by design; on three or more the experts fit unless something else holds VRAM
            std::fprintf(stderr, "fnx: VRAM short by %zu MiB on a card%s: %d more of the coldest experts to RAM\n", short_mib,
                         NCARD >= 3 ? " (another process on it?)" : "", more);
            m.free_experts();
            m.ram_extra += more;
            m.place_experts();
        }
        prefill_stages();
        for (int c = 0; c < NCARD; ++c) {
            const ExpertCache& ec = m.cards[c].ec;
            if (ec.arena_pool < 0) throw std::runtime_error("FNX_ARENA: card " + std::to_string(c) + " has no arena pool");
            ck(cudaSetDevice(m.cards[c].dev), "cudaSetDevice");
            size_t fr = 0, tot = 0;
            ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
            const size_t slots = (size_t) ec.n[ec.arena_pool] * ec.slot_bytes[ec.arena_pool], page = kvp_page(c) * sizeof(uint16_t);
            const int fit = fr > slots + reserve ? (int) ((fr - slots - reserve) / page) : 0;
            if (fit <= extra_pages)
                throw std::runtime_error("FNX_ARENA: card " + std::to_string(c) + " has " + std::to_string(fr >> 20) + " MiB free: the arena's " +
                                         std::to_string(slots >> 20) + " MiB of slots, " + std::to_string(reserve >> 20) +
                                         " MiB in reserve (FNX_ARENA_RESERVE_MIB) and the lanes' blocks do not fit");
            n_pages = std::min(n_pages, fit);
        }
        if (std::getenv("FNX_ARENA_BASE_PAGES")) {
            const long long want = env_int("FNX_ARENA_BASE_PAGES", 0);
            if (want <= extra_pages || want > n_pages)
                throw std::runtime_error("FNX_ARENA_BASE_PAGES: want " + std::to_string(extra_pages + 1) + " to " + std::to_string(n_pages) +
                                         " (the lanes' blocks take " + std::to_string(extra_pages) + ")");
            n_pages = (int) want;
        }
        arena_base_ = arena_w_ = n_pages;
        arena_cap_ = INT32_MAX;
        for (int c = 0; c < NCARD; ++c) {
            const ExpertCache& ec = m.cards[c].ec;
            const int p = ec.arena_pool, keep = EXPERT_WINDOW_SLOTS;
            const size_t page = kvp_page(c) * sizeof(uint16_t);
            const size_t room = (size_t) (ec.n[p] - keep) * ec.slot_bytes[p] + (arena_slot_off(c) - (size_t) n_pages * page);
            arena_cap_ = std::min(arena_cap_, n_pages + (int) (room / page));
        }
    } else {
        prefill_stages();
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf* b = cb_[c];
        ck(cudaSetDevice(b->dev), "cudaSetDevice");
        // the K/V pool and lane 0's page table (pages taken as it grows); with an arena the pool is its start and
        // n_pages counts the pages it can grow to
        if (m.arena) {
            ExpertCache& ec = m.cards[c].ec;
            const int p = ec.arena_pool;
            b->n_pages = arena_cap_;
            char* region = (char*) dalloc<uint8_t>(arena_slot_off(c) + (size_t) ec.n[p] * ec.slot_bytes[p]);
            b->pool = (uint16_t*) region;
            ec.pool[p] = region + arena_slot_off(c);
            m.cards[c].bytes += (size_t) ec.n[p] * ec.slot_bytes[p];
            arena_fill(c, b->st);
            size_t fr = 0, tot = 0;
            ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
            std::fprintf(stderr, "fnx: arena: card %d: %zu MiB free after its arena\n", c, fr >> 20);
        } else {
            b->n_pages = n_pages;
            b->pool = dalloc<uint16_t>((size_t) b->n_pages * kvp_page(c));
        }
        b->pt = dalloc<int>((size_t) b->n_pages);
        b->pg.pt = b->pt;
        b->pg.page = kvp_page(c);
        b->pg.rows = kvp_rows(c);
        b->pg.q8 = kv_q8();
        const size_t kv_halves = kvp_rows_halves(c, kv_q8());
        for (int l = 0, slot = 0; l < NL; ++l)
            if (is_attn(l)) {
                b->kc[l] = b->pool + (size_t) slot * kvp_slot(c);
                b->vc[l] = b->kc[l] + kv_halves;
                b->qsa_pooled[l] = b->kc[l] + 2 * kv_halves;
                ++slot;
            }
        {   // lane 0: the buffers above
            LaneDev L;
            L.pt = b->pt;
            L.S = b->S;
            L.chist = b->chist;
            L.ple_hist = b->ple_hist;
            L.ring = b->qsa_ring;
            L.ctl = dalloc<int>(2);
            b->lanes.push_back(L);
        }
    }
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "init sync"); }
    lanes_.emplace_back();
    for (int i = m.arena ? arena_base_ : cb_[0]->n_pages; i-- > 0;) free_pages_.push_back(i);   // taken from the back: low pages first
    if (m.arena) {
        std::string per;
        for (int c = 0; c < NCARD; ++c) {
            const ExpertCache& ec = m.cards[c].ec;
            const int p = ec.arena_pool;
            char buf[96];
            std::snprintf(buf, sizeof buf, "%s%d x %.2f MiB (%.1f GiB)", c ? ", " : "", ec.n[p], ec.slot_bytes[p] / 1048576.0,
                          (double) ec.n[p] * ec.slot_bytes[p] / (1ull << 30));
            per += buf;
        }
        std::fprintf(stderr, "fnx: arena: the K/V pool's own %d pages (%d positions), up to %d (%d positions) by lending the slots of %s\n",
                     arena_base_, arena_base_ * KVP_POS, arena_cap_, arena_cap_ * KVP_POS, per.c_str());
    }
    if (std::getenv("FNX_RING_TAG_START")) {
        const long long v = env_int("FNX_RING_TAG_START", 0);
        if (v < 1 || v >= (long long) TAG_END)
            throw std::runtime_error("FNX_RING_TAG_START: want 1.." + std::to_string(TAG_END - 1) + ", got " + std::to_string(v));
        tag_max_ = (unsigned) v;
    }
}

// card c's clocks at load, and the ring time-out. clock64 (every device wait's clock) counts the core clock's cycles, whose rate moves
// with the clock; the steady counter has a constant rate (hipDeviceAttributeWallClockRate). A short kernel measures one against the other, so the
// log says what the time-out is in milliseconds at this moment's clock (an idle card's clock; a time-out's record measures it again).
// FNX_RING_TIMEOUT_CYCLES sets the decode ring sums' time-out in cycles (exact); FNX_RING_TIMEOUT_MS in milliseconds by the rate measured here
// (2,000 MHz when it could not be measured). Unset: RING_TIMEOUT_CYCLES, as before.
void Engine::diag_load(int c, CardBuf& b) {
    double khz = 0;
#if FNX_HIP
    int v = 0;
    if (hipDeviceGetAttribute(&v, hipDeviceAttributeWallClockRate, b.dev) == hipSuccess && v > 0) khz = v;
    cudaGetLastError();
#endif
    diag_wall_khz_[c] = khz;
    diag_stall(-1, b, false);   // FNX_DIAG_STALL read here: a bad value fails the load, not a launch thread
    diag_clock(b.ring.err, b.st);
    ck(cudaStreamSynchronize(b.st), "diag clock");
    unsigned w[ERR_WORDS] = {};
    ck(cudaMemcpy(w, b.ring.err, sizeof w, cudaMemcpyDeviceToHost), "diag clock");
    const double mhz = khz > 0 && w[DG_CAL0_WALL] > 0 ? (double) w[DG_CAL0_CYC] * khz / w[DG_CAL0_WALL] / 1000.0 : 0.0;
    diag_clock_mhz_[c] = mhz;
    static const long long cycles = env_int("FNX_RING_TIMEOUT_CYCLES", 0);
    static const double ms = env_num("FNX_RING_TIMEOUT_MS", 0);
    if (cycles < 0 || ms < 0 || ms > 600000 || (cycles > 0 && ms > 0))
        throw std::runtime_error("FNX_RING_TIMEOUT_CYCLES / FNX_RING_TIMEOUT_MS: want one of them, cycles > 0 or 0 < ms <= 600000");
    if (cycles > 0) b.ring.timeout = cycles;
    else if (ms > 0) b.ring.timeout = (long long) (ms * 1000.0 * (mhz > 0 ? mhz : 2000.0));
    diag_ring_timeout_[c] = b.ring.timeout;
    // FNX_WAIT_EVICT (unset: on in a FAST build): a wait that reaches its time-out begins again while the host's eviction watch saw the driver stop this process's
    // queues a moment ago (diag.cuh diag_wait_on). 0: off, a wait fails at its time-out as before the watch
    // (FNX_WAIT_JUMP_CYCLES=0 is read as the same). Off too when the watch cannot run (no KFD files for this process)
    static const bool ext = env_int("FNX_WAIT_EVICT", FAST) != 0 && env_int("FNX_WAIT_JUMP_CYCLES", 1) != 0;
    // the watch (a host thread that reads KFD's evicted_ms files) starts only when the waits use it: a build without STRATA_SPLIT_FAST starts none
    const unsigned long long* word = ext ? evict_watch_word() : nullptr;
    const unsigned long long* watch = word;
    diag_ext_on_ = watch != nullptr;
    const unsigned watch_w[2] = {(unsigned) (uintptr_t) watch, (unsigned) ((unsigned long long) (uintptr_t) watch >> 32)};
    ck(cudaMemcpy(b.ring.err + DG_WATCH_LO, watch_w, sizeof watch_w, cudaMemcpyHostToDevice), "watch word");
#if FNX_HIP
    if (watch) fnx_evict_watch.store(watch, std::memory_order_relaxed);   // the prompt path's flag waits (hip/cuda.h)
#endif
    if (!diag_ext_h_) diag_ext_h_ = halloc<unsigned>((size_t) NCARD);
    diag_ext_h_[c] = 0;
    diag_ext_seen_[c] = 0;
    if (c == 0) {
        if (watch)
            std::fprintf(stderr, "fnx: diag: waits ride driver evictions: a wait at its time-out begins again when KFD's evicted_ms of this process rose at most "
                                 "%u ms plus about the time-out before (%d files, sampled every 10 ms while the engine works); with no rise it gets %lld cycles "
                                 "more, once, then fails; a wait extended by %lld cycles fails at its next time-out (FNX_WAIT_EVICT=0: off)\n",
                         WAIT_EVICT_WINDOW_MS, evict_watch_stats().files, WAIT_GRACE_CYCLES, WAIT_EXT_CAP);
        else if (!ext)
            std::fprintf(stderr, "fnx: diag: FNX_WAIT_EVICT off: a wait fails at its time-out (a driver eviction can fail a window); no eviction watch\n");
        else std::fprintf(stderr, "fnx: diag: no eviction watch (no KFD evicted_ms file of this process, or no host page): a wait fails at its time-out\n");
    }
    const bool set = cycles > 0 || ms > 0;
    if (mhz > 0)
        std::fprintf(stderr, "fnx: diag: card %d: clock64 runs at %.0f MHz now (steady counter %.0f kHz): the ring time-out of %lld cycles%s is %.0f ms at this rate\n",
                     c, mhz, khz, b.ring.timeout, set ? " (set by the environment)" : "", (double) b.ring.timeout / (mhz * 1000.0));
    else
        std::fprintf(stderr, "fnx: diag: card %d: the clock64 rate could not be measured (steady counter %.0f kHz, %u cycles in %u ticks): the ring time-out is %lld cycles%s\n",
                     c, khz, w[DG_CAL0_CYC], w[DG_CAL0_WALL], b.ring.timeout, set ? " (set by the environment)" : "");
}

void Engine::clear_rings() {
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "ring clear sync"); }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        ck(cudaSetDevice(b.dev), "set");
        ck(cudaMemset(b.ring.in, 0, ring_in_words(MAXT) * sizeof(unsigned long long)), "ring clear");
        ck(cudaMemset(b.ring.ag, 0, ring_ag_words(MAXT) * sizeof(unsigned long long)), "ring clear");
        ck(cudaMemset(b.att_ring.in, 0, att_ring_words() * sizeof(unsigned long long)), "ring clear");
        if (!pf_.empty()) pf_clear_ring(c);
    }
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "ring clear sync"); }
    std::fprintf(stderr, "fnx: ring tags reached %u: the rings' buffers cleared, tags start again\n", tag_max_);
    tag_max_ = 1;
}

void Engine::tags_room(unsigned n) {
    if (tag_max_ + n + 1u >= TAG_END) clear_rings();
}

// an arena's K/V pool grows by at least this many pages a lend (64: ~12k positions f16), so lends stay rare
constexpr int ARENA_STEP = 64;

size_t Engine::arena_slot_off(int c) const {
    return ((size_t) arena_base_ * kvp_page(c) * sizeof(uint16_t) + 255) & ~(size_t) 255;
}

// the arena pool as loaded: the slot tables back to the model's first assignment (an earlier engine in this process
// lent, reclaimed and replaced slots), every expert of the pool's layers at its host copy, then each slot's expert
// copied in from it
void Engine::arena_fill(int c, cudaStream_t st) {
    ExpertCache& ec = m_.cards[c].ec;
    const CardW& w = m_.cards[c];
    const int p = ec.arena_pool;
    // only the arena pool's part: the other pools' slots keep what an earlier engine copied into them, and their
    // tables say so
    const int s0 = ec.s0[p], n = ec.n[p];
    ck(cudaMemcpy(ec.owner + s0, w.ec_owner0.data() + s0, (size_t) n * sizeof(int), cudaMemcpyHostToDevice), "arena: owners");
    std::vector<int> slot_of((size_t) NL * NEXP);
    ck(cudaMemcpy(slot_of.data(), ec.slot_of, slot_of.size() * sizeof(int), cudaMemcpyDeviceToHost), "arena: slots");
    for (int l = 0; l < NL; ++l)
        if (ec.pool_of[l] == p)
            for (int e = 0; e < NEXP; ++e) slot_of[(size_t) l * NEXP + e] = w.ec_slot0[(size_t) l * NEXP + e];
    ck(cudaMemcpy(ec.slot_of, slot_of.data(), slot_of.size() * sizeof(int), cudaMemcpyHostToDevice), "arena: slots");
    ck(cudaMemset(ec.ref + s0, 0, (size_t) n), "arena: refs");
    ck(cudaMemcpy(ec.hand + p, &ec.s0[p], sizeof(int), cudaMemcpyHostToDevice), "arena: hand");
    ck(cudaMemset(ec.lent + p, 0, sizeof(int)), "arena: lent");
    const std::vector<int> owner(w.ec_owner0.begin() + s0, w.ec_owner0.begin() + s0 + n);
    std::vector<const char*> epg((size_t) NL * NEXP), epd(epg.size());
    ck(cudaMemcpy(epg.data(), ec.ep_gu, epg.size() * sizeof(char*), cudaMemcpyDeviceToHost), "arena: tables");
    ck(cudaMemcpy(epd.data(), ec.ep_dn, epd.size() * sizeof(char*), cudaMemcpyDeviceToHost), "arena: tables");
    for (int l = 0; l < NL; ++l)
        if (ec.pool_of[l] == p)
            for (int e = 0; e < NEXP; ++e) {
                const size_t i = (size_t) l * NEXP + e;
                epg[i] = w.ec_host_gu[i];
                epd[i] = w.ec_host_dn[i];
            }
    // the slots' experts in one kernel reading their host copies (a copy call each ran the cards one after another)
    std::vector<int> list{n};
    for (int s = 0; s < ec.n[p]; ++s) {
        const int i = owner[(size_t) s];
        char* at = ec.pool[p] + (size_t) s * ec.slot_bytes[p];
        epg[(size_t) i] = at;
        epd[(size_t) i] = at + ec.dn_off[p];
        list.push_back(i);
    }
    int* dlist = dalloc<int>(list.size());
    ck(cudaMemcpy(dlist, list.data(), list.size() * sizeof(int), cudaMemcpyHostToDevice), "arena: fill list");
    cache_copy(ec, p, dlist, st);
    ck(cudaMemcpyAsync(ec.ep_gu, epg.data(), epg.size() * sizeof(char*), cudaMemcpyHostToDevice, st), "arena: tables");
    // no wait here: the cards fill at once, and the constructor syncs every card before the engine is used (the tables
    // come from pageable vectors, copied out before cudaMemcpyAsync returns)
    ck(cudaMemcpyAsync(ec.ep_dn, epd.data(), epd.size() * sizeof(char*), cudaMemcpyHostToDevice, st), "arena: tables");
}

int Engine::arena_slots_for(int c, int w) const {
    const ExpertCache& ec = m_.cards[c].ec;
    const long need = (long) w * (long) (kvp_page(c) * sizeof(uint16_t)) - (long) arena_slot_off(c);
    if (need <= 0) return 0;
    const long sb = (long) ec.slot_bytes[ec.arena_pool];
    return (int) ((need + sb - 1) / sb);
}

std::vector<int> Engine::arena_lent() const {
    std::vector<int> v;
    if (arena_cap_)
        for (int c = 0; c < NCARD; ++c) v.push_back(arena_slots_for(c, arena_w_));
    return v;
}

// slots [to, from) of card c's arena pool, just given back empty: each takes one of the pool's experts not in a slot,
// the hottest first (the profile's score), copied in from its host copy (the card idle). Nothing lent then means every
// expert is in a slot, which the plan's early return relies on (cache_plan, moe.cuh)
void Engine::arena_refill(int c, int from, int to) {
    ExpertCache& ec = m_.cards[c].ec;
    const CardW& w = m_.cards[c];
    const int p = ec.arena_pool, s0 = ec.s0[p];
    std::vector<int> slot_of((size_t) NL * NEXP);
    ck(cudaMemcpy(slot_of.data(), ec.slot_of, slot_of.size() * sizeof(int), cudaMemcpyDeviceToHost), "arena: slots");
    std::vector<int> out;   // the pool's experts not in a slot
    for (int l = 0; l < NL; ++l)
        if (ec.pool_of[l] == p)
            for (int e = 0; e < NEXP; ++e)
                if (slot_of[(size_t) l * NEXP + e] < 0) out.push_back(l * NEXP + e);
    std::stable_sort(out.begin(), out.end(), [&](int a, int b) { return m_.score[(size_t) a] > m_.score[(size_t) b]; });
    if (ec.arena_full && to == 0 && (int) out.size() > from)
        throw std::runtime_error("arena: " + std::to_string(out.size()) + " experts out of a slot, " + std::to_string(from) + " slots to take them");
    const int n = std::min((int) out.size(), from - to);
    std::vector<int> owner((size_t) (from - to), -1);
    std::vector<const char*> epg((size_t) NL * NEXP), epd(epg.size());
    ck(cudaMemcpy(epg.data(), ec.ep_gu, epg.size() * sizeof(char*), cudaMemcpyDeviceToHost), "arena: tables");
    ck(cudaMemcpy(epd.data(), ec.ep_dn, epd.size() * sizeof(char*), cudaMemcpyDeviceToHost), "arena: tables");
    for (int k = 0; k < n; ++k) {
        const int i = out[(size_t) k], s = to + k;
        const LayerW& L = w.L[(size_t) (i / NEXP)];
        char* at = ec.pool[p] + (size_t) s * ec.slot_bytes[p];
        ck(cudaMemcpy(at, w.ec_host_gu[(size_t) i], L.gu_e_stride, cudaMemcpyDefault), "arena: refill");
        ck(cudaMemcpy(at + ec.dn_off[p], w.ec_host_dn[(size_t) i], L.down_e_stride, cudaMemcpyDefault), "arena: refill");
        epg[(size_t) i] = at;
        epd[(size_t) i] = at + ec.dn_off[p];
        slot_of[(size_t) i] = s0 + s;
        owner[(size_t) k] = i;
    }
    ck(cudaMemcpy(ec.ep_gu, epg.data(), epg.size() * sizeof(char*), cudaMemcpyHostToDevice), "arena: tables");
    ck(cudaMemcpy(ec.ep_dn, epd.data(), epd.size() * sizeof(char*), cudaMemcpyHostToDevice), "arena: tables");
    ck(cudaMemcpy(ec.slot_of, slot_of.data(), slot_of.size() * sizeof(int), cudaMemcpyHostToDevice), "arena: slots");
    ck(cudaMemcpy(ec.owner + s0 + to, owner.data(), owner.size() * sizeof(int), cudaMemcpyHostToDevice), "arena: owners");
}

// lending: the slots' experts to RAM (their tables first), then K/V may write there; giving back: the pages are free,
// the slots come back and are filled again at once (arena_refill). Every card idle first: a window or a prompt pass in
// flight may read a slot being lent, or a page being given back
void Engine::arena_set(int w) {
    if (w == arena_w_) return;
    if (w < arena_base_ || w > arena_cap_) throw std::runtime_error("arena: top " + std::to_string(w) + " outside its range");
    const auto t0 = std::chrono::steady_clock::now();
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const ExpertCache& ec = m_.cards[c].ec;
        ck(cudaSetDevice(b.dev), "set");
        ck(cudaDeviceSynchronize(), "arena: cards idle");
        const int from = arena_slots_for(c, arena_w_), to = arena_slots_for(c, w);
        if (to > ec.n[ec.arena_pool] - EXPERT_WINDOW_SLOTS) throw std::runtime_error("arena: card " + std::to_string(c) + " would keep too few slots");
        arena_slots(ec, from, to, b.st);
        ck(cudaStreamSynchronize(b.st), "arena: slots");
        if (to < from) arena_refill(c, from, to);
    }
    if (w > arena_w_) {
        for (int pg = arena_w_; pg < w; ++pg) free_pages_.push_back(pg);
        // zeroed: they held expert weights, and the draft layer reads K/V it never wrote (batched windows skip it), which
        // as int8 rows and scales could be inf or NaN (the own pages only ever held finite K/V or zeros)
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            const size_t page = kvp_page(c) * sizeof(uint16_t);
            ck(cudaMemset((char*) cb_[c]->pool + (size_t) arena_w_ * page, 0, (size_t) (w - arena_w_) * page), "arena: zero");
        }
        arena_last_lend_ = std::chrono::steady_clock::now();
    } else {
        free_pages_.erase(std::remove_if(free_pages_.begin(), free_pages_.end(), [&](int pg) { return pg >= w; }), free_pages_.end());
    }
    std::sort(free_pages_.begin(), free_pages_.end(), std::greater<int>());
    const int old = arena_w_;
    arena_w_ = w;
    const std::vector<int> now = arena_lent();
    std::string per;
    double gib = 0;
    for (int c = 0; c < NCARD; ++c) {
        const ExpertCache& ec = m_.cards[c].ec;
        per += (c ? " / " : "") + std::to_string(now[(size_t) c]);
        gib = std::max(gib, (double) now[(size_t) c] * ec.slot_bytes[ec.arena_pool] / (1ull << 30));
    }
    std::fprintf(stderr, "fnx: arena: K/V pool %d -> %d pages (own %d, at most %d); slots lent %s (up to %.2f GiB of a card's experts in RAM), in %.1f ms\n",
                 old, w, arena_base_, arena_cap_, per.c_str(), gib,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

void Engine::arena_reclaim(double hold_s) {
    if (!arena_cap_ || arena_w_ == arena_base_) return;
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - arena_last_lend_).count() < hold_s) return;
    int top = arena_base_;
    for (const LaneHost& L : lanes_)
        for (int pg : L.pages) top = std::max(top, pg + 1);
    if (top < arena_w_) arena_set(top);
}

// FNX_ARENA_CHURN=1 (a test): before every window, ARENA_STEP more pages lent and the pool's top then given back to the
// highest page in use (those slots refilled), so lending and giving back run beside every kind of window
void Engine::arena_churn() {
    static const bool on = env_flag("FNX_ARENA_CHURN");
    if (!on || !arena_cap_) return;
    int top = arena_base_;
    for (const LaneHost& L : lanes_)
        for (int pg : L.pages) top = std::max(top, pg + 1);
    arena_set(std::min(arena_cap_, std::max(arena_w_, top) + ARENA_STEP));
    arena_set(top);
}

void Engine::ensure_pages(int n_pos) { ensure_pages_of(cur_, n_pos); }

void Engine::ensure_pages_of(int lane, int n_pos) {
    LaneHost& L = lanes_[(size_t) lane];
    const int need = n_pos / KVP_POS + 1;   // the page of position n_pos too: a block's pooled key can sit one page on
    if (need > cb_[0]->n_pages) throw Refused("context full: " + std::to_string(n_pos) + " positions");
    const size_t had = L.pages.size();
    // with an arena, slots lent for what the free pages lack: at least ARENA_STEP pages at a time
    if (const int short_by = need - (int) had - (int) free_pages_.size(); short_by > 0 && arena_w_ < arena_cap_)
        arena_set(std::min(arena_cap_, arena_w_ + std::max(short_by, ARENA_STEP)));
    if (const int short_by = need - (int) had - (int) free_pages_.size(); short_by > 0 && on_pool_short && !in_pool_short_ && pending_.empty()) {
        const int was = cur_;
        in_pool_short_ = true;
        try {
            on_pool_short(lane, short_by);
        } catch (...) {
            in_pool_short_ = false;
            throw;
        }
        in_pool_short_ = false;
        if (cur_ != was) throw std::runtime_error("on_pool_short left lane " + std::to_string(cur_) + " current, not " + std::to_string(was));
    }
    // all the pages or none: a lane holding pages its device page table lacks would write K/V through stale entries
    if (need - (int) had > (int) free_pages_.size())
        throw Refused("the K/V pool is full (" + std::to_string(n_pos) + " positions on this lane, " + std::to_string(need - (int) had) +
                      " more pages wanted, " + std::to_string(free_pages_.size()) + " free)");
    while ((int) L.pages.size() < need) {
        L.pages.push_back(free_pages_.back());
        free_pages_.pop_back();
    }
    if (L.pages.size() == had) return;
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        ck(cudaMemcpy(b->lanes[(size_t) lane].pt + had, L.pages.data() + had, (L.pages.size() - had) * sizeof(int), cudaMemcpyHostToDevice),
           "page table");
    }
}

std::vector<unsigned long long> Engine::probe_counts() const {
    const CardBuf& b = *cb_[0];
    if (!b.probe_cnt) return {};
    std::vector<unsigned long long> v(7);
    ck(cudaSetDevice(b.dev), "set");
    ck(cudaMemcpy(v.data(), b.probe_cnt, 7 * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "probe counts");
    return v;
}

std::vector<unsigned> Engine::route_counts() const {
    const CardBuf& b = *cb_[0];
    if (!b.route_cnt) return {};
    std::vector<unsigned> v((size_t) 2 * NL * NEXP);
    ck(cudaSetDevice(b.dev), "set");
    ck(cudaMemcpy(v.data(), b.route_cnt, v.size() * sizeof(unsigned), cudaMemcpyDeviceToHost), "route counts");
    return v;
}

std::vector<unsigned> Engine::route_trace(size_t from) const {
    const CardBuf& b = *cb_[0];
    if (!b.route_trace) return {};
    unsigned long long n = 0;
    ck(cudaSetDevice(b.dev), "set");
    ck(cudaMemcpy(&n, b.route_trace_len, sizeof n, cudaMemcpyDeviceToHost), "route trace length");
    if (n > b.route_trace_cap) throw std::runtime_error("FNX_ROUTE_TRACE: " + std::to_string(n) + " picks, room for " +
                                                         std::to_string(b.route_trace_cap) + ": raise it");
    if (from >= n) return {};
    std::vector<unsigned> v((size_t) (n - from));
    std::memcpy(v.data(), b.route_trace + from, v.size() * sizeof(unsigned));
    return v;
}

std::vector<int> Engine::graph_key(const std::vector<int>& lanes) const {
    std::vector<int> k(lanes);
    for (int l : lanes) k.push_back(cb_[0]->lanes[(size_t) l].mrope != nullptr);
    k.push_back(cache_live());
    return k;
}

// a lane's states on each card, carved from a run of pool pages: recurrent states, conv histories (two halves), the
// PLE history, the QSA raw-key rings (every attention layer's and the draft layer's)
int Engine::lane_block_pages() const {
    const size_t ring_bytes = (size_t) qsa_ring_rows() * 128 * sizeof(uint16_t);
    int n = 0;   // the most any card needs (pages differ in size between cards)
    for (int c = 0; c < NCARD; ++c) {
        const CardW& w = m_.cards[c];
        size_t bytes = (size_t) 9 * D * 4 + ring_bytes + 256 * 64;   // the PLE history, the draft layer's ring, alignment
        for (int l = 0; l < NL; ++l)
            bytes += is_attn(l) ? ring_bytes : (size_t) w.nv() * HS * HS * 4 + (size_t) 2 * 3 * w.conv_ch() * 4;
        const size_t page = kvp_page(c) * sizeof(uint16_t);
        n = std::max(n, (int) ((bytes + page - 1) / page));
    }
    return n;
}

int Engine::add_lane() {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("add_lane: a window is not committed");
    const size_t ring_bytes = (size_t) qsa_ring_rows() * 128 * sizeof(uint16_t);
    auto page_bytes = [](int c) { return kvp_page(c) * sizeof(uint16_t); };
    const int n = lane_block_pages();
    // a run of n free pages, highest first
    std::vector<int> fr(free_pages_);
    std::sort(fr.begin(), fr.end());
    int start = -1;
    for (int i = (int) fr.size() - n; i >= 0; --i)
        if (fr[(size_t) i + n - 1] - fr[(size_t) i] == n - 1) {
            start = fr[(size_t) i];
            break;
        }
    // an arena lends the run at its top (a lane's block stays as long as the engine: those slots stay lent)
    if (start < 0 && arena_w_ + n <= arena_cap_) {
        start = arena_w_;
        arena_set(arena_w_ + n);
        std::fprintf(stderr, "fnx: arena: lane %zu's block at pages %d..%d, lent for good (no run of %d free pages below)\n", lanes_.size(),
                     start, start + n - 1, n);
    }
    if (start < 0) throw std::runtime_error("add_lane: no run of " + std::to_string(n) + " free pages in the K/V pool");
    free_pages_.erase(std::remove_if(free_pages_.begin(), free_pages_.end(), [&](int p) { return p >= start && p < start + n; }),
                      free_pages_.end());
    LaneHost h;
    h.block0 = start;
    h.block_n = n;
    lanes_.push_back(h);
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        char* at = (char*) (b.pool + (size_t) start * kvp_page(c));
        ck(cudaMemset(at, 0, (size_t) n * page_bytes(c)), "lane block");
        auto take = [&](size_t bytes) {
            char* p = at;
            at += (bytes + 255) & ~(size_t) 255;
            return p;
        };
        LaneDev L;
        L.pt = dalloc<int>((size_t) b.n_pages);
        L.S.assign(NL, nullptr);
        L.chist.assign(NL, nullptr);
        L.ring.assign(NL, nullptr);
        for (int l = 0; l < NL; ++l) {
            if (is_attn(l)) {
                L.ring[l] = (uint16_t*) take(ring_bytes);
            } else {
                L.S[l] = (float*) take((size_t) w.nv() * HS * HS * 4);
                L.chist[l] = (float*) take((size_t) 2 * 3 * w.conv_ch() * 4);
            }
        }
        L.ple_hist = (float*) take((size_t) 9 * D * 4);
        L.mtp_ring = (uint16_t*) take(ring_bytes);
        L.ctl = dalloc<int>(2);
        b.lanes.push_back(L);
    }
    return (int) lanes_.size() - 1;
}

// the token rows into each of the HC streams, and the PLE rows
void Engine::embed_into(int r, int k, int p0, const std::function<int(int)>& tok_at, const char*) {
    if (!h_.ple) h_.ple = halloc<float>((size_t) MAXT * N);
    std::vector<float> emb((size_t) k * N);
    embed_rows(p0, k, tok_at, emb.data(), h_.ple + (size_t) r * N);
    for (int j = 0; j < k; ++j)
        for (int c = 0; c < HC; ++c) std::memcpy(h_.R0 + (size_t) (r + j) * D + c * N, emb.data() + (size_t) j * N, N * sizeof(float));
}

void Engine::rows_upload(CardBuf& b, int T) {
    ck(cudaMemcpyAsync(b.R, h_.R0, (size_t) T * D * 4, cudaMemcpyHostToDevice, b.st), "R0");
    ck(cudaMemcpyAsync(b.ple_emb, h_.ple, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "ple");
}

void Engine::embed_rows(int p0, int T, const std::function<int(int)>& tok_at_cell, float* emb, float* ple) const {
    std::memset(ple, 0, (size_t) T * N * sizeof(float));
    static const int8_t kv4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    const Tensor& pt = *m_.ple_table;
    const size_t prb = pt.row_bytes();
    // the PLE n-grams read an image cell as the model's image token
    const auto tok_at = [&](int q) -> int64_t {
        const int t = tok_at_cell(q);
        return t >= IMAGE_TOK ? m_.ple_image : t;
    };
    for (int t = 0; t < T; ++t) {
        const int p = p0 + t;
        embed_token(tok_at_cell(p), p, emb + (size_t) t * N);
        // PLE rows: n-grams of this token and the PLE_ORDERS - 1 before it (EOS / sequence start cut the window)
        int64_t ctx[PLE_ORDERS];
        ctx[0] = tok_at(p);
        bool cut = false;
        for (int s = 1; s < PLE_ORDERS; ++s) {
            const int q = p - s;
            const int64_t tk = q >= 0 ? tok_at(q) : -1;
            cut = cut || tk < 0 || tk == m_.ple_eos;
            ctx[s] = cut ? m_.ple_eos : tk;
        }
        for (int n = 2; n <= PLE_ORDERS; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * m_.ple_mult[0];
            for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * m_.ple_mult[j];
            for (int gi = 0; gi < 8; ++gi) {
                const int h = (n - 2) * 8 + gi;
                const uint64_t r = mixed % m_.ple_vocab[h] + m_.ple_off[h];
                const uint8_t* src = pt.data + r * prb;
                for (int b = 0; b < PLE_DIM / 32; ++b) {
                    float* dst = &ple[(size_t) t * N + h * PLE_DIM + b * 32];
                    uint16_t dh;
                    if (pt.type == T_Q8_0) {
                        std::memcpy(&dh, src + b * 34, 2);
                        const float d = h2f(dh);
                        for (int j = 0; j < 32; ++j) dst[j] = d * (float) (int8_t) src[b * 34 + 2 + j];
                        continue;
                    }
                    std::memcpy(&dh, src + b * 18, 2);   // IQ4_NL
                    const float d = h2f(dh);
                    const uint8_t* qs = src + b * 18 + 2;
                    for (int j = 0; j < 16; ++j) {
                        dst[j] = d * kv4[qs[j] & 15];
                        dst[j + 16] = d * kv4[qs[j] >> 4];
                    }
                }
            }
        }
    }
}

// a sequence's device state on one card, in a fixed order: (pointer, bytes) pieces
std::vector<std::pair<char*, size_t>> state_pieces(CardBuf& b, const CardW& w, bool mtp, int pos, bool kv, unsigned hist_half,
                                                   const std::vector<int>& pages) {
    std::vector<std::pair<char*, size_t>> v;
    const size_t ring_bytes = (size_t) qsa_ring_rows() * 128 * sizeof(uint16_t);
    for (int l = 0; l < NL; ++l) {
        if (is_attn(l)) {
            v.emplace_back((char*) b.qsa_ring[l], ring_bytes);
        } else {
            v.emplace_back((char*) b.S[l], (size_t) w.nv() * HS * HS * sizeof(float));
            v.emplace_back((char*) (b.chist[l] + (size_t) hist_half * 3 * w.conv_ch()), (size_t) 3 * w.conv_ch() * sizeof(float));
        }
    }
    v.emplace_back((char*) b.ple_hist, (size_t) 9 * D * sizeof(float));
    if (mtp) v.emplace_back((char*) b.mtp_qsa_ring, ring_bytes);
    // the pages holding positions 0 .. pos (every layer's K/V rows and pooled keys, the draft layer's too)
    if (kv)
        for (int i = 0; i <= pos / KVP_POS && i < (int) pages.size(); ++i)
            v.emplace_back((char*) (b.pool + (size_t) pages[(size_t) i] * b.pg.page), b.pg.page * sizeof(uint16_t));
    return v;
}

// out[t][r] = W[r] . y[t]: the split Q8_0 kernel, or mmvq for other types
void Engine::dense(const DW& W, const void* y, float* out, CardBuf& b, int T) {
    if (W.s.q) q8s_gemv(W.s, W.n_out, W.n_in, y, out, W.n_out, T, b.st);
    else mmvq(W, 0, b.dense[T], nullptr, 1, y, out, W.n_out, b.st);
}

// FNX_BATCH_FUSE (on in a FAST build; 0: as before): the many-user window builders (enqueue_batch, enqueue_rows, mtp_enqueue_lanes) take the one-user builder's
// fused launches: q, k and v in one, ab beside qkvz in one. The same values (each output as its matrix alone gives it); 2 launches fewer in an
// attention layer's window and 1 in a GDN layer's
static bool batch_fuse() {
    static const bool on = env_int("FNX_BATCH_FUSE", FAST) != 0;
    return on;
}

// an attention layer's q, k and v from b.xq: one launch where the build has it (fuse), else three
void Engine::dense_qkv(const LayerW& L, CardBuf& b, int T, bool fuse) {
    if (fuse && L.wq.s.q && L.wk.s.q && L.wv.s.q && L.wq.n_in == L.wk.n_in && L.wq.n_in == L.wv.n_in &&
        L.wq.n_out + L.wk.n_out + L.wv.n_out >= (ATT_HEADS ? 1 : 8192)) {   // q, k and v in one launch (compiled for 8192 rows and more; by heads any)
        const Q8S ws[3] = {L.wq.s, L.wk.s, L.wv.s};
        const int rows[3] = {L.wq.n_out, L.wk.n_out, L.wv.n_out};
        float* const outs[3] = {b.qg, b.k, b.v};
        q8s_gemv_segs(ws, rows, outs, rows, 3, L.wq.n_in, b.xq, T, b.st);
    } else {
        dense(L.wq, b.xq, b.qg, b, T);
        dense(L.wk, b.xq, b.k, b, T);
        dense(L.wv, b.xq, b.v, b, T);
    }
}

// a GDN layer's qkvz (from b.xq) and ab (from b.mixed): one launch (fuse, where the build has it), else two in a row on the stream
void Engine::dense_qkvz_ab(const LayerW& L, CardBuf& b, int nv, int T, bool fuse) {
    if (fuse && L.qkvz.s.q && 2 * nv <= 160 && L.qkvz.n_out >= 4096 && L.qkvz.n_out < 8192) {
        q8s_gemv_f32(L.qkvz.s, L.qkvz.n_out, L.qkvz.n_in, b.xq, b.qkvz, L.ab, b.mixed, b.ab, 2 * nv, T, b.st);
    } else {
        gemv_f32(L.ab, b.mixed, b.ab, 2 * nv, N, T, b.st);
        dense(L.qkvz, b.xq, b.qkvz, b, T);
    }
}

// an attention layer's mixer after its hc read (mixed in b.xq): q/k/v, the window's K/V into kc/vc at ctl[0], the
// attention, the output projection into b.bo_a
void Engine::attention(int T, CardBuf& b, const CardW& w, const LayerW& L, uint16_t* kc, uint16_t* vc, const int* ctl,
                       uint16_t* qsa_ring, uint16_t* qsa_pooled, int call) {
    // the indexer first (it reads mixed / its q8 copy, which the projections below leave as they are)
    if (!L.idx_wq) dense(L.idx_qk, b.xq, b.qsa_y, b, T);
    qsa(L.idx_wq, L.idx_wk, b.mixed, b.qsa_y, L.idx_qn, L.idx_kn, qsa_ring, qsa_pooled, b.pg, b.qsa_q, b.qsa_scores, b.qsa_cells,
        b.qsa_ncells, max_ctx_ / 4, ctl, T, m_.eps, rope_of(b.mrope), b.att_ring.card, b.st);
    dense_qkv(L, b, T, true);
    attn_prep_append(b.qg, b.k, b.v, L.q_norm, L.k_norm, kc, vc, b.pg, b.att_ring.card, T, ctl, m_.eps, rope_of(b.mrope), b.st);
    // (the q8_1 form of the output for wo: attn makes it, in the combine's launch where it can)
    void* const aq = w.nq == NQH_LOC ? b.xq : nullptr;
    attn(b.qg, kc, vc, b.pg, b.att, b.att_ring, (const unsigned*) (ctl + 1), call, ctl, T, b.att_part, b.st, b.qsa_cells, b.qsa_ncells, aq);
    if (!aq) quantize_q8_1(b.att, b.xq, w.nq * HD, T, b.st);
    mall_pf_fork(b, w, L, 3);
    dense(L.wo, b.xq, b.bo_a, b, T);
}

#if FNX_MALL_PF
#ifdef FNX_MALL_NT
#define FNX_MALL_NT_LEVEL FNX_MALL_NT
#else
#define FNX_MALL_NT_LEVEL 0
#endif
// the cache prefetch (kernels.hpp PfSet, engine.hpp). Settings from the environment, read once at load:
//   FNX_MALL_PF_WG    workgroups of 256 threads (default 8; 0: no prefetch)        FNX_MALL_PF_CAP   MiB a set (default 40, 1..60)
//   FNX_MALL_PF_AT    moe (default) | mlp | mix: the fork point in a layer         FNX_MALL_PF_WRAP  1: the last layer stages layer 0's set
//   FNX_MALL_PF_FULL  1 (default): every 16 bytes, 8 loads in flight; 0: one load a line of FNX_MALL_PF_LINE bytes (default 128)
//                     (engine arms 2026-11-14, new schedule + gate, 4 cards Q4: FULL=1 WG=8 355 tok/s; FULL=0 WG=4 342; FULL=1 WG=4 349; FULL=1 WG=16 344)
//   FNX_MALL_PF_HCJOIN 1: join the side stream before every hc read (default 0: one join at the window's end; graph mode only)
//   FNX_MALL_PF_MODE  run (default): one persistent kernel a window on the side stream, outside the window's graph, that follows a count of
//                     events of the window (below); graph: a fork a layer inside the graph (measured: ~90 us a layer for the side-stream node on
//                     ROCm 7.2.4; FNX_MALL_PF_AT is this mode's)
//   FNX_MALL_PF_TIMEOUT_MS  run: a prefetcher that has not seen its window's last router by then ends itself (default 500)
//   FNX_MALL_PF_PROBE 0: run: skip the load-time check that the hc read's grids are resident beside the prefetcher's blocks
//   FNX_MALL_PF_PRIO  run: the side stream's priority class: 1 high (default), -1 low: a hardware queue of its own; 0: a default stream (it can
//                     share one of the card's 4 hardware queues: every launch of the stream that shares it waits for the prefetcher's end)
//   FNX_MALL_PF_QUERY 1: run: a stream query after every prefetcher launch (a test switch; the runtime sends a launch at once)
// A set is a layer's fixed weights on this card in the order its kernels read them; a range that would pass the cap is left out and smaller ones
// after it may still fit. Routed experts are never in a set (they are known only after the router).
// The run mode's schedule. A layer's parts, in the order it reads them: hc0 (the first hc read's matrices), proj (the mixer's projections), hc1 (the
// second hc read's), router (the table k_router_route reads: bf16 with FNX_ROUTER_BF16), sh (the shared expert); mixer = hc0 + proj (the layer's
// first half), moe = hc1 + router + sh (what its second half reads before the routed experts), all, none.
//   FNX_MALL_PF_EVENTS  the kernels of a layer that are events (each adds one to the count the prefetcher follows), names joined by '+':
//                       hc0 the first hc read, hc1 the second hc read, router the router's start, moe the router's end (its logits are written:
//                       the expert kernels come next; not together with router). Default: hc0 (a build with FNX_MALL_NT 2), hc0+hc1 (else).
//   FNX_MALL_PF_ORDER   what each event stages: one entry an event in the layer's order, '/' between them; an entry is "part:off,part:off" (off 0:
//                       the event's layer, 1: the next layer, 2: the one after). Defaults: for hc0: "moe:0,mixer:1" (this layer's second-half
//                       parts, then the next layer's first-half parts: staged beside the mixer half, whose own weights were staged a layer
//                       before); for hc0+hc1: "moe:0/mixer:1" (each part half a layer ahead: the only lead that survives the experts' plain loads
//                       of a build without FNX_MALL_NT 2); for router: "all:1" (the first form of the run mode).
//   FNX_MALL_PF_GATE    1: the prefetcher stops staging when the expert kernels start (they are bound by the bus): moe becomes an event with
//                       nothing to stage, and the blocks look at the count before every piece of FNX_MALL_PF_CHUNK KiB of a range (default 1024;
//                       a piece is shared by all the blocks, so 128 KiB a block with 8 workgroups) and leave the set when a later event
//                       has run (a block that looked just before the event still loads its piece, up to two pieces: the gate bounds the
//                       loads beside the experts, it does not make them zero). 0: a block stages every set to its end. Default 1 with
//                       FNX_MALL_NT 2 (not with the router event), else 0
//                       (there the next layer's mixer parts can only be staged beside the experts).
//   FNX_MALL_PF_WRAP    1: a part of a layer past the last one is layer 0's (for the next window); 0 (default): it is left out.
void Engine::mall_pf_plan() {
    if (!mall_pf_built()) return;
#if FNX_27B
    return;
#else
    pf_wg_ = (int) env_int("FNX_MALL_PF_WG", 8);
    if (pf_wg_ < 0 || pf_wg_ > 64) throw std::runtime_error("FNX_MALL_PF_WG: 0..64");
    const long long cap_mib = env_int("FNX_MALL_PF_CAP", 40);
    if (cap_mib < 1 || cap_mib > 60) throw std::runtime_error("FNX_MALL_PF_CAP: 1..60 MiB");
    const long long line = env_int("FNX_MALL_PF_LINE", 128);
    if (line < 32 || line > 4096 || line % 4) throw std::runtime_error("FNX_MALL_PF_LINE: 32..4096, a multiple of 4");
    const bool full = env_int("FNX_MALL_PF_FULL", 1) != 0;
    pf_wrap_ = env_int("FNX_MALL_PF_WRAP", 0) != 0;
    const char* at = std::getenv("FNX_MALL_PF_AT");
    const std::string ats = at ? at : "moe";
    pf_at_ = ats == "moe" ? 1 : ats == "mlp" ? 2 : ats == "mix" ? 3 : 0;
    if (!pf_at_) throw std::runtime_error("FNX_MALL_PF_AT: moe, mlp or mix");
    // FNX_MALL_PF_HCJOIN=1: the window's stream waits for the side stream before every hc read (k_hc holds a grid barrier and its residency probe
    // ran with the card to itself: with this no prefetch block runs beside it). mlp then forks after the FFN half's hc read, not before it
    pf_hcjoin_ = env_int("FNX_MALL_PF_HCJOIN", 0) != 0;
    if (pf_hcjoin_ && pf_at_ == 2) pf_at_ = 4;
    const char* md = std::getenv("FNX_MALL_PF_MODE");
    const std::string mds = md ? md : "run";
    if (mds != "run" && mds != "graph") throw std::runtime_error("FNX_MALL_PF_MODE: run or graph");
    pf_mode_ = mds == "run" ? 1 : 0;
    if (pf_mode_ == 1 && pf_wg_ && at) throw std::runtime_error("FNX_MALL_PF_AT is the graph mode's: the run mode takes FNX_MALL_PF_EVENTS and FNX_MALL_PF_ORDER");
    if (pf_mode_ == 1) pf_hcjoin_ = false;   // nothing to join: the prefetcher is not in the graph
    pf_query_ = env_int("FNX_MALL_PF_QUERY", 0) != 0;
    const long long to_ms = env_int("FNX_MALL_PF_TIMEOUT_MS", 500);
    if (to_ms < 10 || to_ms > 60000) throw std::runtime_error("FNX_MALL_PF_TIMEOUT_MS: 10..60000");
    if (!pf_wg_) {
        std::fprintf(stderr, "fnx: cache prefetch: built in, off (FNX_MALL_PF_WG=0)\n");
        return;
    }
    // ---- the run mode's schedule: the events of a layer and what each stages
    enum { P_HC0 = 1, P_PROJ = 2, P_HC1 = 4, P_ROUTER = 8, P_SH = 16, P_MIXER = P_HC0 | P_PROJ, P_MOE = P_HC1 | P_ROUTER | P_SH, P_ALL = 31 };
    struct Item { int parts, off; };
    std::vector<std::vector<Item>> order;   // an entry an event of the layer, in the layer's order
    std::vector<std::string> order_text;
    static const char* const ev_name[] = {"hc0", "hc1", "router", "moe"};
    if (pf_mode_ == 1) {
        const auto split = [](const std::string& t, const char* seps) {
            std::vector<std::string> out;
            for (size_t i = 0;;) {
                const size_t j = t.find_first_of(seps, i);
                out.push_back(t.substr(i, j - i));
                if (j == std::string::npos) break;
                i = j + 1;
            }
            return out;
        };
        const auto name_bits = [&](const std::string& t, const std::vector<std::pair<const char*, int>>& names) {   // names joined by '+'; -1: not one
            if (t.empty()) return -1;
            int m = 0;
            for (const std::string& n : split(t, "+")) {
                int bit = -1;
                for (const auto& nb : names)
                    if (n == nb.first) bit = nb.second;
                if (bit < 0) return -1;
                m |= bit;
            }
            return m;
        };
        const std::vector<std::pair<const char*, int>> ev_names = {{"hc0", PF_EV_HC0}, {"hc1", PF_EV_HC1}, {"router", PF_EV_ROUTER}, {"moe", PF_EV_MOE}};
        const std::vector<std::pair<const char*, int>> part_names = {{"hc0", P_HC0},     {"proj", P_PROJ}, {"hc1", P_HC1}, {"router", P_ROUTER}, {"sh", P_SH},
                                                                    {"mixer", P_MIXER}, {"moe", P_MOE},   {"all", P_ALL}, {"none", 0}};
        const bool nt2 = FNX_MALL_NT_LEVEL >= 2;
        const char* evs = std::getenv("FNX_MALL_PF_EVENTS");
        std::string evt = evs ? evs : nt2 ? "hc0" : "hc0+hc1";
        std::replace(evt.begin(), evt.end(), ',', '+');
        pf_events_ = name_bits(evt, ev_names);
        if (pf_events_ <= 0 || ((pf_events_ & PF_EV_ROUTER) && (pf_events_ & PF_EV_MOE)))
            throw std::runtime_error("FNX_MALL_PF_EVENTS: hc0, hc1, router or moe, joined by '+' (router and moe not together)");
        const char* ods = std::getenv("FNX_MALL_PF_ORDER");
        std::string ord = ods ? ods : pf_events_ == PF_EV_HC0 ? "moe:0,mixer:1" : pf_events_ == (PF_EV_HC0 | PF_EV_HC1) ? "moe:0/mixer:1"
                                    : pf_events_ == PF_EV_ROUTER ? "all:1" : "";
        if (ord.empty()) throw std::runtime_error("FNX_MALL_PF_ORDER: these events have no default order");
        order_text = split(ord, "/");
        if ((int) order_text.size() != __builtin_popcount((unsigned) pf_events_))
            throw std::runtime_error("FNX_MALL_PF_ORDER: one entry an event of FNX_MALL_PF_EVENTS, '/' between them");
        for (const std::string& ent : order_text) {
            std::vector<Item> items;
            for (const std::string& it : split(ent, ",")) {
                const size_t colon = it.find(':');
                const int parts = name_bits(it.substr(0, colon), part_names);
                const long long off = colon == std::string::npos ? 0 : std::atoll(it.c_str() + colon + 1);
                if (parts < 0 || off < 0 || off > 2) throw std::runtime_error("FNX_MALL_PF_ORDER: an item is part:off (hc0 proj hc1 router sh mixer moe all none; off 0..2), got '" + it + "'");
                if (parts) items.push_back({parts, (int) off});
            }
            order.push_back(items);
        }
        const long long gate = env_int("FNX_MALL_PF_GATE", nt2 && !(pf_events_ & PF_EV_ROUTER) ? 1 : 0), chunk = env_int("FNX_MALL_PF_CHUNK", 1024);
        if (gate < 0 || gate > 1) throw std::runtime_error("FNX_MALL_PF_GATE: 0 or 1");
        if (chunk < 64 || chunk > 16384 || chunk % 4) throw std::runtime_error("FNX_MALL_PF_CHUNK: 64..16384 KiB, a multiple of 4");
        pf_gate_ = gate ? (int) (chunk << 10) : 0;
        // FNX_MALL_PF_GATE_ROWS=R (default 4, 0: the gate for every window): windows of R rows and fewer run with the gate off: their
        // expert kernels leave the bus free enough, so a set staged to its end pays more than the pause (2026-11-16, one user, same texts:
        // R = 3 / 4 / 5 +1.5 / +1.6 / +1.7 % tok/s; a window of 1 / 2 / 3 / 4 rows -0.59 / -0.29 / -0.20 / -0.11 ms, of 5 rows +0.08 ms with R = 5)
        pf_gate_rows_ = (int) env_int("FNX_MALL_PF_GATE_ROWS", 4);
        if (pf_gate_rows_ < 0 || pf_gate_rows_ > MAXT) throw std::runtime_error("FNX_MALL_PF_GATE_ROWS: 0..16");
        if (gate && (pf_events_ & PF_EV_MOE) && !order.back().empty())   // (moe is the layer's last event)
            throw std::runtime_error("FNX_MALL_PF_GATE=1: the moe event is the pause, its entry must be none (FNX_MALL_PF_ORDER)");
        if (gate && !(pf_events_ & PF_EV_MOE)) {   // the event before the expert kernels, with nothing to stage: the pause
            if (pf_events_ & PF_EV_ROUTER) throw std::runtime_error("FNX_MALL_PF_GATE=1 makes moe an event: not with the router event (FNX_MALL_PF_EVENTS)");
            pf_events_ |= PF_EV_MOE;
            order.push_back({});   // moe is the layer's last event
            order_text.push_back("none (the gate)");
        }
        pf_nev_ = __builtin_popcount((unsigned) pf_events_);
    }
    const long long cap = cap_mib << 20;
    for (int c = 0; c < NCARD; ++c) {
        const CardW& w = m_.cards[c];
        CardBuf& b = *cb_[c];
        // a layer's parts on this card, in the order its kernels read them
        const auto ranges = [&](int l, int parts) {
            std::vector<PfSet::R> out;
            const LayerW& L = w.L[l];
            const auto add = [&](const void* p, long long bytes) {
                if (p && bytes >= 4096) out.push_back(PfSet::R{p, bytes});
            };
            const auto q8s = [&](const Q8S& q, long long rows, long long cols) {
                add(q.q, rows * cols);
                add(q.d, rows * cols / 32 * 2);
            };
            const auto dw = [&](const DW& W) {
                if (W.s.q) q8s(W.s, W.n_out, W.n_in);
                else add(W.p, (long long) W.n_out * (long long) W.row_bytes);
            };
            const auto hc = [&](int h) {
                add(L.hc_norm[h], (long long) D * 4);
                q8s(L.hc_down[h], LR, D);
                add(L.hc_inj[h], (long long) HC * D * 2);
                q8s(L.hc_up[h], D, LR);
            };
            if (parts & P_HC0) hc(0);
            if (parts & P_PROJ) {
                if (!is_attn(l)) {
                    dw(L.qkvz);
                    add(L.ab, (long long) 2 * w.nv() * N * 4);
                    dw(L.ssm_out);
                } else {
                    if (L.idx_wq) {
                        add(L.idx_wq, (long long) QSA_HEADS * QSA_KEY * N * 2);
                        add(L.idx_wk, (long long) QSA_KEY * N * 2);
                    } else {
                        dw(L.idx_qk);
                    }
                    dw(L.wq);
                    dw(L.wk);
                    dw(L.wv);
                    dw(L.wo);
                }
            }
            if (parts & P_HC1) hc(1);
            if (parts & P_ROUTER) {
                if (router_bf_of(L) && router_bf16_mode()) add(router_bf_of(L), (long long) (NEXP + 1) * N * 2);   // the table k_router_route reads
                else add(L.router, (long long) (NEXP + 1) * N * 4);
            }
            if (parts & P_SH) {
                dw(L.sh_gu);
                dw(L.sh_down);
            }
            return out;
        };
        int cuts = 0;
        // ranges appended to a set under the cap (one that would pass it is left out, smaller ones after it may still fit)
        const auto fill = [&](PfSet& s, long long& sum, const std::vector<PfSet::R>& rs) {
            for (const PfSet::R& r : rs) {
                if (sum + r.bytes > cap || s.n == MALL_PF_MAXR) {
                    ++cuts;
                    continue;
                }
                s.r[s.n++] = r;
                sum += r.bytes;
            }
        };
        const auto blank = [&] {
            PfSet s;
            s.line = (int) line;
            s.full = full;
            s.sink = b.pf_sink;
            return s;
        };
        const double mib = 1048576.0;
        const std::string how = full ? "every 16 bytes" : std::string("one word a ") + std::to_string(line) + " bytes";
        if (pf_mode_ == 0) {   // graph: a set a layer, the whole layer
            b.pf_sets.assign(NL, blank());
            long long lo[2] = {-1, -1}, hi[2] = {0, 0}, total = 0;   // [0] GDN layers, [1] attention layers
            for (int l = 0; l < NL; ++l) {
                long long sum = 0;
                fill(b.pf_sets[(size_t) l], sum, ranges(l, P_ALL));
                if (l || pf_wrap_) total += sum;   // layer 0's set is staged only by the last layer (FNX_MALL_PF_WRAP)
                const int k = is_attn(l) ? 1 : 0;
                lo[k] = lo[k] < 0 ? sum : std::min(lo[k], sum);
                hi[k] = std::max(hi[k], sum);
            }
            std::fprintf(stderr,
                         "fnx: cache prefetch: card %d: %d workgroups, a fork a layer in the graph at %s%s, %s; sets GDN %.1f..%.1f MiB, attention "
                         "%.1f..%.1f MiB (cap %lld), %.0f MiB a window, %d ranges cut by the cap%s\n",
                         c, pf_wg_, ats.c_str(), pf_hcjoin_ ? " (joined before every hc read)" : "", how.c_str(), (double) std::max(lo[0], 0LL) / mib,
                         (double) hi[0] / mib, (double) std::max(lo[1], 0LL) / mib, (double) hi[1] / mib, cap_mib, (double) total / mib, cuts,
                         pf_wrap_ ? ", the last layer stages layer 0's" : "");
            continue;
        }
        // run: the persistent form reads its sets from the card. After p events of a window the prefetcher stages entry p (p = the window's
        // events: entry 0): event e's set is entry e + 1
        const int n = NL * pf_nev_;
        std::vector<PfSet> tab((size_t) n, blank());
        std::vector<long long> lo((size_t) pf_nev_, -1), hi((size_t) pf_nev_, 0);
        long long total = 0;
        b.pf_full = 0;
        for (int l = 0, e = 0; l < NL; ++l)
            for (int k = 0; k < pf_nev_; ++k, ++e) {
                PfSet& s = tab[(size_t) ((e + 1) % n)];
                long long sum = 0;
                for (const Item& it : order[(size_t) k]) {
                    int ll = l + it.off;
                    if (ll >= NL) {
                        if (!pf_wrap_) continue;   // a part of the next window's first layers: left out
                        ll -= NL;
                    }
                    fill(s, sum, ranges(ll, it.parts));
                }
                total += sum;
                if (s.n) ++b.pf_full;
                lo[(size_t) k] = lo[(size_t) k] < 0 ? sum : std::min(lo[(size_t) k], sum);
                hi[(size_t) k] = std::max(hi[(size_t) k], sum);
            }
        ck(cudaSetDevice(b.dev), "set");
        mall_pf_table(b.pf_table, tab.data(), n);
        b.pf_timeout = mall_pf_ticks((double) to_ms);
        if (b.pf_timeout <= 0) throw std::runtime_error("cache prefetch: no wall-clock rate on card " + std::to_string(c));
        std::string sched;
        for (int bit = 0, k = 0; bit < 4; ++bit)
            if (pf_events_ >> bit & 1) {
                char t[320];
                std::snprintf(t, sizeof t, "%s%s -> %s (%.1f..%.1f MiB)", k ? "; " : "", ev_name[bit], order_text[(size_t) k].c_str(),
                              (double) std::max(lo[(size_t) k], 0LL) / mib, (double) hi[(size_t) k] / mib);
                sched += t;
                ++k;
            }
        std::fprintf(stderr,
                     "fnx: cache prefetch: card %d: %d workgroups, one persistent kernel a window (run), %s; events a layer: %s; cap %lld MiB a set, "
                     "%d ranges cut by it; %.0f MiB a window; gate %s%s\n",
                     c, pf_wg_, how.c_str(), sched.c_str(), cap_mib, cuts, (double) total / mib,
                     pf_gate_ ? ("on, a look at the count every " + std::to_string(pf_gate_ >> 10) + " KiB").c_str() : "off",
                     pf_wrap_ ? "; parts past the last layer are layer 0's" : "");
    }
    // run: the hc read's grid barrier needs every block of its grid resident at once, and hc_prepare's probe ran with the card to itself. Here the
    // grids in use are probed again beside a resident, idle prefetcher (it polls a count that does not move until the stop word ends it). A grid
    // that is not resident then would time out in every read of a window: the prefetcher is turned off instead (also when the probe gave no answer).
    // (the hc read's ticket queue, FNX_MM_HC, has no barrier and no grid to be resident: nothing to probe)
    if (pf_mode_ == 1 && mm_hc()) std::fprintf(stderr, "fnx: cache prefetch: the hc read is the ticket queue (FNX_MM_HC): no grid to probe beside the prefetcher\n");
    if (pf_mode_ == 1 && !mm_hc() && env_int("FNX_MALL_PF_PROBE", 1) != 0 && !std::getenv("FNX_HC_BAR_WRAP")) {
        bool off = false;
        for (int c = 0; c < NCARD; ++c) {
            CardBuf& b = *cb_[c];
            ck(cudaSetDevice(b.dev), "set");
            PfProbeInfo pi;
            const int bad = mall_pf_hc_probe(b.hcs, b.pf_table, NL * pf_nev_, b.pf_words, pf_wg_, b.st3, b.st, &pi, pf_query_);
            std::fprintf(stderr, "fnx: cache prefetch: card %d: side stream %s; beside %d resident prefetch workgroups:%s\n", c,
                         b.pf_own ? "with a hardware queue of its own" : "a default stream (it can share a hardware queue)", pf_wg_,
                         mall_pf_probe_text(bad, pi, pf_wg_).c_str());
            if (bad) off = true;
        }
        if (off) {
            std::fprintf(stderr, "fnx: cache prefetch: off (lower FNX_MALL_PF_WG, or FNX_HC_GRID2=0, or FNX_MALL_PF_MODE=graph)\n");
            pf_wg_ = 0;
        }
    }
#endif
}

// run: the count word for a kernel of trunk layer L that is one of `events` (PF_EV_*; the draft layer and any other LayerW: none)
unsigned* Engine::mall_pf_mark(CardBuf& b, const CardW& w, const LayerW& L, int events) {
    if (!pf_wg_ || pf_mode_ != 1 || !(pf_events_ & events)) return nullptr;
    for (int i = 0; i < NL && i < (int) w.L.size(); ++i)
        if (&w.L[(size_t) i] == &L) return b.pf_words;
    return nullptr;
}

// run: before a trunk window runs on card b (its graph launch, or run false: an eager window): the window's prefetcher on the side stream, with
// the count its first event will find, and the count moved on by the window's events (pf_nev_ a layer). Nothing waits for the prefetcher: it ends
// at the window's last event (or its time-out), and the next one queues behind it on the stream. Every 256 windows the card's words are read (the
// window before is done, so the count on the card is this one): a status bit is logged once, a count that is not ours is logged and taken.
void Engine::mall_pf_window(CardBuf& b, bool run, int rows) {
    if (!pf_wg_ || pf_mode_ != 1) return;
    if ((b.pf_windows++ & 255u) == 255u) {
        // what the prefetchers did since the last report (words[5..7], a block each): at the first check and every 4096 windows. The side stream
        // is synchronized first: the last window's prefetcher can still be at its last set (it ends by itself, at the latest at its time-out)
        const bool rep = b.pf_runs > b.pf_runs_rep && (b.pf_windows == 256u || (b.pf_windows & 4095u) == 0);
        if (rep) ck(cudaStreamSynchronize(b.st3), "prefetch sync");
        unsigned h[MALL_PF_WORDS] = {};
        ck(cudaMemcpy(h, b.pf_words, sizeof h, cudaMemcpyDeviceToHost), "prefetch words");
        if (rep) {
            // of the sets that have something to stage (b.pf_full a window): staged to their end, left when a later event had run (the gate).
            // Of all sets (one an event): left out (a block that fell behind)
            const double runs = (double) (b.pf_runs - b.pf_runs_rep), sets = std::max(1.0, runs * (double) b.pf_full * (double) pf_wg_);
            const double all = std::max(1.0, runs * (double) (NL * pf_nev_) * (double) pf_wg_);
            const unsigned d5 = h[5] - b.pf_rep[0], d6 = h[6] - b.pf_rep[1], d7 = h[7] - b.pf_rep[2], d8 = h[8] - b.pf_rep[3], d9 = h[9] - b.pf_rep[4];
            const unsigned d10 = h[10] - b.pf_rep[5], d11 = h[11] - b.pf_rep[6];
            std::fprintf(stderr,
                         "fnx: cache prefetch: card (device %d): %.0f prefetchers: %.1f %% of their sets staged to the end, %.1f %% left at a later "
                         "event (those were staged to %.0f %% on average, %.1f MiB of one left cold), %.1f %% of all sets left out, %.1f %% of the "
                         "blocks started at their window's end or later\n",
                         b.dev, runs, 100.0 * (d5 - d9) / sets, 100.0 * d8 / sets, d8 ? 100.0 * d10 / (256.0 * d8) : 0.0,
                         d8 ? (double) d11 / 4.0 / d8 : 0.0, 100.0 * d7 / all, 100.0 * d6 / (runs * pf_wg_));
            {   // block 0's time account (kernels.hpp words[12..15]): a window's side stream (the MB are block 0's position: an estimate)
                const double us = 64.0e3 / (double) std::max(1ll, mall_pf_ticks(1.0)), act = (h[12] - b.pf_rep[7]) * us / runs;
                const double spare = (h[13] - b.pf_rep[8]) * us / runs, pause = (h[14] - b.pf_rep[9]) * us / runs;
                const double mb = (double) (h[15] - b.pf_rep[10]) * 65536.0 / 1e6 / runs;
                std::fprintf(stderr,
                             "fnx: cache prefetch: card (device %d): a window's side stream: %.0f us in sets (~%.1f MB staged, ~%.0f GB/s), %.0f us "
                             "spare after sets staged to the end, %.0f us paused; a layer: %.1f us in sets, %.2f MB, %.1f us spare, %.1f us paused\n",
                             b.dev, act, mb, act > 0 ? mb * 1e3 / act : 0.0, spare, pause, act / NL, mb / NL, spare / NL, pause / NL);
                for (int k = 0; k < 4; ++k) b.pf_rep[7 + k] = h[12 + k];
            }
            b.pf_rep[0] = h[5], b.pf_rep[1] = h[6], b.pf_rep[2] = h[7], b.pf_rep[3] = h[8], b.pf_rep[4] = h[9], b.pf_rep[5] = h[10], b.pf_rep[6] = h[11], b.pf_runs_rep = b.pf_runs;
        }
        if (h[0] != b.pf_count) {
            std::fprintf(stderr, "fnx: cache prefetch: card (device %d): the events' count is %u, the engine's %u: taken from the card\n", b.dev, h[0],
                         b.pf_count);
            b.pf_count = h[0];
        }
        if (const unsigned fresh = h[1] & ~b.pf_seen) {
            std::fprintf(stderr, "fnx: cache prefetch: card (device %d): %s%s (status %u)\n", b.dev, fresh & 1u ? "a prefetcher timed out " : "",
                         fresh & 2u ? "a prefetcher found a count past its window" : "", h[1]);
            b.pf_seen |= h[1];
        }
    }
    if (run) {
        PfRun r;
        // (wrap: entry 0 is the last event's set, mall_pf_plan)
        r.sets = b.pf_table, r.words = b.pf_words, r.n = NL * pf_nev_, r.wrap = 1, r.base = b.pf_count, r.timeout = b.pf_timeout;
        r.gate = rows <= pf_gate_rows_ ? 0 : pf_gate_;   // (a small window's experts leave the bus free: its sets are staged to their end)
        mall_pf_run(r, pf_wg_, b.st3, pf_query_);
        ++b.pf_runs;
    }
    b.pf_count += (unsigned) (NL * pf_nev_);
}

// the fork at `point` of layer L (a trunk layer of this card; the draft layer and any other LayerW: nothing): layer + 1's set on the side stream, after
// the window's work up to here. Stream order on st3 keeps the sets in layer order; nothing waits for them until the window's end (mall_pf_join)
void Engine::mall_pf_fork(CardBuf& b, const CardW& w, const LayerW& L, int point) {
    if (!pf_wg_ || pf_mode_ != 0 || point != pf_at_ || b.pf_sets.empty()) return;
    // the trunk layer's index by pointer equality (the draft layer's LayerW is not an element of w.L: no pointer subtraction across arrays)
    int lyr = -1;
    for (int i = 0; i < NL && i < (int) w.L.size(); ++i)
        if (&w.L[(size_t) i] == &L) {
            lyr = i;
            break;
        }
    if (lyr < 0) return;
    int next = lyr + 1;
    if (next >= NL) {
        if (!pf_wrap_) return;
        next = 0;
    }
    const PfSet& s = b.pf_sets[(size_t) next];
    if (!s.n) return;
    ck(cudaEventRecord(b.pf_fork, b.st), "prefetch fork");
    ck(cudaStreamWaitEvent(b.st3, b.pf_fork, 0), "prefetch fork wait");
    mall_pf(s, pf_wg_, b.st3);
    b.pf_open = true;
}

// before a trunk hc read: with FNX_MALL_PF_HCJOIN the window's stream waits for the side stream here
void Engine::mall_pf_hc(CardBuf& b) {
    if (pf_hcjoin_) mall_pf_join(b);
}

// the window's one join: its stream waits for the side stream's last set (a captured window must end with every forked stream joined)
void Engine::mall_pf_join(CardBuf& b) {
    if (!b.pf_open) return;
    ck(cudaEventRecord(b.pf_join, b.st3), "prefetch join");
    ck(cudaStreamWaitEvent(b.st, b.pf_join, 0), "prefetch join wait");
    b.pf_open = false;
}
#endif   // FNX_MALL_PF

// the FFN half: the hc read with the attention half's write folded in, the shared expert (in the router's and the
// routed gate|up's launches on Q4_K layers, else on the second stream beside the router), the routing and the routed
// experts, the weighted sum across the cards into b.bo_f
void Engine::sum_later(CardBuf& b, float* x, int T, const unsigned* tag, int call, const RingMoe& m) {
    sum_flush(b);
    if (T > hc_sum_rows() || sum_now_) {   // (hc_read has the other conditions: it launches the sum itself when it cannot take it)
        ring_sum(b.ring, x, T, tag, call, b.st, m);
        return;
    }
    b.pend.ring = &b.ring, b.pend.x = x, b.pend.tag = tag, b.pend.call = call, b.pend.m = m;
    b.pend_T = T, b.pend_on = true;
}
// every window sum of cards lo .. hi - 1 was launched: a read took it or sum_flush did (a sum left pending would be a wrong window, silently)
void Engine::sum_done(int lo, int hi) const {
    for (int c = lo; c < hi; ++c)
        if (cb_[c]->pend_on) throw std::logic_error("a window sum was left pending on card " + std::to_string(c) + " (FNX_HC_SUM)");
}
void Engine::sum_flush(CardBuf& b) {
    if (!b.pend_on) return;
    b.pend_on = false;
    ring_sum(*b.pend.ring, b.pend.x, b.pend_T, b.pend.tag, b.pend.call, b.st, b.pend.m);
}
const HcSumReq* Engine::sum_take(CardBuf& b, const float* bo, int T) {
    if (!b.pend_on) return nullptr;
    if (b.pend.x != bo || b.pend_T != T) {
        sum_flush(b);
        return nullptr;
    }
    b.pend_on = false;
    return &b.pend;
}

void Engine::ffn_half(int T, int call, CardBuf& b, const CardW& w, const LayerW& L, float* R, const int* ctl) {
    const int P = T * TOPK, ffs = w.ff1 - w.ff0;
    mall_pf_fork(b, w, L, 2);
    mall_pf_hc(b);
    const HcSumReq* const hsum = sum_take(b, b.bo_a, T);
    hc_read(R, b.bo_a, b.inj_a, L.hc_norm[1], L.hc_down[1], L.hc_up[1], L.hc_inj[1], m_.eps, b.mixed, b.inj_f, b.hcs, b.ring.err, T, b.st, b.xq
#if FNX_MALL_PF
            ,
            mall_pf_mark(b, w, L, PF_EV_HC1)
#endif
            ,
            hsum);
    mall_pf_fork(b, w, L, 4);
    // the shared expert (rows P + t of part): its gate|up in the router's launch and its swiglu and down in the routed
    // gate|up's (Q4_K experts), else on the second stream (a fork and a join cost ~5 us more than the launches' blocks)
    // (every expert type moe_gate_up handles; FNX_SH_FUSE=0 puts the non-Q4_K types back on the second stream)
    static const bool sh_all = env_int("FNX_SH_FUSE", 1) != 0;
    const bool sh_in = (L.gu_e.type == T_Q4_K || (sh_all && (L.gu_e.type == T_Q5_K || L.gu_e.type == T_Q5_1 || L.gu_e.type == T_Q4_0 || L.gu_e.type == T_Q8_0 || L.gu_e.type == T_IQ4_NL || L.gu_e.type == T_IQ4_XS || L.gu_e.type == T_IQ3_S || L.gu_e.type == T_IQ3_XXS || L.gu_e.type == T_IQ2_S))) && L.sh_gu.s.q && L.sh_down.s.q && ffs % 32 == 0 && ffs <= SH_MAXF;
    if (!sh_in) {
        ck(cudaEventRecord(b.fork, b.st), "fork");
        ck(cudaStreamWaitEvent(b.st2, b.fork, 0), "fork wait");
        q8s_gemv(L.sh_gu.s, 2 * ffs, N, b.xq, b.sh_gu, 2 * ffs, T, b.st2);
        swiglu_stacked_q8_1(b.sh_gu, b.sh_hq, ffs, T, b.st2);
        q8s_gemv(L.sh_down.s, N, ffs, b.sh_hq, b.part + (size_t) P * N, N, T, b.st2);
        ck(cudaEventRecord(b.join, b.st2), "join");
    }
    const int lyr = (int) (&L - w.L.data());
    // experts in RAM: the router's last block plans this window's slots (the prefetch probe scores the picks against
    // the slots as they were, so with it the plan runs after, in moe_cache)
    const bool live = cache_live() && lyr >= 0 && lyr < NL;
    const bool plan_in_router = live && !b.probe_cnt;
    mall_pf_fork(b, w, L, 1);
    router_route(L.router, b.mixed, b.logits, T, b.topi, b.topw, b.jobs_gu, b.jobs_dn, b.njobs, b.st, plan_in_router ? &w.ec : nullptr,
                 lyr, b.cache_list, sh_in ? &L.sh_gu.s : nullptr, 2 * ffs, b.xq, b.sh_gu, router_bf_of(L)
#if FNX_MALL_PF
                 ,
                 mall_pf_mark(b, w, L, PF_EV_ROUTER | PF_EV_MOE), (pf_events_ & PF_EV_MOE) ? 1 : 0
#endif
    );
    if (const int l = lyr; l >= 0 && l < NL) {
        if (b.route_cnt) route_count(b.topi, P, b.route_cnt + (size_t) l * NEXP, b.st);
        if (b.route_trace) fnx::route_trace(b.topi, P, l, b.route_trace, b.route_trace_len, b.route_trace_cap, b.st);
        if (b.probe_cnt) {
            if (l > 0)
                probe_score(b.topi, b.probe_guess + (size_t) l * MAXT * TOPK, T, w.ec.n_slots ? w.ec.slot_of + (size_t) l * NEXP : nullptr,
                            b.probe_cnt, b.st);
            if (l + 1 < NL) {
                gemv_f32(w.L[(size_t) l + 1].router, b.mixed, b.probe_logits, NEXP, N, T, b.st);
                probe_guess(b.probe_logits, T, b.probe_guess + (size_t) (l + 1) * MAXT * TOPK, b.st);
            }
        }
        // the planned experts copied into their slots before the MoE kernels read the layer's tables
        if (live) moe_cache(l, L.gu_e_stride, L.down_e_stride, w.ec, b.cache_list, b.st, plan_in_router ? nullptr : b.jobs_gu, b.njobs);
    }
    if (sh_in) moe_gate_up(L.gu_e, L.gu_e_stride, b.jobs_gu, b.njobs, P, b.xq, b.gu, b.st, &L.sh_down.s, ffs, b.sh_gu, b.part + (size_t) P * N);
    else moe_gate_up(L.gu_e, L.gu_e_stride, b.jobs_gu, b.njobs, P, b.xq, b.gu, b.st);
    moe_down(L.down_e, L.down_e_stride, b.jobs_dn, b.njobs, P, b.gu, b.part, b.st);
    if (!sh_in) ck(cudaStreamWaitEvent(b.st, b.join, 0), "join wait");
    RingMoe m;
    m.part = b.part;
    m.topw = b.topw;
    m.logits = b.logits;
    sum_later(b, b.bo_f, T, (const unsigned*) (ctl + 1), call, m);
}

// every card's work for one window of T rows, in stream order (captured into a graph unless dumping)
void Engine::enqueue(int T, bool dumps) {
    for (int c = 0; c < NCARD; ++c) cb_[c]->pend_on = false;   // (a window that ended early leaves no sum behind)
    sum_now_ = dumps;
    const float eps = m_.eps;
    int call = 0;
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    auto dd = [&](const std::string& name, int layer, const float* p, int n) {
        if (dumps) dump_dev(name, layer, p, T, n);
    };
    for (int l = 0; l < NL; ++l) {
        const bool pending = l > 0;
        // ---- attention half
        each([&](int c, CardBuf& b, const CardW& w) {
            const LayerW& L = w.L[l];
            bool apply = pending;
            if (l == PLE_LAYER) {
                sum_flush(b);
                hc_write(b.R, b.bo_f, b.inj_f, T, b.st);
                apply = false;
                quantize_q8_1(b.ple_emb, b.xq, N, T, b.st);
                dense(w.ple_key, b.xq, b.ple_key, b, T);
                dense(w.ple_value, b.xq, b.ple_val, b, T);
                ple_post(b.R, b.ple_key, b.ple_val, w.ple_norm_key, w.ple_norm_query, w.ple_norm_conv, w.ple_conv,
                         b.ple_hist, b.ple_hsnap, T, eps, b.st, b.ple_key + (size_t) MAXT * D);
            }
            mall_pf_hc(b);
            if (!apply) sum_flush(b);
            const HcSumReq* const hsum = apply ? sum_take(b, b.bo_f, T) : nullptr;
            hc_read(b.R, apply ? b.bo_f : nullptr, b.inj_f, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed,
                    b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq
#if FNX_MALL_PF
                    ,
                    mall_pf_mark(b, w, L, PF_EV_HC0)
#endif
                    ,
                    hsum);
            if (!is_attn(l)) {
                const int nv = w.nv(), ch = w.conv_ch();
                // ab beside qkvz in one launch (a fork and a join: ~5 us more; compiled for 4096..8191 rows)
                if (L.qkvz.s.q && 2 * nv <= 160 && L.qkvz.n_out >= 4096 && L.qkvz.n_out < 8192) {
                    q8s_gemv_f32(L.qkvz.s, L.qkvz.n_out, L.qkvz.n_in, b.xq, b.qkvz, L.ab, b.mixed, b.ab, 2 * nv, T, b.st);
                } else {
                    ck(cudaEventRecord(b.fork, b.st), "fork");
                    ck(cudaStreamWaitEvent(b.st2, b.fork, 0), "fork wait");
                    gemv_f32(L.ab, b.mixed, b.ab, 2 * nv, N, T, b.st2);
                    ck(cudaEventRecord(b.join, b.st2), "join");
                    dense(L.qkvz, b.xq, b.qkvz, b, T);
                    ck(cudaStreamWaitEvent(b.st, b.join, 0), "join wait");
                }
                gdn_core(b.qkvz, b.ab, b.chist[l], L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.S[l], b.xq, b.ctl, b.gdn_scratch, b.snap[l],
                         b.hsnap[l], w.nk, nv, w.vmap, T, eps, b.st);
                mall_pf_fork(b, w, L, 3);
                dense(L.ssm_out, b.xq, b.bo_a, b, T);
            } else {
                attention(T, b, w, L, b.kc[l], b.vc[l], b.ctl, b.qsa_ring[l], b.qsa_pooled[l], 128 + l);
            }
            sum_later(b, b.bo_a, T, (const unsigned*) (b.ctl + 1), call);
        });
        ++call;
        dd("hc_mixed", l, cb_[0]->mixed, N);
        dd("mixer_out", l, cb_[0]->bo_a, N);
        // ---- FFN half
        each([&](int, CardBuf& b, const CardW& w) { ffn_half(T, call, b, w, w.L[l], b.R, b.ctl); });
        df2_tap_window(l, T);
        ++call;
        dd("ffn_mixed", l, cb_[0]->mixed, N);
        dd("ffn_out", l, cb_[0]->bo_f, N);
        dd("R_pending", l, cb_[0]->R, D);
        dd("inj_ffn", l, cb_[0]->inj_f, HC);
    }
    // ---- head: the final mixer (no injection) with the last write folded in, then this card's vocabulary
    each([&](int c, CardBuf& b, const CardW& w) {
        mall_pf_hc(b);
        const HcSumReq* const hsum = sum_take(b, b.bo_f, T);
        hc_read(b.R, b.bo_f, b.inj_f, w.head_norm, w.head_down, w.head_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq
#if FNX_MALL_PF
                ,
                nullptr
#endif
                ,
                hsum);
        dense(w.out, b.xq, b.head_logits, b, T);
        argmax(b.head_logits, w.v1 - w.v0, T, w.v0, b.best_v, b.best_i, b.am_keys, b.ring.err, b.st);
        mall_pf_join(b);
    });
    dd("result_norm", -1, cb_[0]->mixed, N);
    sum_done(0, NCARD);
}

// enqueue()'s layers for a window of one row per lane (row r: lanes[r]): the shared work over all rows, the
// sequence's own work (the PLE block, the recurrent core, QSA, the rotation, the K/V append, the attention partials)
// per row with its lane's state and ctl
void Engine::enqueue_batch(const std::vector<int>& lanes) {
    for (int c = 0; c < NCARD; ++c) cb_[c]->pend_on = false;   // (a window that ended early leaves no sum behind)
    sum_now_ = false;
    const int T = (int) lanes.size();
    const float eps = m_.eps;
    int call = 0;
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    for (int l = 0; l < NL; ++l) {
        each([&](int c, CardBuf& b, const CardW& w) {
            const LayerW& L = w.L[l];
            bool apply = l > 0;
            if (l == PLE_LAYER) {
                sum_flush(b);
                hc_write(b.R, b.bo_f, b.inj_f, T, b.st);
                apply = false;
                quantize_q8_1(b.ple_emb, b.xq, N, T, b.st);
                dense(w.ple_key, b.xq, b.ple_key, b, T);
                dense(w.ple_value, b.xq, b.ple_val, b, T);
                for (int r = 0; r < T; ++r)   // gated and normed rows past every key row (ple_key holds 3 MAXT rows)
                    ple_post(b.R + (size_t) r * D, b.ple_key + (size_t) r * D, b.ple_val + (size_t) r * N, w.ple_norm_key, w.ple_norm_query,
                             w.ple_norm_conv, w.ple_conv, b.lanes[(size_t) lanes[r]].ple_hist, b.ple_hsnap, 1, eps, b.st,
                             b.ple_key + (size_t) MAXT * D);
            }
            mall_pf_hc(b);
            if (!apply) sum_flush(b);
            const HcSumReq* const hsum = apply ? sum_take(b, b.bo_f, T) : nullptr;
            hc_read(b.R, apply ? b.bo_f : nullptr, b.inj_f, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed,
                    b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq
#if FNX_MALL_PF
                    ,
                    mall_pf_mark(b, w, L, PF_EV_HC0)
#endif
                    ,
                    hsum);
            if (!is_attn(l)) {
                const int nv = w.nv();
                dense_qkvz_ab(L, b, nv, T, batch_fuse());
                float** rec = b.lane_gdn[(size_t) T];
                const GdnLanes gl{rec + (size_t) l * MAXT, rec + (size_t) (NL + l) * MAXT, b.lane_rows[(size_t) T]};
                gdn_core_lanes(b.qkvz, b.ab, L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.xq, b.gdn_scratch, w.nk, nv, w.vmap, gl, T, eps, b.st);
                mall_pf_fork(b, w, L, 3);
                dense(L.ssm_out, b.xq, b.bo_a, b, T);
            } else {
                if (!L.idx_wq) dense(L.idx_qk, b.xq, b.qsa_y, b, T);
                else qsa_proj(L.idx_wq, L.idx_wk, b.mixed, b.qsa_y, T, b.st);
                dense_qkv(L, b, T, batch_fuse());
                const int mp = max_ctx_ / 4;
                for (int r = 0; r < T; ++r) {
                    const LaneDev& LD = b.lanes[(size_t) lanes[r]];
                    KvPages pg = b.pg;   // the card's page size and rows, the lane's table
                    pg.pt = LD.pt;
                    qsa_index(b.qsa_y + (size_t) r * qsa_rows(), L.idx_qn, L.idx_kn, LD.ring[l], b.qsa_pooled[l], pg, b.qsa_q + (size_t) r * 512,
                              LD.ctl, 1, eps, rope_of(LD.mrope), b.st);
                    attn_prep(b.qg + (size_t) r * QG_W, b.k + (size_t) r * KV_W, L.q_norm, L.k_norm, NQH_LOC, KVH_LOC, LD.ctl, 1, eps,
                              rope_of(LD.mrope), b.st);
                    kv_append(b.k + (size_t) r * KV_W, b.v + (size_t) r * KV_W, b.kc[l], b.vc[l], pg, b.att_ring.card, 1, LD.ctl,
                              b.st);
                }
                // every row's selection and attention partials at once, each on its lane's position and pages
                const LaneRows lr{b.lane_rows[(size_t) T], b.lane_rows[(size_t) T] + MAXT};
                qsa_select_lanes(b.qsa_q, b.qsa_pooled[l], b.pg, b.qsa_scores, b.qsa_cells, b.qsa_ncells, mp, lr, T, b.att_ring.card, b.st);
                attn_part_lanes(b.qg, b.kc[l], b.vc[l], b.pg, lr, T, b.att_part, b.st, b.qsa_cells, b.qsa_ncells, b.att_ring.card);
                attn_finish(b.qg, b.att, b.att_ring, (const unsigned*) (b.ctl + 1), 128 + l, T, b.att_part, b.st, w.nq == NQH_LOC ? b.xq : nullptr);
                if (w.nq != NQH_LOC) quantize_q8_1(b.att, b.xq, w.nq * HD, T, b.st);
                mall_pf_fork(b, w, L, 3);
                dense(L.wo, b.xq, b.bo_a, b, T);
            }
            sum_later(b, b.bo_a, T, (const unsigned*) (b.ctl + 1), call);
        });
        ++call;
        each([&](int, CardBuf& b, const CardW& w) { ffn_half(T, call, b, w, w.L[l], b.R, b.ctl); });
        ++call;
    }
    each([&](int, CardBuf& b, const CardW& w) {
        mall_pf_hc(b);
        const HcSumReq* const hsum = sum_take(b, b.bo_f, T);
        hc_read(b.R, b.bo_f, b.inj_f, w.head_norm, w.head_down, w.head_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq
#if FNX_MALL_PF
                ,
                nullptr
#endif
                ,
                hsum);
        dense(w.out, b.xq, b.head_logits, b, T);
        argmax(b.head_logits, w.v1 - w.v0, T, w.v0, b.best_v, b.best_i, b.am_keys, b.ring.err, b.st);
        mall_pf_join(b);
    });
    sum_done(0, NCARD);
}

// enqueue_batch()'s layers for a window of several rows a lane (verify_rows): the shared work over all rows as there;
// each lane's own work over its rows as a window of its own does it (the rows chained through its recurrent state, the
// state and histories after each row kept for commit_rows), and every row's selection and attention partials at once,
// each at its own position (row_pos) on its lane's pages
void Engine::enqueue_rows(const std::vector<int>& lanes, const std::vector<int>& counts) {
    for (int c = 0; c < NCARD; ++c) cb_[c]->pend_on = false;   // (a window that ended early leaves no sum behind)
    sum_now_ = false;
    const RowSegs sg = row_segs(counts);
    const int T = sg.T, nl = (int) lanes.size();
    const float eps = m_.eps;
    // FNX_LANE_ATT: the attention layers' per-row steps (indexer rows, norms and rotary, K/V append) as one launch a step over
    // all the lanes' rows (the lanes' pointers from lane_rows), not a launch a lane. On for 4 AMD cards in a FAST build (measured there: same
    // bits, a 16-row window of 8 lanes 28.4 -> 27.7 ms); the other builds keep the launches a lane until they are measured
#if FNX_HIP
    const long long lane_att_default = FAST && NCARD == 4 ? 1 : 0;
#else
    const long long lane_att_default = 0;
#endif
    static const bool lane_att = env_int("FNX_LANE_ATT", lane_att_default) != 0;
    int call = 0;
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    for (int l = 0; l < NL; ++l) {
        each([&](int c, CardBuf& b, const CardW& w) {
            const LayerW& L = w.L[l];
            bool apply = l > 0;
            if (l == PLE_LAYER) {
                sum_flush(b);
                hc_write(b.R, b.bo_f, b.inj_f, T, b.st);
                apply = false;
                quantize_q8_1(b.ple_emb, b.xq, N, T, b.st);
                dense(w.ple_key, b.xq, b.ple_key, b, T);
                dense(w.ple_value, b.xq, b.ple_val, b, T);
                for (int i = 0; i < nl; ++i) {   // gated and normed rows past every key row (ple_key holds 3 MAXT rows)
                    const size_t r = (size_t) sg.r0[(size_t) i];
                    ple_post(b.R + r * D, b.ple_key + r * D, b.ple_val + r * N, w.ple_norm_key, w.ple_norm_query, w.ple_norm_conv,
                             w.ple_conv, b.lanes[(size_t) lanes[i]].ple_hist, b.ple_hsnap + r * 9 * D, counts[(size_t) i], eps, b.st,
                             b.ple_key + (size_t) MAXT * D);
                }
            }
            mall_pf_hc(b);
            if (!apply) sum_flush(b);
            const HcSumReq* const hsum = apply ? sum_take(b, b.bo_f, T) : nullptr;
            hc_read(b.R, apply ? b.bo_f : nullptr, b.inj_f, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed,
                    b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq
#if FNX_MALL_PF
                    ,
                    mall_pf_mark(b, w, L, PF_EV_HC0)
#endif
                    ,
                    hsum);
            if (!is_attn(l)) {
                const int nv = w.nv();
                dense_qkvz_ab(L, b, nv, T, batch_fuse());
                float** rec = b.lane_gdn[(size_t) T];
                for (const RowGroup& g : sg.groups) {
                    const GdnLanes gl{rec + (size_t) l * MAXT + g.s0, rec + (size_t) (NL + l) * MAXT + g.s0, b.seg_ctl[(size_t) T] + g.s0,
                                      b.seg_row0[(size_t) T] + g.s0, b.snap[l], b.hsnap[l]};
                    gdn_core_segs(b.qkvz, b.ab, L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.xq, b.gdn_scratch, w.nk, nv, w.vmap, gl, g.n,
                                  g.count, eps, b.st);
                }
                mall_pf_fork(b, w, L, 3);
                dense(L.ssm_out, b.xq, b.bo_a, b, T);
            } else {
                if (!L.idx_wq) dense(L.idx_qk, b.xq, b.qsa_y, b, T);
                else qsa_proj(L.idx_wq, L.idx_wk, b.mixed, b.qsa_y, T, b.st);
                dense_qkv(L, b, T, batch_fuse());
                const int mp = max_ctx_ / 4;
                const LaneRows lr{b.lane_rows[(size_t) T], b.lane_rows[(size_t) T] + MAXT};
                if (lane_att) {   // the per-row steps over all the lanes' rows at once: a launch a step, not one a lane
                    LaneRows la = lr;
                    la.tab = b.lane_rows[(size_t) T] + (size_t) 2 * MAXT;
                    la.ring = reinterpret_cast<uint16_t* const*>(const_cast<int**>(b.lane_rows[(size_t) T]) + (size_t) (3 + l) * MAXT);
                    qsa_index_lanes(b.qsa_y, L.idx_qn, L.idx_kn, b.qsa_pooled[l], b.pg, b.qsa_q, la, T, eps, rope_of(nullptr), b.st);
                    attn_prep_lanes(b.qg, b.k, L.q_norm, L.k_norm, NQH_LOC, KVH_LOC, la, T, eps, rope_of(nullptr), b.st);
                    kv_append_lanes(b.k, b.v, b.kc[l], b.vc[l], b.pg, b.att_ring.card, T, la, b.st);
                }
                for (int i = 0; i < (lane_att ? 0 : nl); ++i) {
                    const LaneDev& LD = b.lanes[(size_t) lanes[i]];
                    KvPages pg = b.pg;   // the card's page size and rows, the lane's table
                    pg.pt = LD.pt;
                    const size_t r = (size_t) sg.r0[(size_t) i];
                    const int k = counts[(size_t) i];
                    qsa_index(b.qsa_y + r * qsa_rows(), L.idx_qn, L.idx_kn, LD.ring[l], b.qsa_pooled[l], pg, b.qsa_q + r * 512, LD.ctl, k, eps,
                              rope_of(LD.mrope), b.st);
                    attn_prep(b.qg + r * QG_W, b.k + r * KV_W, L.q_norm, L.k_norm, NQH_LOC, KVH_LOC, LD.ctl, k, eps,
                              rope_of(LD.mrope), b.st);
                    kv_append(b.k + r * KV_W, b.v + r * KV_W, b.kc[l], b.vc[l], pg, b.att_ring.card, k, LD.ctl, b.st);
                }
                qsa_select_lanes(b.qsa_q, b.qsa_pooled[l], b.pg, b.qsa_scores, b.qsa_cells, b.qsa_ncells, mp, lr, T, b.att_ring.card, b.st);
                attn_part_lanes(b.qg, b.kc[l], b.vc[l], b.pg, lr, T, b.att_part, b.st, b.qsa_cells, b.qsa_ncells, b.att_ring.card);
                attn_finish(b.qg, b.att, b.att_ring, (const unsigned*) (b.ctl + 1), 128 + l, T, b.att_part, b.st, w.nq == NQH_LOC ? b.xq : nullptr);
                if (w.nq != NQH_LOC) quantize_q8_1(b.att, b.xq, w.nq * HD, T, b.st);
                mall_pf_fork(b, w, L, 3);
                dense(L.wo, b.xq, b.bo_a, b, T);
            }
            sum_later(b, b.bo_a, T, (const unsigned*) (b.ctl + 1), call);
        });
        ++call;
        each([&](int, CardBuf& b, const CardW& w) { ffn_half(T, call, b, w, w.L[l], b.R, b.ctl); });
        df2_tap_window(l, T);
        ++call;
    }
    each([&](int, CardBuf& b, const CardW& w) {
        mall_pf_hc(b);
        const HcSumReq* const hsum = sum_take(b, b.bo_f, T);
        hc_read(b.R, b.bo_f, b.inj_f, w.head_norm, w.head_down, w.head_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq
#if FNX_MALL_PF
                ,
                nullptr
#endif
                ,
                hsum);
        dense(w.out, b.xq, b.head_logits, b, T);
        argmax(b.head_logits, w.v1 - w.v0, T, w.v0, b.best_v, b.best_i, b.am_keys, b.ring.err, b.st);
        mall_pf_join(b);
    });
    sum_done(0, NCARD);
}

}  // namespace fnx

namespace fnx {

#if FNX_DF2
// DFlash2 (df2_bridge.hpp): the drafter's card taps the window's rows after the FFN half of the tapped layers, inside the window's graph
void Engine::df2_tap_window(int l, int T) {
    if (!df2_ || df2_slot_[(size_t) l] < 0) return;
    CardBuf& b = *cb_[0];
    ck(cudaSetDevice(b.dev), "set");
    sum_flush(b);
    df2::Bridge::tap(b.R, b.bo_f, b.inj_f, T, df2_slot_[(size_t) l], df2_tapw_, (int64_t) df2::kTaps * df2::kNEmbd, b.st);
}

void Engine::df2_attach(df2::Bridge& br) {
    if (br.device() != cb_[0]->dev) throw std::runtime_error("df2_attach: the drafter is not on the first card");
    if (br.lanes() < (int) lanes_.size()) throw std::runtime_error("df2_attach: fewer drafters than lanes");
    if (br.tap_layers().size() != (size_t) df2::kTaps) throw std::runtime_error("df2_attach: the pack taps " + std::to_string(br.tap_layers().size()) + " layers, the engine expects " + std::to_string(df2::kTaps));
    df2_slot_.assign((size_t) NL, -1);
    for (size_t i = 0; i < br.tap_layers().size(); ++i) {
        const int l = br.tap_layers()[i];
        if (l < 0 || l >= NL - 1) throw std::runtime_error("df2_attach: tapped layer " + std::to_string(l) + " is outside the model");
        df2_slot_[(size_t) l] = (int) i;
    }
    ck(cudaSetDevice(cb_[0]->dev), "set");
    df2_tapw_ = dalloc<float>((size_t) MAXT * df2::kTaps * df2::kNEmbd);
    df2_lane_.assign((size_t) br.lanes(), Df2Lane{});
    df2_ = &br;
}
#else
void Engine::df2_tap_window(int, int) {}
void Engine::df2_attach(df2::Bridge&) { throw std::runtime_error("this engine was built without DFlash2"); }
#endif

void Engine::mtp_attach(const Mtp& mtp) {
    if (mtp.cards.size() != (size_t) NCARD) throw std::runtime_error("mtp_attach: the draft layer is not on every card");
    mtp_ = &mtp;
    // draft_lanes' graphs and its row table hold the buffers made below: none of them outlives a new attachment
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        for (CardBuf::GraphEntry& e : b->dl_cache)
            if (e.exec) cudaGraphExecDestroy(e.exec);
        b->dl_cache.clear();
        for (CardBuf::GraphSlot& g : b->mtp_graphs) g.key.clear();   // (draft_rows' graphs too: the next call of each size captures again)
    }
    dl_tab_key_.clear();
    dl_seen_.clear();
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        const MtpCard& mc = mtp.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        b.mtp_io = dalloc<char>(draft_io::BYTES);
        ck(cudaMemset(b.mtp_io, 0, draft_io::BYTES), "draft block");
        b.mtp_pen_k = -1;
        b.mtp_emb = reinterpret_cast<float*>(b.mtp_io + draft_io::EMB);
        b.mtp_en = dalloc<float>((size_t) MAXT * N);
        b.mtp_e2 = dalloc<float>((size_t) MAXT * N);
        b.mtp_src = dalloc<float>((size_t) MAXT * D);
        b.mtp_hn = dalloc<float>((size_t) MAXT * D);
        b.mtp_h2 = dalloc<float>((size_t) MAXT * D);
        b.mtp_R = dalloc<float>((size_t) MAXT * D);
        b.mtp_kc = b.pool + (size_t) (KVP_SLOTS - 1) * kvp_slot(c);   // the draft layer: the pool's last slot
        b.mtp_vc = b.mtp_kc + kvp_rows_halves(c, kv_q8());
        b.lanes[0].mtp_ring = dalloc<uint16_t>((size_t) qsa_ring_rows() * 128);   // lanes after 0 carry theirs in their blocks
        b.mtp_qsa_ring = b.lanes[(size_t) cur_].mtp_ring;
        b.mtp_qsa_pooled = b.mtp_kc + 2 * kvp_rows_halves(c, kv_q8());
        b.mtp_logits = dalloc<float>((size_t) MAXT * std::max(1, mc.dv1 - mc.dv0));
        b.mtp_bv = reinterpret_cast<float*>(b.mtp_io + draft_io::BV);
        b.mtp_bi = reinterpret_cast<int*>(b.mtp_io + draft_io::BI);
        b.mtp_ctl = reinterpret_cast<int*>(b.mtp_io + draft_io::CTL);
        b.mtp_lctl = reinterpret_cast<int*>(b.mtp_io + draft_io::LCTL);
        b.mtp_row_pos = reinterpret_cast<int*>(b.mtp_io + draft_io::ROW_POS);
        b.mtp_lane_rows = dalloc<const int*>((size_t) LANE_ROW_COLS * MAXT);
        b.mtp_pen_l = dalloc<int>((size_t) MAXT * (1 + 2 * PEN_CAP));
        b.mtp_pen_par_l = reinterpret_cast<float*>(b.mtp_io + draft_io::PEN_PAR_L);
        b.mtp_tk_v_l = dalloc<float>((size_t) MAXT * TOPK_N);
        b.mtp_tk_i_l = dalloc<int>((size_t) MAXT * TOPK_N);
        b.mtp_tk_err_l = dalloc<unsigned>(MAXT);
        b.mtp_pen = dalloc<int>(1 + 2 * PEN_CAP);
        b.mtp_pen_par = reinterpret_cast<float*>(b.mtp_io + draft_io::PEN_PAR);
        b.mtp_tk_v = reinterpret_cast<float*>(b.mtp_io + draft_io::TK_V);
        b.mtp_tk_i = reinterpret_cast<int*>(b.mtp_io + draft_io::TK_I);
        b.mtp_tk_err = reinterpret_cast<unsigned*>(b.mtp_io + draft_io::TK_ERR);
        b.mtp_jobs = dalloc<CopyJob>(MAXT);
        b.mtp_last = dalloc<float>((size_t) MAXT * std::max(1, mc.dv1 - mc.dv0));
        b.mtp_zero_l = dalloc<int>(MAXT);
        ck(cudaMemset(b.mtp_zero_l, 0, (size_t) MAXT * 4), "mtp zero");
        b.mtp_zero = dalloc<int>(1);
        ck(cudaMemset(b.mtp_zero, 0, 4), "mtp zero");
    }
    if (!h_.mtp_emb) {
        h_.mtp_io = halloc<char>(draft_io::BYTES);
        std::memset(h_.mtp_io, 0, draft_io::BYTES);   // (its TK_ERR word stays zero: each upload clears the card's)
        h_.mtp_res = halloc<char>((size_t) NCARD * draft_io::RES_END);
        std::memset(h_.mtp_res, 0, (size_t) NCARD * draft_io::RES_END);
        h_.mtp_emb = reinterpret_cast<float*>(h_.mtp_io + draft_io::EMB);
        h_.mtp_jobs = halloc<CopyJob>((size_t) NCARD * MAXT);
        h_.mtp_ctl = reinterpret_cast<int*>(h_.mtp_io + draft_io::CTL);
        h_.mtp_lctl = reinterpret_cast<int*>(h_.mtp_io + draft_io::LCTL);
        h_.mtp_row_pos = reinterpret_cast<int*>(h_.mtp_io + draft_io::ROW_POS);
        h_.mtp_pen_l = halloc<int>((size_t) MAXT * NCARD * (1 + 2 * PEN_CAP));
        h_.mtp_pen_par_l = reinterpret_cast<float*>(h_.mtp_io + draft_io::PEN_PAR_L);
        h_.mtp_tk_v_l = halloc<float>((size_t) MAXT * NCARD * TOPK_N);
        h_.mtp_tk_i_l = halloc<int>((size_t) MAXT * NCARD * TOPK_N);
        h_.mtp_tk_err_l = halloc<unsigned>((size_t) MAXT * NCARD);
        h_.mtp_pen = halloc<int>((size_t) NCARD * (1 + 2 * PEN_CAP));
        h_.mtp_pen_par = reinterpret_cast<float*>(h_.mtp_io + draft_io::PEN_PAR);
        h_.mtp_tk_v = halloc<float>((size_t) NCARD * TOPK_N);
        h_.mtp_tk_i = halloc<int>((size_t) NCARD * TOPK_N);
        h_.mtp_tk_err = halloc<unsigned>(NCARD);
    }
    draft_index_.assign((size_t) m_.vocab, -1);
    for (size_t i = 0; i < mtp.vocab.size(); ++i) draft_index_[(size_t) mtp.vocab[i]] = (int) i;
    if (!h_.bv) {
        h_.bv = halloc<float>(NCARD * MAXT);
        h_.bi = halloc<int>(NCARD * MAXT);
        h_.err = halloc<unsigned>(NCARD);
    }
}

// one draft-layer window of T rows from b.mtp_src (rows of D) and b.mtp_emb (the tokens' embeddings), cells at
// mtp_ctl[0]: R = fc_hidden(rms(src) w) per stream + fc_embedding(rms(emb) w), then the layer as a main attention
// layer, then the final mixer and this card's draft head rows
void Engine::mtp_enqueue(int T) {
    const float eps = m_.eps;
    for (int c = 0; c < NCARD; ++c) cb_[c]->pend_on = false;   // (a capture that ended early leaves no sum behind)
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        const MtpCard& mc = mtp_->cards[c];
        const LayerW& L = mc.L;
        ck(cudaSetDevice(b.dev), "set");
        rms_norm_rows(b.mtp_emb, mc.norm_emb, b.mtp_en, T, N, eps, b.st);
        quantize_q8_1(b.mtp_en, b.xq, N, T, b.st);
        dense(mc.fc_emb, b.xq, b.mtp_e2, b, T);
        rms_norm_rows(b.mtp_src, mc.norm_hid, b.mtp_hn, T, D, eps, b.st);
        quantize_q8_1(b.mtp_hn, b.xq, N, HC * T, b.st);
        for (int r0 = 0; r0 < HC * T; r0 += MAXT)   // every stream of every row through fc_hidden
            q8s_gemv(mc.fc_hid.s, N, N, (const uint8_t*) b.xq + (size_t) r0 * q8_1_bytes(N), b.mtp_h2 + (size_t) r0 * N, N,
                     std::min(MAXT, HC * T - r0), b.st);
        add_streams(b.mtp_h2, b.mtp_e2, b.mtp_R, T, b.st);
        hc_read(b.mtp_R, nullptr, nullptr, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T,
                b.st, b.xq);
        attention(T, b, w, L, b.mtp_kc, b.mtp_vc, b.mtp_ctl, b.mtp_qsa_ring, b.mtp_qsa_pooled, 128);
        sum_later(b, b.bo_a, T, (const unsigned*) (b.mtp_ctl + 1), 0);   // (ffn_half's read takes it)
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        ck(cudaSetDevice(b.dev), "set");
        ffn_half(T, 1, b, m_.cards[c], mtp_->cards[c].L, b.mtp_R, b.mtp_ctl);
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const MtpCard& mc = mtp_->cards[c];
        ck(cudaSetDevice(b.dev), "set");
        const HcSumReq* const hsum = sum_take(b, b.bo_f, T);   // ffn_half's sum
        hc_read(b.mtp_R, b.bo_f, b.inj_f, mc.mix_norm, mc.mix_down, mc.mix_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq
#if FNX_MALL_PF
                ,
                nullptr
#endif
                ,
                hsum);
        dense(mc.head, b.xq, b.mtp_logits, b, T);
        penalize_row(b.mtp_logits + (size_t) (T - 1) * (mc.dv1 - mc.dv0), mc.dv1 - mc.dv0, b.mtp_pen, b.mtp_pen_par, b.st);
        argmax(b.mtp_logits, mc.dv1 - mc.dv0, T, mc.dv0, b.mtp_bv, b.mtp_bi, b.am_keys, b.ring.err, b.st);
    }
    sum_done(0, NCARD);
}


// The draft layer for several lanes in ONE window (Engine::draft_lanes): mtp_enqueue's layer, with the attention of enqueue_rows' batched windows for the draft
// layer's own K/V slot (each lane's rows at its cells on its pages). Greedy: no penalties, argmax of every row.
void Engine::mtp_enqueue_lanes(const std::vector<DraftReq>& reqs, int T, int only, int form) {
    const float eps = m_.eps;
    const int nl = (int) reqs.size(), mp = max_ctx_ / 4;
    std::vector<int> off((size_t) nl);
    for (int i = 0, o = 0; i < nl; ++i) off[(size_t) i] = o, o += reqs[(size_t) i].T;
    // (FNX_DRAFT_THREADS): only >= 0 queues that card's launches alone, all three parts in order, on the calling thread's device (the card
    // pool's thread of that card); only < 0: every card, part by part, as published
    const int c_lo = only < 0 ? 0 : only, c_hi = only < 0 ? NCARD : only + 1;
    for (int c = c_lo; c < c_hi; ++c) cb_[c]->pend_on = false;   // (a capture that ended early leaves no sum behind; only this call's cards)
    // FNX_DRAFT_LANES_TAB (draft_lanes_tab()): no launch argument depends on which lanes the rows are of or on where a lane's rows
    // start: the launches depend on the number of lanes, the rows, whether any lane penalizes and whether any takes candidates
    const bool tab = form < 0 ? draft_lanes_tab() : form != 0;
    bool any_pen = false, any_cands = false;
    for (const DraftReq& q : reqs) any_pen = any_pen || q.pen, any_cands = any_cands || q.cands;
    for (int c = c_lo; c < c_hi; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        const MtpCard& mc = mtp_->cards[c];
        const LayerW& L = mc.L;
        if (only < 0) ck(cudaSetDevice(b.dev), "set");
        if (tab) copy_many(b.mtp_jobs, nl, b.st);   // each lane's source rows into mtp_src (the jobs: draft_lanes' upload)
        rms_norm_rows(b.mtp_emb, mc.norm_emb, b.mtp_en, T, N, eps, b.st);
        quantize_q8_1(b.mtp_en, b.xq, N, T, b.st);
        dense(mc.fc_emb, b.xq, b.mtp_e2, b, T);
        rms_norm_rows(b.mtp_src, mc.norm_hid, b.mtp_hn, T, D, eps, b.st);
        quantize_q8_1(b.mtp_hn, b.xq, N, HC * T, b.st);
        for (int r0 = 0; r0 < HC * T; r0 += MAXT)
            q8s_gemv(mc.fc_hid.s, N, N, (const uint8_t*) b.xq + (size_t) r0 * q8_1_bytes(N), b.mtp_h2 + (size_t) r0 * N, N, std::min(MAXT, HC * T - r0), b.st);
        add_streams(b.mtp_h2, b.mtp_e2, b.mtp_R, T, b.st);
        hc_read(b.mtp_R, nullptr, nullptr, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        // the attention as a batched window does it, on the draft layer's slot of the pool
        if (!L.idx_wq) dense(L.idx_qk, b.xq, b.qsa_y, b, T);
        else qsa_proj(L.idx_wq, L.idx_wk, b.mixed, b.qsa_y, T, b.st);
        dense_qkv(L, b, T, batch_fuse());
        if (tab) {   // the per-row steps over all the lanes' rows at once, as a batched window's attention layers (enqueue_rows, FNX_LANE_ATT):
            // row r at its own position word (mtp_row_pos + r), on its lane's page table, M-RoPE table and draft-layer ring (column 3)
            LaneRows la{b.mtp_lane_rows, b.mtp_lane_rows + MAXT};
            la.tab = b.mtp_lane_rows + (size_t) 2 * MAXT;
            la.ring = reinterpret_cast<uint16_t* const*>(const_cast<int**>(b.mtp_lane_rows) + (size_t) 3 * MAXT);
            qsa_index_lanes(b.qsa_y, L.idx_qn, L.idx_kn, b.mtp_qsa_pooled, b.pg, b.qsa_q, la, T, eps, rope_of(nullptr), b.st);
            attn_prep_lanes(b.qg, b.k, L.q_norm, L.k_norm, NQH_LOC, KVH_LOC, la, T, eps, rope_of(nullptr), b.st);
            kv_append_lanes(b.k, b.v, b.mtp_kc, b.mtp_vc, b.pg, b.att_ring.card, T, la, b.st);
        }
        for (int i = 0; i < (tab ? 0 : nl); ++i) {
            const LaneDev& LD = b.lanes[(size_t) reqs[(size_t) i].lane];
            KvPages pg = b.pg;
            pg.pt = LD.pt;
            const size_t r = (size_t) off[(size_t) i];
            const int k = reqs[(size_t) i].T;
            const int* lc = b.mtp_lctl + 2 * i;
            qsa_index(b.qsa_y + r * qsa_rows(), L.idx_qn, L.idx_kn, LD.mtp_ring, b.mtp_qsa_pooled, pg, b.qsa_q + r * 512, lc, k, eps, rope_of(LD.mrope), b.st);
            attn_prep(b.qg + r * QG_W, b.k + r * KV_W, L.q_norm, L.k_norm, NQH_LOC, KVH_LOC, lc, k, eps, rope_of(LD.mrope), b.st);
            kv_append(b.k + r * KV_W, b.v + r * KV_W, b.mtp_kc, b.mtp_vc, pg, b.att_ring.card, k, lc, b.st);
        }
        const LaneRows lr{b.mtp_lane_rows, b.mtp_lane_rows + MAXT};
        qsa_select_lanes(b.qsa_q, b.mtp_qsa_pooled, b.pg, b.qsa_scores, b.qsa_cells, b.qsa_ncells, mp, lr, T, b.att_ring.card, b.st);
        attn_part_lanes(b.qg, b.mtp_kc, b.mtp_vc, b.pg, lr, T, b.att_part, b.st, b.qsa_cells, b.qsa_ncells, b.att_ring.card);
        attn_finish(b.qg, b.att, b.att_ring, (const unsigned*) (b.mtp_ctl + 1), 128, T, b.att_part, b.st, w.nq == NQH_LOC ? b.xq : nullptr);
        if (w.nq != NQH_LOC) quantize_q8_1(b.att, b.xq, w.nq * HD, T, b.st);
        dense(L.wo, b.xq, b.bo_a, b, T);
        sum_later(b, b.bo_a, T, (const unsigned*) (b.mtp_ctl + 1), 0);   // (ffn_half's read takes it)
    }
    for (int c = c_lo; c < c_hi; ++c) {
        CardBuf& b = *cb_[c];
        if (only < 0) ck(cudaSetDevice(b.dev), "set");
        ffn_half(T, 1, b, m_.cards[c], mtp_->cards[c].L, b.mtp_R, b.mtp_ctl);
    }
    for (int c = c_lo; c < c_hi; ++c) {
        CardBuf& b = *cb_[c];
        const MtpCard& mc = mtp_->cards[c];
        if (only < 0) ck(cudaSetDevice(b.dev), "set");
        const HcSumReq* const hsum = sum_take(b, b.bo_f, T);   // ffn_half's sum
        hc_read(b.mtp_R, b.bo_f, b.inj_f, mc.mix_norm, mc.mix_down, mc.mix_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq
#if FNX_MALL_PF
                ,
                nullptr
#endif
                ,
                hsum);
        dense(mc.head, b.xq, b.mtp_logits, b, T);
        const int nv = mc.dv1 - mc.dv0;
        if (tab && any_pen)   // (the lanes' rows and which of them penalize: mtp_lctl's second words)
            penalize_lanes(b.mtp_logits, nv, b.mtp_pen_l, 1 + 2 * PEN_CAP, b.mtp_pen_par_l, b.mtp_lctl, nl, b.st);
        for (int i = 0; i < (tab ? 0 : nl); ++i)   // each lane's last row, penalized as draft() does before the argmax
            if (reqs[(size_t) i].pen)
                penalize_row(b.mtp_logits + (size_t) (off[(size_t) i] + reqs[(size_t) i].T - 1) * nv, nv, b.mtp_pen_l + (size_t) i * (1 + 2 * PEN_CAP),
                             b.mtp_pen_par_l + 3 * i, b.st);
        argmax(b.mtp_logits, mc.dv1 - mc.dv0, T, mc.dv0, b.mtp_bv, b.mtp_bi, b.am_keys, b.ring.err, b.st);
        if (tab && any_cands && nv > 0) {
            // the candidates: the lanes' last rows gathered by lane, then one launch with a block a lane (the kernel of the launch a lane; a
            // lane that takes no candidates gets them too, unread). Its error bit (a NaN or inf in a row) goes to the first lane's word for
            // all: argmax has set the window's error bit for such a row already, so the call throws either way
            gather_lanes(b.mtp_logits, nv, b.mtp_lctl, b.mtp_last, nl, b.st);
            ck(cudaMemsetAsync(b.mtp_tk_err_l, 0, (size_t) nl * 4, b.st), "draft top err");
            fnx::top_tokens(b.mtp_last, nv, nl, mc.dv0, b.mtp_zero_l, b.mtp_zero_l, b.mtp_zero_l, 0, b.mtp_pen_par_l, b.mtp_tk_v_l, b.mtp_tk_i_l,
                            b.mtp_tk_err_l, b.st);
        }
        for (int i = 0; i < (tab ? 0 : nl); ++i)   // and the candidates of the lanes that sample their drafts
            if (reqs[(size_t) i].cands && nv > 0) {
                ck(cudaMemsetAsync(b.mtp_tk_err_l + i, 0, 4, b.st), "draft top err");
                fnx::top_tokens(b.mtp_logits + (size_t) (off[(size_t) i] + reqs[(size_t) i].T - 1) * nv, nv, 1, mc.dv0, b.mtp_zero, b.mtp_zero, b.mtp_zero, 0,
                                b.mtp_pen_par_l + 3 * i, b.mtp_tk_v_l + (size_t) i * TOPK_N, b.mtp_tk_i_l + (size_t) i * TOPK_N, b.mtp_tk_err_l + i, b.st);
            }
    }
    sum_done(c_lo, c_hi);
}

}  // namespace fnx
