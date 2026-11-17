// fnx kernels: launch wrappers. Activations are f32 rows of `n` values, one row per token (or per routed pair);
// quantized weights are raw ggml blocks, one weight row per output.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "model.hpp"
#include "rope.hpp"  // Rope, rope_pos, rope_unit_freq

// the cache prefetch (build option STRATA_SPLIT_FAST); see PfSet below
#ifndef FNX_MALL_PF
#define FNX_MALL_PF 0
#endif

namespace fnx {

// one matrix (or one expert's matrix) applied to up to MAXT activation rows
struct Job {
    int w;              // matrix index: an expert, or 0
    int n;              // rows of the job
    int y[MAXT];        // activation rows (q8_1)
    int out[MAXT];      // output rows
};

// the prompt path's expert jobs: up to PF_JOB rows of one expert (64: four 16-token tiles per job; 128: eight, the expert's weight tile is then
// dequantized once for twice the tokens; a build option, STRATA_SPLIT_PF_JOB)
#ifndef FNX_PF_JOB
#define FNX_PF_JOB 64
#endif
constexpr int PF_JOB = FNX_PF_JOB;
static_assert(PF_JOB == 64 || PF_JOB == 128, "PF_JOB: 64 or 128");
struct BigJob {
    int w;
    int n;
    int y[PF_JOB];
    int out[PF_JOB];
};

constexpr size_t q8_1_bytes(int n) { return (size_t) (n / 32) * 36; }

void quantize_q8_1(const float* x, void* q, int n, int rows, void* st);
// registers the split twin of a q8_1 activation buffer [q8, q8 + bytes): row r of n columns has its values at
// tq + r n and its scales at td + r n / 32. Writers into the buffer fill the twin too; the dense kernel reads it
void q8_twin(const void* q8, size_t bytes, int8_t* tq, float* td);
// forgets q8's twin: with the buffers it names when they are freed (the registry outlives an engine: a stale entry
// would hand the next engine's buffer at the same address a freed twin)
void q8_untwin(const void* q8);
// out[out_row][r] = W[job.w][r] . y[y_row] for every job in jobs[0..*njobs) (or njobs_host when njobs is null)
void mmvq(const DW& W, size_t w_stride, const Job* jobs, const int* njobs, int njobs_max, const void* y,
          float* out, int out_stride, void* st);
// out[t][r] = W[r] . y[t] for t < T: W in the split Q8_0 layout [rows][cols], y q8_1 rows of cols
void q8s_gemv(const Q8S& w, int rows, int cols, const void* y, float* out, int out_stride, int T, void* st);
// y[t][r] = W[r] . x[t] (f32 weights [rows][n], f32 x)
void gemv_f32(const float* W, const float* x, float* y, int rows, int n, int T, void* st);

// the hyper-connection read (build_hc_mix) with the previous half's write folded in when bo_prev is not null:
// R' = R + bo_prev * 2 sigmoid(inj_prev / HC); R <- R'; mixed = mean_c xn_c * sigmoid(up . silu(down . xn / HC));
// inject = w_inj . xn (when w_inj); mixed also as q8_1 rows into q8 when not null. scratch: hc_scratch_floats(MAXT)
// floats, zeroed once (its last float is the read's grid barrier counter). One launch of a block per SM (up to
// hc_blocks()) that needs every one of them resident: nothing else may run beside it. err: bit 4 set when its barrier
// timed out.
size_t hc_scratch_floats(int T);
int hc_blocks();    // the read's blocks of work; a card with fewer SMs runs them on one block per SM
#if FNX_KBENCH
void hc_set_stub(int bits);   // kbench only (hc.cuh, HC_STUB)
#endif
// sizes the read's grid to the current card (before its first hc_read or hc_preset_barrier) and zeroes the barrier
// counter of scratch, the card's one hc scratch (the counter counts in generations of the grid it ran with); returns the
// grid. cap > 0: at most cap blocks, as a card with that many SMs would run (kbench)
int hc_prepare(float* scratch, int cap = 0);
int hc_grid_for(int T);            // the grid (blocks) a read of T rows takes on the current card
int hc1_rows();   // FNX_HC1: windows of up to this many rows take the read's one-barrier form (read once; throws if bad)
struct HcSumReq;   // FNX_HC_SUM (below, after Ring): the window's sum of bo_prev, made by this read's kernel or by a launch before it
void hc_read(float* R, const float* bo_prev, const float* inj_prev, const float* w_norm, const Q8S& w_down,
             const Q8S& w_up, const uint16_t* w_inj, float eps, float* mixed, float* inject, float* scratch, unsigned* err,
             int T, void* st, void* q8 = nullptr
#if FNX_MALL_PF
             ,
             unsigned* pf_word = nullptr   // the persistent prefetcher's count (PfRun words[0]): this read is an event, it adds one
#endif
             ,
             const HcSumReq* sum = nullptr);
// sets the barrier counter in an hc scratch so that it wraps after barriers_left more barriers (2 per read): a test
// of the wrap (FNX_HC_BAR_WRAP)
void hc_preset_barrier(float* scratch, unsigned barriers_left);

// MoE routing: logits [T][NEXP] -> softmax, top-10, weights renormalized; jobs for the gate/up launch (rows = tokens,
// outputs = pairs t*TOPK+k) and for the down launch (rows = outputs = pairs), counts in njobs[0..1]
void route(const float* logits, int T, int* topi, float* topw, Job* jobs_gu, Job* jobs_down, int* njobs, void* st);
// h[r] = silu(g[r]) * u[r] quantized to q8_1, rows of n
void swiglu_q8_1(const float* g, const float* u, void* q, int n, int rows, void* st);
// out[t] = sum_k topw[t][k] * part[t*TOPK + k] + sigmoid(sh_logit[t]) * sh[t]
void moe_combine(const float* part, const float* topw, const float* sh, const float* sh_logit, float* out, int T, void* st);

// recurrent layer
void gdn_conv(const float* qkv, float* hist, const float* w, float* out, int ch, int T, void* st);
void gdn_qk_norm(float* conv, int nk, int ch, int T, void* st);
void gdn_gates(const float* alpha, const float* beta_raw, const float* dt_bias, const float* ssm_a, float* g, float* beta,
               int nv, int T, void* st);
void gdn_step(float* S, const float* conv, const float* g, const float* beta, float* o, int nk, int nv, int ch, int T, void* st);
void gdn_out_norm(const float* o, const float* z, const float* w, float* y, int nv, int T, float eps, void* st);
// the whole core in one launch: qkvz rows [q|k|v|z], ab rows [alpha|beta]; hist [2][3][ch] by window parity (ctl[1]);
// writes the gated, normed output as q8_1 rows of nv*HS. scratch: gdn_scratch_floats(nv) floats, zeroed once (the
// per-head counters in it are left at zero). For cutting a window back: snap [MAXT - 1][nv][HS][HS] gets the state
// after each row but the last, hsnap [MAXT][3][ch] the history left by keeping only the first n rows (at n - 1)
size_t gdn_scratch_floats(int nv);
void gdn_core(const float* qkvz, const float* ab, float* hist, const float* convw, const float* dt, const float* a,
              const float* nw, float* S, void* yq, const int* ctl, float* scratch, float* snap, float* hsnap, int nk, int nv,
              const int* vmap, int T, float eps, void* st, float* y32 = nullptr);
// a batched window's T rows, one row each from a lane of its own: device arrays of row r's lane's state, conv history
// (this layer's) and ctl. With row0: segments of several rows of one lane, indexed by segment, segment s's rows from
// window row row0[s] (device), its snapshots (snap and hsnap as gdn_core's, this layer's) from that row on
struct GdnLanes {
    float* const* S = nullptr;
    float* const* hist = nullptr;
    const int* const* ctl = nullptr;
    const int* row0 = nullptr;
    float* snap = nullptr;
    float* hsnap = nullptr;
};
// gdn_core over those rows at once (qkvz, ab and yq rows by window row; no cut-back snapshots: one row per lane)
void gdn_core_lanes(const float* qkvz, const float* ab, const float* convw, const float* dt, const float* a, const float* nw, void* yq,
                    float* scratch, int nk, int nv, const int* vmap, const GdnLanes& gl, int T, float eps, void* st);
// gdn_core over nseg segments of T rows each (gl with row0, snap and hsnap): each segment's rows chain through its lane's
// state as a window of T rows on that lane alone, its snapshots written for a cut-back
void gdn_core_segs(const float* qkvz, const float* ab, const float* convw, const float* dt, const float* a, const float* nw, void* yq,
                   float* scratch, int nk, int nv, const int* vmap, const GdnLanes& gl, int nseg, int T, float eps, void* st);
// the same over a prompt chunk of C rows (y32 only, no snapshots): ctl is the chunk's (position, tag) list, its ng groups
// of MAXT rows from ctl[2]; xc (C rows of ch), sq (C x nv x 4) and chw (gdn_prompt_work_floats: the sub-chunk path's
// state-free parts, FNX_PF_GDN_CHUNK) are scratch
// the file's routed-expert quantization decides the prompt GDN scan: the tensor-core scan is OFF for the IQ3_S-type sets (their suite lost 3 agent tasks with it, UD-Q4 did not) and ON
// for the others; FNX_PFX_GDN_TC=0/1 overrides. Set once by Model::Model.
void gdn_tc_set_default(bool on);
// the same for the int8 prompt attention (FNX_PFX_ATTN_I8): ON for the IQ3_S-type files (their suite shows no loss), OFF for the others (UD-Q4_K_XL lost agent-A8 and A10 with it, 11-03); the variable overrides. Defined in kernels part 4, read by the attention part.
void attn_i8_set_default(bool on);
bool attn_i8_default();
void gdn_prompt(const float* qkvz, const float* ab, float* hist, const float* convw, const float* dt, const float* a,
                const float* nw, float* S, const int* ctl, int ng, int nk, int nv, const int* vmap, int C, float eps, float* xc,
                float* sq, float* y32, float* chw, void* st);
size_t gdn_prompt_work_floats(int C, int nv);


// attention: per-head RMS norms and NEOX rotary on the first NROT dims, then causal attention over [0, pos+t]
// pos0: the window's first position, in device memory (graphs replay with the value it holds at launch)
void attn_prep(float* qg, float* k, const float* q_norm, const float* k_norm, int nq, int nkv, const int* pos0, int T,
               float eps, Rope base, void* st);
struct LaneRows;
// the same for a batched window's T rows of their own lanes (lr with tab): row t at its lane's position
void attn_prep_lanes(float* qg, float* k, const float* q_norm, const float* k_norm, int nq, int nkv, const LaneRows& lr, int T, float eps,
                     Rope base, void* st);
size_t attn_scratch_floats(int nq);
// The K/V cache is split by position: blocks of 4 positions dealt in turn to the cards (block j on card j % NCARD), so
// every card attends over 1 / NCARD of the positions. A card's local block of block j: its place among the card's
// blocks (j / NCARD); a position's local row: its local block * 4 + its place in the block.
// FNX_ATT_HEADS (model.hpp): by heads every block's rows (of the card's one KV head) are on every card
__host__ __device__ inline int kv_card(int j) { return j % NCARD; }
// block j's rows are on this card (by heads: every block)
__host__ __device__ inline bool kv_mine(int j, int card) { return ATT_HEADS || kv_card(j) == card; }
__host__ __device__ inline int kv_local_block(int j) { return ATT_HEADS ? j : j / NCARD; }
// position p's local row on its card(s)
__host__ __device__ inline int kv_local_row(int p) { return kv_local_block(p / 4) * 4 + p % 4; }
// The paged K/V pool (per card): a page holds 192 positions (48 blocks, 48 / NCARD on each card: this card's kvp_rows local rows)
// for every attention layer and the draft layer: per layer its K rows, V rows (NKVH * HD f16 each) and the 48 pooled
// QSA keys (128 f16 each; every card holds every block's). A sequence's page table maps its logical pages to
// physical ones; a layer's K, V and pooled pointers are the pool plus their offsets in a page, and an index is placed
// by page: K/V local row r at page pt[r / rows], pooled block b at page pt[b / 48].
// int8 K/V (the default; FNX_KV8=0: f16): the K and V rows int8 in groups of 32 with an f16 scale each (llama.cpp's
// q8_0); a layer's slot holds its K rows (int8), their scales, the V rows, their scales, then the pooled keys (f16
// either way: QSA's block selection reads them). The pool then has as many more pages as fit in the f16 pool's bytes.
#if FNX_27B
// the 27B's pages hold S27_PP positions in S27_SLOTS slots, laid out as s27_attn.cuh says. The blocks and slots here are
// Flash-Next's, for the attention kernels of kernels.cu a 27B build compiles and never launches
constexpr int KVP_BLOCKS = 48, KVP_POS = S27_PP, KVP_SLOTS = 13;
#else
constexpr int KVP_BLOCKS = 48, KVP_POS = 192, KVP_SLOTS = 13;   // slots: 12 attention layers + the draft layer
#endif
static_assert(KVP_BLOCKS % NCARD == 0, "a page holds whole rounds of the cards");
constexpr float RING_MAX = 65504.0f;   // fp16's largest finite value: past it a sum between cards is out of range
constexpr int KV8_GROUP = 32;
bool kv_q8();
__host__ __device__ inline int kvp_rows(int) { return ATT_HEADS ? KVP_BLOCKS * 4 : KVP_BLOCKS / NCARD * 4; }
// halves of one layer's K (or V) rows in a page, with their scales when int8
inline size_t kvp_rows_halves(int card, bool q8) {
    return q8 ? (size_t) kvp_rows(card) * KV_W / 2 + (size_t) kvp_rows(card) * KV_W / KV8_GROUP
              : (size_t) kvp_rows(card) * KV_W;
}
inline size_t kvp_slot(int card, bool q8) { return 2 * kvp_rows_halves(card, q8) + (size_t) KVP_BLOCKS * 128; }   // halves
inline size_t kvp_page(int card, bool q8) { return KVP_SLOTS * kvp_slot(card, q8); }                            // halves
inline size_t kvp_slot(int card) { return kvp_slot(card, kv_q8()); }
inline size_t kvp_page(int card) { return kvp_page(card, kv_q8()); }
struct KvPages {
    const int* pt = nullptr;   // device: logical page -> physical page
    size_t page = 0;           // halves per page (kvp_page of the card)
    int rows = 0;              // local rows per page (kvp_rows of the card)
    int q8 = 0;                // the rows are int8 (kv_q8)
};
#if defined(__CUDACC__) || defined(__HIPCC__)
__device__ __forceinline__ size_t kvp_row(const KvPages& g, int r) {
    return (size_t) g.pt[r / g.rows] * g.page + (size_t) (r % g.rows) * KV_W;
}
// int8 rows: local row r's first byte from the layer's K (or V) pointer, and its first scale (halves from the
// pointer); a row's scales follow the layer's rows in the page
__device__ __forceinline__ size_t kvq_row(const KvPages& g, int r) {
    return (size_t) g.pt[r / g.rows] * g.page * 2 + (size_t) (r % g.rows) * KV_W;
}
__device__ __forceinline__ size_t kvq_scale(const KvPages& g, int r) {
    return (size_t) g.pt[r / g.rows] * g.page + (size_t) g.rows * KV_W / 2 + (size_t) (r % g.rows) * KV_W / KV8_GROUP;
}
__device__ __forceinline__ size_t kvp_block(const KvPages& g, int b) {
    return (size_t) g.pt[b / KVP_BLOCKS] * g.page + (size_t) (b % KVP_BLOCKS) * 128;
}
#endif
// a batched window's rows, each from a lane of its own: device arrays of a row's lane's ctl (its first word is the
// row's position) and page table. Empty (pos null): the rows are one sequence's, at *pos0 + t, on the KvPages' table
struct LaneRows {
    const int* const* pos = nullptr;
    const int* const* pt = nullptr;
    // for the per-row steps' lane forms (qsa_index_lanes, attn_prep_lanes): a row's lane's M-RoPE table (null: none) and
    // its raw-key ring of the layer
    const int* const* tab = nullptr;
    uint16_t* const* ring = nullptr;
};
#if defined(__CUDACC__) || defined(__HIPCC__)
__device__ __forceinline__ int row_pos(const LaneRows& lr, const int* pos0, int t) { return lr.pos ? *lr.pos[t] : *pos0 + t; }
__device__ __forceinline__ KvPages row_pages(KvPages g, const LaneRows& lr, int t) {
    if (lr.pt) g.pt = lr.pt[t];
    return g;
}
__device__ __forceinline__ Rope row_rope(Rope base, const LaneRows& lr, int t) {
    if (lr.tab) base.tab = lr.tab[t];
    return base;
}
#endif
// QSA (see qsa.cuh): the indexer's projection (from wq/wk on x, or y already holding it when wq is null), its query,
// the raw key into ring (QSA_RING x 128 f16), a completed block's pooled key into pooled (f16 rows of 128), and for
// rows seeing more than 512 blocks the scores and the selected positions: cells[t] (qsa_maxsel() each), n_cells[t]
// (-1: dense)
int qsa_maxsel();
int qsa_rows();
int qsa_ring_rows();
// the prompt path's rows per chunk, before any raw-key ring is allocated (the rings hold a chunk and a block back)
void set_prompt_rows(int rows);
// the indexer's per-row steps alone, for rows already projected (y): the query, the raw keys, the completed blocks
void qsa_index(const float* y, const float* qn, const float* kn, uint16_t* ring, uint16_t* pooled, const KvPages& g, float* q, const int* pos0, int T,
               float eps, Rope base, void* st);
// qsa_index for a batched window's T rows of their own lanes (lr with tab and ring; y, q by window row): the same values
// as a call per lane
void qsa_index_lanes(const float* y, const float* qn, const float* kn, uint16_t* pooled, const KvPages& g, float* q, const LaneRows& lr, int T,
                     float eps, Rope base, void* st);
// the scores and the selection of rows whose query is in q
void qsa_select(const float* q, const uint16_t* pooled, const KvPages& g, float* scores, int* cells, int* n_cells, int max_pools, const int* pos0, int T,
                int card, void* st);
// the same with the decode path's scores (qsa's), for a batched window's T rows of their own lanes (lr)
void qsa_select_lanes(const float* q, const uint16_t* pooled, const KvPages& g, float* scores, int* cells, int* n_cells, int max_pools,
                      const LaneRows& lr, int T, int card, void* st);
void qsa(const uint16_t* wq, const uint16_t* wk, const float* x, float* y, const float* qn, const float* kn, uint16_t* ring,
         uint16_t* pooled, const KvPages& g, float* q, float* scores, int* cells, int* n_cells, int max_pools, const int* pos0, int T, float eps,
         Rope base, int card, void* st);
// the merge of the cards' attention partials: this card's receive slots ([2 sets][NCARD - 1 steps][ATT_ROWS][ATT_OWN heads][HD + 2] fp32
// words) and the next card's
struct AttRing {
    int card = 0;
    void* in = nullptr;
    void* next_in = nullptr;
    void* peer_in[NCARD] = {};   // every card's `in` (the one-hop merge pushes each owner its partials: set when the engine links the cards)
    bool onehop = false;         // NCARD > 2: the partials go to their owners in one hop instead of NCARD - 1
    unsigned* err = nullptr;
};
size_t att_ring_words();
#if FNX_ATT_MM
// kbench: 1 / 0 turns FNX_ATT_MM's kernels on / off for the calls that follow (-1: the environment's FNX_ATT_MM decides, the default)
void att_mm_force(int on);
#if FNX_KBENCH
// the decode path's scores and selection alone, with (mm) or without FNX_ATT_MM's kernels (scores, mm = 2: the prompt path's k_idx_score_tc)
void kb_idx_score(const float* q, const uint16_t* pooled, const KvPages& g, float* scores, int max_pools, const int* pos0, int T, int mm, void* st);
void kb_idx_select(const float* scores, int* cells, int* n_cells, int max_pools, const int* pos0, int T, int card, int mm, void* st);
#endif
#endif
int attn_rows();   // the most rows one attn call takes
// attention over this card's positions (the K/V cache split by position: block j on card j % NCARD, local index
// (j / NCARD) * 4 + p % 4), every head (qg: [T][NQH][q | gate]); cells/n_cells: QSA's selection as this card's local
// indices (n_cells[t] = -1: every owned position up to the row's). The cards' partials are merged on the ring: out
// gets this card's NQH / NCARD heads (heads card * NQH / NCARD ..), gated
void attn(const float* qg, const uint16_t* kc, const uint16_t* vc, const KvPages& g, float* out, const AttRing& ar, const unsigned* tag, int call,
          const int* pos0, int T, float* scratch, void* st, const int* cells = nullptr, const int* n_cells = nullptr, void* xq = nullptr);
// attn in two parts, for a window of several sequences: attn_part_lanes the partials of its T rows, each at its lane's
// position on its lane's pages (lr; cells / n_cells and qg indexed by the row), attn_finish the merge across the cards
void attn_part_lanes(const float* qg, const uint16_t* kc, const uint16_t* vc, const KvPages& g, const LaneRows& lr, int T, float* scratch,
                     void* st, const int* cells, const int* n_cells, int card);
// xq (attn, attn_finish): the output also as q8_1 rows of NQH_LOC * HD values (a buffer with a twin): one launch with the combine where that exists
void attn_finish(const float* qg, float* out, const AttRing& ar, const unsigned* tag, int call, int T, float* scratch, void* st, void* xq = nullptr);
// the indexer's projection alone (qsa with wq set does it first)
void qsa_proj(const uint16_t* wq, const uint16_t* wk, const float* x, float* y, int T, void* st);
// the same over a prompt chunk of C rows at positions *pos0 + t, in two parts around an exchange the caller makes:
// attn_prompt leaves every head's partial in part, in blocks per owner card ([NCARD][C][NQH / NCARD heads][HD + 2]: block k for card
// k); attn_prompt_merge merges this card's block with the others' (recv: [NCARD - 1][C][NQH / NCARD][HD + 2], from card + 1,
// card + 2, ...) and gates them into out ([C][NQH / NCARD][HD])
size_t attn_prompt_block_floats(int C);   // one block of part (one slot of recv)
void attn_prompt(const float* qg, const uint16_t* kc, const uint16_t* vc, const KvPages& g, const int* pos0, const int* cells, const int* n_cells,
                 int card, int C, int r0, int T, float* part, void* st, int* uscr = nullptr);   // rows r0 .. r0 + T - 1 of the C
// FNX_PF_ATT_UNION (HIP, by heads; default 4, 0 = off): rows a group of the prompt attention's union walk; attn_prompt_union_ints: the scratch
// (uscr) a chunk of C rows needs for it (0 when off)
int attn_prompt_union_rows();
size_t attn_prompt_union_ints(int C);
void attn_prompt_merge(const float* own, const float* recv, const float* qg, float* out, int C, int card, void* st);
// the rows of the window's positions this card owns (block j on card j % NCARD) = k/v (both KV heads) rounded to f16, or
// quantized to int8 (g.q8)
void kv_append(const float* k, const float* v, uint16_t* kc, uint16_t* vc, const KvPages& g, int card, int T, const int* pos0, void* st);
// the same for a batched window's T rows of their own lanes (lr: each row's position and page table)
void kv_append_lanes(const float* k, const float* v, uint16_t* kc, uint16_t* vc, const KvPages& g, int card, int T, const LaneRows& lr, void* st);
// attn_prep (NQH, NKVH) and kv_append in one launch, the same values
void attn_prep_append(float* qg, float* k, const float* v, const float* q_norm, const float* k_norm, uint16_t* kc, uint16_t* vc,
                      const KvPages& g, int card, int T, const int* pos0, float eps, Rope base, void* st);
// cnt[e] += how many of the n picks in topi are expert e (FNX_ROUTE_COUNT's tallies)
void route_count(const int* topi, int n, unsigned* cnt, void* st);
// FNX_PREFETCH_PROBE: guess = the top 10 of T rows of NEXP logits; score: cnt[0..6] += the window's distinct experts
// picked, guessed, both, 1, and with slot_of (the layer's, or null) picked not in a slot, those guessed, guesses wasted
void probe_guess(const float* logits, int T, int* guess, void* st);
void probe_score(const int* topi, const int* guess, int T, const int* slot_of, unsigned long long* cnt, void* st);
// the n picks in order as layer << 16 | expert at trace[*len ..] (up to cap; *len counts on past it): FNX_ROUTE_TRACE
void route_trace(const int* topi, int n, int layer, unsigned* trace, unsigned long long* len, unsigned long long cap, void* st);
// owned positions of a card for a context of n_ctx (the cache's rows per card)
int kv_rows_per_card(int n_ctx);

// per-layer token embedding block (layer 1)
// hsnap [MAXT][9][D]: the history left by keeping only the first n rows (at n - 1), for cutting a window back
void ple_post(float* R, const float* key, const float* value, const float* nk, const float* nq, const float* nc,
              const float* convw, float* hist, float* hsnap, int T, float eps, void* st, float* scratch);   // scratch: 2 T D floats, apart from key

// the routed experts. router + route: logits rows of NEXP + 1 = W [NEXP + 1][N] . x (the router's rows, then the shared
// expert's gate), top-10 per token by logit, weights the selected softmax renormalized; jobs: the distinct experts,
// njobs[0] of them (njobs[1] is the block counter, left at zero). Output rows: pair t*TOPK+k (the shared expert's go to
// T*TOPK+t). With an ExpertCache (model.hpp) the same last block also plans the layer's slots (the jobs' experts not
// in a slot take one, CLOCK, the layer's tables updated) and writes list (1 + MAXT * TOPK ints: how many took a slot,
// then their ids) for moe_cache's copy.
// sh_gu (rows sh_rows < 4096): the shared expert's gate|up over the q8_1 rows sh_y into sh_out, in the same launch
void router_route(const float* W, const float* x, float* logits, int T, int* topi, float* topw, Job* jobs_gu, Job* jobs_dn,
                  int* njobs, void* st, const ExpertCache* ec = nullptr, int layer = -1, int* list = nullptr, const Q8S* sh_gu = nullptr,
                  int sh_rows = 0, const void* sh_y = nullptr, float* sh_out = nullptr, const uint16_t* Wbf = nullptr
#if FNX_MALL_PF
                  ,
                  unsigned* pf_word = nullptr,   // the persistent prefetcher's count (PfRun words[0]): this launch adds one,
                  int pf_late = 0                // at its start (0) or when its logits are done, right before the expert kernels (1)
#endif
);
// FNX_ROUTER_BF16 (build option STRATA_SPLIT_FAST, model.hpp): Wbf is W as bf16 in router_bf16_pack's order (a layer has it when every value
// of W is a bf16 value); router_route then reads Wbf in place of W by router_bf16_mode: FNX_ROUTER_BF16 = 0 the f32 table, 1 (default) bf16, 2 bf16
// with one warp a logit row. The logits, picks, weights and jobs are the same in every mode, bit for bit (but for the order of a job's rows, which
// no mode fixes). Without the build option the mode is 0 and Wbf is not read.
// router_bf16_force: kbench's switch in one process (-1: by the environment).
int router_bf16_mode();
void router_bf16_force(int mode);
// gu rows of 2*ffs: [gate | up] (A: per-expert stacked rows, stride sa). sh_down (Q4_K experts only): the shared
// expert's swiglu of sh_gu ([T][2 * sh_ffs]) and down (N rows) into sh_out (row stride N), in the same launch
void moe_gate_up(const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_max, const void* xq, float* gu, void* st,
                 const Q8S* sh_down = nullptr, int sh_ffs = 0, const float* sh_gu = nullptr, float* sh_out = nullptr);
// the same for jobs of up to MAXT rows each, njobs_cap of them at most (the prompt path's)
void moe_gate_up_jobs(const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_cap, const void* xq, float* gu, void* st);
// the same with BigJobs (njobs_cap of them at most)
void moe_gate_up_big(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const void* xq, float* gu, void* st);
void moe_down_big(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const float* gu, float* part, void* st);
// the routed experts on the tensor cores (moe_mma.cuh): gate|up from f16 activation rows [token][N]; down from h16, the
// f16 swiglu rows [pair][ff] that swiglu_h16 writes from gate|up's output
void moe_gate_up_mma(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* xh, float* gu, void* st);
void swiglu_h16(const float* gu, int ffs, int P, uint16_t* h16, void* st);
void moe_down_mma(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* h16, float* part, void* st);
// the same with the rows rounded to f16 (FNX_PF_PART16: the prompt combine reads half the bytes)
void moe_down_mma16(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* h16, uint16_t* part16, void* st);
// a decode window's layer with an ExpertCache (model.hpp): the experts router_route's plan listed copied in from RAM to
// their new slots by a grid of the SMs (wide 16-byte reads; the MoE kernels' own reads over the link are latency-bound).
// With jobs, the plan runs here first, a launch of its own (when router_route did not plan)
void moe_cache(int layer, size_t gu_stride, size_t dn_stride, const ExpertCache& ec, int* list, void* st, const Job* jobs = nullptr,
               const int* njobs = nullptr);
// pool p's listed experts (list: device, its count then the experts' ids, each with its slot in ec.slot_of already) from
// their host copies into their slots, one kernel: the arena's fill
void cache_copy(const ExpertCache& ec, int p, const int* list, void* st);
// FNX_ARENA: the arena pool's first `to` slots lent where `from` were (from < to: slots [from, to) lent, their experts
// pointed back at RAM; from > to: slots [to, from) given back empty); the pool's lent count and hand follow
void arena_slots(const ExpertCache& ec, int from, int to, void* st);
// part rows of N from silu(gate) * up of the gu rows
void moe_down(const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_max, const float* gu, float* part, void* st);
// T rows of [gate (ffs) | up (ffs)] -> silu(gate) * up as q8_1 rows of ffs (the shared expert)
void swiglu_stacked_q8_1(const float* gu, void* hq, int ffs, int T, void* st);
// time-out diagnostics. Every err pointer that a kernel with a wait gets (Ring::err, AttRing::err, HcArgs::err, the prompt path's err) is a
// block of ERR_WORDS words of device memory, zero at the start: word 0 holds the error bits as before; the words after it are written only in a time-out
// branch (the first time-out on the card writes the record; every one counts itself), but for DG_PROG, which block 0's first thread of each decode ring
// sum stores at its start while the card has no ring time-out. Nothing on a window's way without a time-out loads them. The record is kept small:
// the device code is inlined at every wait.
constexpr int ERR_WORDS = 32;
enum ErrDiag : int {
    DG_CLAIM = 1,    // 0: no record; else the first time-out on this card: DiagKind << 28 | source card << 24 | element (diag_claim)
    DG_TAG,          // what it waited for: the ring tag / the prompt ring's sequence / the grid barrier's generation start / the down tickets
    DG_SEEN_HI,      // what the word held then: a ring word's tag half / the flag / the barrier counter / the down tickets done
    DG_SEEN_LO,      // a ring word's payload half / the grid / the tickets taken
    DG_CYC_LO,       // clock64 cycles the wait had taken when it gave up (far over the time-out: the thread itself did not run for that long)
    DG_CYC_HI,
    DG_WALL_LO,      // the card's steady counter (wall_clock64) at the time-out
    DG_WALL_HI,
    DG_COUNT,        // the waits that timed out on this card (every one adds itself)
    DG_SRCMASK,      // bit s: a word of card s did not come (bit DG_NO_SRC: a wait with no source card)
    DG_PROG,         // the tag of the last decode ring sum this card entered before a ring time-out of its own
    DG_CAL0_CYC,     // diag_clock (at load, and at a report): clock64 cycles and steady ticks of one spin
    DG_CAL0_WALL,
    DG_NOW_LO,       // and the steady counter after the spin (the report dates the record with it)
    DG_NOW_HI,
    DG_WATCH_LO,     // written by the host at load: the address of the host's eviction watch word (0 0: off). The host samples the driver's evicted time
    DG_WATCH_HI,     // of this process and keeps in that word: bits 63..32 its clock in ms, bits 31..0 the clock at the last sample that saw it rise
    DG_EXT_N,        // waits (threads) on this card that began again at their time-out because the driver had stopped the process's queues just before
};
// A wait that reaches its time-out reads the watch word (diag.cuh diag_wait_on, in the time-out branch only). The driver stops all of a process's
// queues to move a page of its host memory (a KFD eviction) and starts them again card by card, so a wait can run out while its own card was
// stopped, or while it runs and the peer's card is still stopped. When the host saw evicted_ms rise at most WAIT_EVICT_WINDOW_MS plus about the
// time-out ago, the wait begins again. With no rise seen the wait gets WAIT_GRACE_CYCLES more, once (the host's sample of a card that came back this
// instant can be late), then fails as before. A wait that was extended by WAIT_EXT_CAP cycles or more fails at its next time-out, so a wait whose
// peer never comes ends, whatever the driver and the host's watch do.
constexpr unsigned WAIT_EVICT_WINDOW_MS = 250u;
constexpr long long WAIT_GRACE_CYCLES = 1LL << 28;       // 80 to 130 ms
constexpr long long WAIT_EXT_CAP = 200000000000LL;     // 60 to 100 s
constexpr unsigned DG_NO_SRC = 15u;
// the element: row * (N / 2) + pair of a decode sum; the block of an hc read or a chunk sum
constexpr unsigned diag_claim(unsigned kind, unsigned src, unsigned elem) { return kind << 28 | (src & 15u) << 24 | (elem & 0xffffffu); }
// DK_RING: the one-hop sums (every card reads every other card's word); DK_RING_CHAIN / DK_RING_F32: the ring that passes sums from card to card
enum DiagKind : unsigned { DK_RING = 1, DK_RING_F32, DK_ATT_MERGE, DK_CHUNK_RING, DK_PROMPT_SUM, DK_HC_BARRIER, DK_HC_GATE, DK_RING_CHAIN };
// the cycles a decode ring sum waits for a peer's word before it sets err bit 1 (Ring::timeout; FNX_RING_TIMEOUT_MS sets it per card from the card's measured
// clock rate). The legacy merges and the prompt path's waits keep this constant
constexpr long long RING_TIMEOUT_CYCLES = 4000000000LL;
// one spin on the current card: clock64 cycles into err[DG_CAL0_CYC], the steady counter's ticks into err[DG_CAL0_WALL], the steady counter after it into
// err[DG_NOW_LO / HI] (err: an err block); queued on st
void diag_clock(unsigned* err, void* st);
// test hooks (FNX_DIAG_STALL): a kernel that holds the stream for `ticks` of the steady counter; *p += add in stream order
void diag_spin(long long ticks, void* st);
void diag_poke(unsigned* p, unsigned add, void* st);

// the weighted sum of the expert rows, read by the ring in place of x when part is not null
struct RingMoe {
    const float* part = nullptr;
    const uint16_t* part16 = nullptr;   // prompt path, FNX_PF_PART16: the routed rows in f16 (part keeps the shared rows at C * TOPK * N)
    const float* topw = nullptr;
    const float* logits = nullptr;
};

// the sum across the cards (ring; words of two fp16 values and a tag); bufs are this card's receive buffers, peers'
// via P2P
struct Ring {
    int card;
    void* in;            // this card: [2 sets][NCARD - 1 steps][MAXT][N / NCARD + 1] u64
    void* ag;            // this card: [2 sets][MAXT][N] u64
    void* next_in;       // card (card + 1) % NCARD's
    void* next_ag;
    void* peer_in[NCARD] = {};   // every card's `in` (the one-hop sum reads each card's mailbox)
    bool onehop = false;         // NCARD > 2 and fp16 words: one hop for a window's sum (k_ring_a2a), the ring's 2 (NCARD - 1) hops otherwise
    // FNX_PEER (AMD, src/hip/peer.h): the one-hop sum pushed into the other cards' device memory instead of read from host mailboxes. pk: this card's mailbox
    // [2 sets][NCARD sources][MAXT * N / 2] tagged words (fine-grained device memory); peer_pk[d]: card d's
    void* pk = nullptr;
    void* peer_pk[NCARD] = {};
    bool peer = false;
    unsigned* err;       // bit 1: a peer's value never came; bit 2: a NaN or inf (on the fp16 ring: a value past its
                         // range); bit 4: an hc read's grid barrier timed out; bit 16: a NaN or inf among the head's
                         // logits (the window's argmax). A block of ERR_WORDS words (ErrDiag)
    bool f32 = false;    // fp32 words (buffers sized for them) instead of fp16 pairs
    int rows = MAXT;     // the row capacity the buffers were sized for
    // a ring for chunk-sized sums (plain fp16 words and a flag per block and hop): this card's flags and the next's
    unsigned* flags = nullptr;
    unsigned* next_flags = nullptr;
    long long timeout = RING_TIMEOUT_CYCLES;   // the decode sums' wait for a peer's word, in clock64 cycles
};
size_t ring_flag_words();
// the ring's buffers for a capacity of `rows` rows (fp32 words): words of in, of ag
size_t ring_in_words(int rows);
size_t ring_ag_words(int rows);
// the tag: *tag * 256 + call + 1, with *tag in device memory (one value per window) and call < 255 per window: a
// call of 255 or more would carry into the next tag's field and match that exchange's words (ring_call_check)
constexpr int RING_CALLS = 255;
// a window: two calls a layer (2 l, 2 l + 1); a prompt chunk the same on its own tag, and 128 + l on its groups' tags
static_assert(2 * NL <= 128 && 128 + NL <= RING_CALLS, "the ring calls of a window or a prompt chunk do not fit the tag's call field");
inline void ring_call_check(int call, const char* who) {
    if (call < 0 || call >= RING_CALLS)
        throw std::runtime_error(std::string(who) + ": ring call " + std::to_string(call) + " outside 0.." + std::to_string(RING_CALLS - 1));
}
// with m.part set, this card's addend is the experts' weighted sum (written into x with the result)
void ring_sum(const Ring& r, float* x, int T, const unsigned* tag, int call, void* st, const RingMoe& m = {});
// (FNX_HC_SUM): a ring_sum not yet launched (its arguments but the rows and the stream): hc_read makes it, in its own kernel where it can
struct HcSumReq {
    const Ring* ring = nullptr;
    float* x = nullptr;
    const unsigned* tag = nullptr;
    int call = 0;
    RingMoe m;
};

// R += bo * 2 sigmoid(inj / HC) per stream: a pending write applied on its own (before the PLE block)
void hc_write(float* R, const float* bo, const float* inj, int T, void* st);

// y[r] = x[r] / sqrt(mean(x[r]^2) + eps) * w for rows of n
void rms_norm_rows(const float* x, const float* w, float* y, int rows, int n, float eps, void* st);
// the draft layer's input: R[t][c] = h[t][c] + e[t] for each of the HC streams (h rows of D, e rows of N)
void add_streams(const float* h, const float* e, float* R, int T, void* st);

// pull [p, p + bytes) into L2
void l2_prefetch(const void* p, size_t bytes, void* st);
// n (1..3) Q8_0 matrices of `cols` columns over the same q8_1 rows y in one launch: matrix s, rows[s] rows, into
// out[s] (row stride out_stride[s]); each output as q8s_gemv of that matrix alone gives it
void q8s_gemv_segs(const Q8S* w, const int* rows, float* const* out, const int* out_stride, int n, int cols, const void* y, int T,
                   void* st);
// q8s_gemv(w, rows, cols, y, out, rows) and gemv_f32(W, x, yf, rf <= 160, N) in one launch, the same values; rows
// 4096..8191 only (q8s_gemv_segs: 8192 rows and more in all)
void q8s_gemv_f32(const Q8S& w, int rows, int cols, const void* y, float* out, const float* W, const float* x, float* yf, int rf, int T,
                  void* st);

// (FNX_MM, build option STRATA_SPLIT_APPROX; HIP on gfx12): the dense Q8 projections on the int8 matrix instruction (dense.cuh q8s_mm_block):
// q8s_gemv, q8s_gemv_segs and q8s_gemv_f32 take it for windows of FNX_MM_DENSE rows and more (default 0: never; 1: every window). Its sums are the same
// integers a 32-column block, added in another order than the dp4a kernels' (dense.cuh), so its outputs differ from theirs in the last bits.
// FNX_MM_HC (build option STRATA_SPLIT_APPROX): hc_read as a queue of tickets without grid barriers (hc_mm.cuh), taken unless FNX_MM_HC=0. Its sums
// are added in another order than k_hc's: its outputs differ from k_hc's in the last bits.
// mm_force: kbench's switch between the kernels in one process (-1: by the environment).
#ifndef FNX_MM
#define FNX_MM 0
#endif
#ifndef FNX_MM_HC
#define FNX_MM_HC 0
#endif
#if FNX_MM_HC && !FNX_MM
#error "FNX_MM_HC needs FNX_MM"
#endif
bool mm_dense(int T);            // does a window of T rows take the matrix kernel on the current card? (false without FNX_MM or off gfx12)
bool mm_hc();                    // does hc_read take the ticket queue on the current card? (false without FNX_MM_HC or off gfx12)
// generator (src/tk.hpp): the engine's start: the run's kernel variants are read and checked (a wrong name or a variant whose launch
// path is off in this run fails here)
void tk_start();
void mm_force(int dense, int hc);
int hc_mm_grid();                // the ticket queue's grid, FNX_MM_HC_GRID blocks (0 without FNX_MM_HC)
int hc_sum_rows();               // FNX_HC_SUM: the window's sum rides in the hc read that follows it, for windows of this many rows and fewer (8 in an APPROX build; 0: off)

// FNX_MALL_PF (build option STRATA_SPLIT_FAST): the next layer's fixed weights staged into the card's 64 MB Infinity Cache while
// the routed experts run. A set is up to MALL_PF_MAXR ranges in
// order of use, passed by value. mall_pf launches `wgs` blocks of 256 threads on st (a side stream): each lane walks the ranges in order, grid-striding,
// with plain loads of one word a `line` bytes (full: every 16 bytes, 8 loads in flight a lane). It writes nothing but, about never, one
// word at set.sink (a store the compiler cannot drop: without it the loads are dead code and the kernel is empty). Without the build
// option mall_pf_built() is false and mall_pf throws.
constexpr int MALL_PF_MAXR = 48;
struct PfSet {
    struct R {
        const void* base;
        long long bytes;
    } r[MALL_PF_MAXR];
    int n = 0, line = 128, full = 0, pad = 0;
    unsigned* sink = nullptr;   // 4 writable bytes on the card, the set's owner's (see k_mall_pf)
};
bool mall_pf_built();
void mall_pf(const PfSet& set, int wgs, void* st);
// The persistent form: one kernel a window on a side stream, launched before the window's graph and outside it (kernels.cu k_mall_pf_run). It follows
// the window by a count on the card: router_route's pf_word, one more a trunk layer. words (MALL_PF_WORDS unsigned on the card, zeroed):
//   [0] the fork points run so far   [1] status: bit 0 a prefetcher timed out, bit 1 the count was past its window (base is not this window's)
//   [2] the sink (see PfSet::sink)   [3] stop: not 0 ends a prefetcher within 64 polls (the host's, for the residency probe)
//   [4] the prefetcher blocks started so far (the probe waits for a prefetcher's blocks with it)
//   [5] sets staged, one a block (touched; not a proof that the touch ended before the layer read the set)
//   [6] blocks whose first sight of the count was their window's end or later (started too late)
//   [7] sets a block left out (the count had moved on by more than one, or the block ended early)
//   [8] sets a block began and left when a later event had run (PfRun::gate)       [9] sets with no range (of [5])
//   [10] of the sets of [8]: the staged share of each in 1/256, summed   [11] of the sets of [8]: the bytes left cold in units of 256 KiB, summed
//   (a report every 4096 windows stays under 2^32 in both with one set a layer that has ranges, 64 blocks and 40 MiB sets; with two such sets a
//   layer, or the 60 MiB cap, only up to ~32 blocks: the report's two averages can then be low, nothing else reads these words)
//   [5] + [7] + [8] = (n - 1 + wrap) sets a block
//   block 0's time account, units of 64 wall-clock ticks (4096 windows of a 500 ms time-out stay under 2^32): [12] in sets with ranges (from its
//   sight of the event to the set's end or the leave), [13] waits for the next event after a set staged to its end (spare time), [14] waits
//   after an empty set (the pause)
//   [15] block 0's position in its sets when it left them, in units of 64 KiB: an estimate of the card's bytes (every block takes a share of
//   each piece; with the gate another block can leave a set a piece before or after block 0)
//   (32-bit counts: a reader takes differences)
// Every word is read and written at device scope by kernels of the card (a host copy is not: mall_pf_poke / mall_pf_peek are the host's way in
// while a prefetcher runs).
constexpr int MALL_PF_WORDS = 16;
struct PfRun {
    const PfSet* sets = nullptr;   // on the card: a set a layer
    unsigned* words = nullptr;
    int n = 0, wrap = 0;           // the layers; wrap 1: after the last layer's fork point, layer 0's set
    unsigned base = 0;             // words[0] before this window's first fork point
    // The gate: 0 none: a block stages a set to its end, then looks at the count. Else the bytes of a set a block stages between two looks at the
    // count (a multiple of 4096, 65536 or more): when a later event has run, the block leaves the set there and goes on to the newest one. An
    // event with an empty set then is a pause: nothing is loaded until the next event (the engine: the event before a layer's expert kernels)
    int gate = 0;
    long long timeout = 0;         // wall-clock ticks (mall_pf_ticks) before it gives up
};
// query: a stream query after the launch (a test switch: by the runtime's source a launch is sent to the card at once and needs no flush)
void mall_pf_run(const PfRun& r, int wgs, void* st, bool query = false);
long long mall_pf_ticks(double ms);   // ms in the current card's wall-clock ticks
void mall_pf_table(PfSet* dev_table, const PfSet* sets, int n);   // n checked sets copied to the card's table (PfRun::sets)
// The persistent prefetcher's side stream on the current card. The runtime maps a card's streams onto at most GPU_MAX_HW_QUEUES (4) hardware queues
// a priority class, and a kernel launch waits for every earlier packet of its hardware queue (the AQL barrier bit): a stream that shared the
// prefetcher's queue would wait for the prefetcher's end with every launch, and the prefetcher for that stream's work. prio 1 (high) or -1 (low):
// a stream of that class, the only one, so a hardware queue of its own (*own_queue true); 0, or the class cannot be had: a default stream.
void* mall_pf_stream(int prio, bool* own_queue = nullptr);
// one of the words by a kernel on st, at device scope: poke writes val; peek copies the word to dev_out (4 bytes on the card, read after st's sync)
void mall_pf_poke(unsigned* words, int idx, unsigned val, void* st);
void mall_pf_peek(const unsigned* words, int idx, unsigned* dev_out, void* st);
// the hc read's grids in use, probed beside `wgs` resident prefetcher workgroups (kernels.cu): 0 resident, bit 0 / 1 a grid is not, bit 2 no answer
struct PfProbeInfo {
    int reads = 0;                               // reads of words[4] until every block had started (or 250 ms had gone)
    unsigned started = 0, status = 0;            // words[4] at the last read; words[1] after the stop
    double ms_up = 0, ms_probe = 0, ms_stop = 0; // the wait for the blocks, the grids' probes, the stop until the prefetcher's end
};
int mall_pf_hc_probe(float* hc_scratch, const PfSet* table, int n, unsigned* words, int wgs, void* side, void* work, PfProbeInfo* info = nullptr,
                     bool query = false);
// the probe's answer in words (after "beside N resident prefetch workgroups:")
std::string mall_pf_probe_text(int bad, const PfProbeInfo& info, int wgs);

// device-to-device copies in one launch: jobs (device memory) of 16-byte aligned pointers and lengths
struct CopyJob {
    void* dst;
    const void* src;
    size_t bytes;
};
void copy_many(const CopyJob* jobs, int n, void* st);

// argmax over this card's vocab slice: best[t] = (value, index + v0). keys: MAXT words, zeroed once (left at zero).
// err bit 16: a logit was NaN or inf
void argmax(const float* logits, int rows, int T, int v0, float* best_v, int* best_i, unsigned long long* keys, unsigned* err, void* st);
// the sampler's penalties on one row of n logits, in place, as top_tokens applies them: pen = {count k, k indexes into
// the row, their k counts}, par = {repeat, freq, present}
void penalize_row(float* row, int n, const int* pen, const float* par, void* st);
// a batched draft call's lanes from the card's table sel (lane i's word sel[2 * i + 1]: bits 0-15 its last row, bit 16: it
// penalizes): penalize_row for every lane that penalizes (pens: pen_stride ints a lane, pars: 3 floats a lane) in one launch, and each lane's
// last row copied to row i of out
void penalize_lanes(float* logits, int n, const int* pens, int pen_stride, const float* pars, const int* sel, int lanes, void* st);
void gather_lanes(const float* logits, int n, const int* sel, float* out, int lanes, void* st);
// the sampler's candidates: per row t < T of logits (n of this card's vocabulary from token v0), the row's penalties
// applied in place (pen_tok / pen_cnt: pen_n[t] tokens and their counts at row t * pen_stride), then its TOPK_N highest
// into out_v / out_i (logit descending, id ascending on ties, however many tie); err bit 16: a logit was NaN or inf
// before the penalties
constexpr int TOPK_N = 64;
void top_tokens(float* logits, int n, int T, int v0, const int* pen_tok, const int* pen_cnt, const int* pen_n, int pen_stride,
                const float* pen_par, float* out_v, int* out_i, unsigned* err, void* st);   // pen_par: per row repeat, freq, present

}  // namespace fnx
