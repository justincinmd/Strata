// include/strata/program/batch_groups.hpp - how many pipelined slot groups `--batch` runs with.
//
// `--batch-groups N` pipelines N groups of slots through the stages of a layer split; `--batch-groups auto` picks the
// most groups (at most one per stage) that divide the slots.  Since 0.1.41 a layer split with `--batch` >= 2 and no
// `--batch-groups` at all means `auto`.  The pipelined path does not run `--batch-mtp`'s drafts or the batch
// adaptive tier (`--adapt-async 1` between batch windows), so when one of those is asked for and the number of groups
// was NOT given, `auto` resolves to one group (the serial windows they run in) and says why.  A number or `auto`
// given on the command line is honoured as before.  `--lanes 2` replaces the batch windows altogether (each slot is a
// lane with its own verifiers and drafter); the default gives way to it the same way, and generate.cpp turns the lanes
// off when a number above 1 or `auto` was given.  `lane_windows` is the lanes' MTP window cap (the solo decode's).
//
// Pure (no GPU, no environment read): generate.cpp passes in what it knows.
#pragma once

#include <algorithm>

namespace strata::program::batch_groups {

/// What kept `auto` from pipelining.
enum class Serial { None, BatchMtp, AdaptAsync, Lanes };

struct Input {
    bool set = false;          ///< --batch-groups was given (a number or `auto`)
    bool auto_given = false;   ///< ... as `auto`
    int given = 1;             ///< the number given (1 when none)
    int batch = 0;             ///< the slots
    int later_stages = 0;      ///< the layer split's stages after CUDA0 (0: no layer split)
    bool batch_mtp = false;    ///< --batch-mtp is on (it passed the gates that do not concern the groups)
    bool adapt_async = false;  ///< --adapt-async 1 was asked for
    bool lanes = false;        ///< --lanes 2 is on (it passed its gates)
};

struct Result {
    int groups = 1;
    bool from_auto = false;            ///< the number came from `auto` (given or the 0.1.41 default)
    Serial serial = Serial::None;      ///< `auto` would have pipelined, but this feature needs the serial windows
    int would_be = 1;                  ///< what `auto` would have picked without it (== groups unless `serial`)
};

/// The most groups, at most one per stage of the layer split (`later_stages` + 1 GPUs), that divide `batch`.
inline int auto_groups(int later_stages, int batch) {
    int best = 1;
    if (later_stages <= 0) return best;
    for (int d = 2; d <= later_stages + 1 && d <= batch; ++d)
        if (batch % d == 0) best = d;
    return best;
}

inline Result resolve(const Input& in) {
    Result r;
    r.groups = in.given;
    r.would_be = in.given;
    const bool defaulted = !in.set && in.later_stages > 0 && in.batch > 1;   // the 0.1.41 default
    if (!in.auto_given && !defaulted) return r;
    r.from_auto = true;
    r.would_be = auto_groups(in.later_stages, in.batch);
    r.groups = r.would_be;
    if (!in.set && r.would_be > 1) {   // only the default gives way; `--batch-groups auto` asked for the pipeline
        if (in.lanes) r.serial = Serial::Lanes;
        else if (in.batch_mtp) r.serial = Serial::BatchMtp;
        else if (in.adapt_async) r.serial = Serial::AdaptAsync;
        if (r.serial != Serial::None) r.groups = 1;
    }
    return r;
}

/// --lanes 2: a lane's MTP windows.  `cap` is the solo decode's cap (its S_mtp: --mtp-max-t when given, at most --spec;
/// else --spec), so `--mtp-max-t 1` gives one-token windows (no drafts) in a lane as in the solo path.  `capture` is
/// how many window sizes the lane verifiers and drafters capture at start-up: at least 2, as before the cap could be 1
/// (the verifiers' allocation is sized from --spec / --batch at init, not from this).
struct LaneWindows {
    int cap = 1;
    int capture = 2;
};

inline LaneWindows lane_windows(int spec, int mtp_max_t) {
    LaneWindows w;
    w.cap = std::max(1, mtp_max_t > 0 ? std::min(mtp_max_t, spec) : spec);
    w.capture = std::max(2, w.cap);
    return w;
}

inline const char* serial_what(Serial s) {
    return s == Serial::BatchMtp ? "--batch-mtp" : s == Serial::AdaptAsync ? "--adapt-async 1"
         : s == Serial::Lanes ? "--lanes 2" : "";
}

}  // namespace strata::program::batch_groups
