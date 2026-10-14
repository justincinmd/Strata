// src/program/batch_groups_test.cpp - how many pipelined slot groups --batch runs with (header-only, CPU).
//
//   1. `auto` picks the most groups, at most one per stage, that divide the slots;
//   2. a layer split with --batch >= 2 and no --batch-groups means `auto` (0.1.41);
//   3. --batch-mtp (or --adapt-async 1) turns that default into one group, and says which feature did it;
//   4. a number or `auto` given explicitly is honoured, with or without those features;
//   5. no layer split: the number given (1) stays;
//   6. --lanes 2 turns the default into one group and is named before --batch-mtp / --adapt-async;
//   7. a lane's MTP window cap is the solo decode's (--mtp-max-t 1 included), its captured sizes at least 2.
#include "strata/program/batch_groups.hpp"

#include <algorithm>
#include <cstdio>

namespace bg = strata::program::batch_groups;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
bg::Input in(int batch, int stages) {
    bg::Input i;
    i.batch = batch;
    i.later_stages = stages;
    return i;
}
}  // namespace

int main() {
    std::printf("batch_groups_test\n");
    check(bg::auto_groups(0, 4) == 1, "auto: no layer split, one group");
    check(bg::auto_groups(1, 2) == 2 && bg::auto_groups(1, 4) == 2 && bg::auto_groups(1, 3) == 1,
          "auto: two GPUs, 2 groups when the slots divide by 2");
    check(bg::auto_groups(3, 8) == 4 && bg::auto_groups(3, 6) == 3 && bg::auto_groups(3, 2) == 2,
          "auto: at most one group per GPU, and a divisor of the slots");

    {   // the 0.1.41 default
        const auto r = bg::resolve(in(2, 1));
        check(r.groups == 2 && r.from_auto && r.serial == bg::Serial::None, "layer split, --batch 2, nothing given: 2 groups");
    }
    {
        const auto r = bg::resolve(in(1, 1));
        check(r.groups == 1 && !r.from_auto, "--batch 1: no default");
    }
    {
        const auto r = bg::resolve(in(2, 0));
        check(r.groups == 1 && !r.from_auto, "no layer split: one group");
    }

    {   // the production shape: parallel 2 on a 2-GPU split with --batch-mtp
        bg::Input i = in(2, 1);
        i.batch_mtp = true;
        const auto r = bg::resolve(i);
        check(r.groups == 1 && r.from_auto && r.serial == bg::Serial::BatchMtp && r.would_be == 2,
              "--batch-mtp, nothing given: one group, because of --batch-mtp");
        check(bg::serial_what(r.serial)[0] == '-', "the reason names the feature");
    }
    {
        bg::Input i = in(2, 1);
        i.adapt_async = true;
        const auto r = bg::resolve(i);
        check(r.groups == 1 && r.serial == bg::Serial::AdaptAsync, "--adapt-async 1, nothing given: one group");
    }
    {
        bg::Input i = in(2, 1);
        i.batch_mtp = true;
        i.adapt_async = true;
        check(bg::resolve(i).serial == bg::Serial::BatchMtp, "both: --batch-mtp is named");
    }
    {   // --lanes 2 (two lanes on a two-GPU split): the default gives way to it
        bg::Input i = in(2, 1);
        i.lanes = true;
        const auto r = bg::resolve(i);
        check(r.groups == 1 && r.from_auto && r.serial == bg::Serial::Lanes && r.would_be == 2,
              "--lanes 2, nothing given: one group, because of --lanes");
        i.adapt_async = true;
        check(bg::resolve(i).serial == bg::Serial::Lanes, "--lanes 2 with --adapt-async 1: --lanes is named");
    }
    {   // --lanes 2: the window cap is the solo serve loop's S_mtp (generate.cpp), --mtp-max-t 1 is honoured
        auto solo = [](int spec, int mtp_max_t) { return mtp_max_t > 0 ? std::min(mtp_max_t, spec) : spec; };
        bool same = true;
        for (int spec = 2; spec <= 8; ++spec)
            for (int m = 0; m <= 10; ++m) same = same && bg::lane_windows(spec, m).cap == solo(spec, m);
        check(same, "lanes: cap == the solo S_mtp for --spec 2..8, --mtp-max-t 0..10");
        auto w = bg::lane_windows(4, 0);
        check(w.cap == 4 && w.capture == 4, "lanes: --spec 4, no --mtp-max-t: 4-token windows, 4 captured");
        w = bg::lane_windows(4, 1);
        check(w.cap == 1 && w.capture == 2, "lanes: --spec 4 --mtp-max-t 1: one-token windows, 2 captured");
        w = bg::lane_windows(4, 2);
        check(w.cap == 2 && w.capture == 2, "lanes: --spec 4 --mtp-max-t 2: 2-token windows");
        w = bg::lane_windows(4, 3);
        check(w.cap == 3 && w.capture == 3, "lanes: --mtp-max-t below --spec: its value");
        w = bg::lane_windows(4, 4);
        check(w.cap == 4 && w.capture == 4, "lanes: --mtp-max-t equal to --spec: --spec");
        w = bg::lane_windows(4, 6);
        check(w.cap == 4 && w.capture == 4, "lanes: --mtp-max-t above --spec: --spec");
        w = bg::lane_windows(2, 1);
        check(w.cap == 1 && w.capture == 2, "lanes: --spec 2 --mtp-max-t 1: one-token windows, 2 captured");
    }
    {   // nothing to give way: the slots do not divide, so auto is one group anyway and there is nothing to say
        bg::Input i = in(3, 1);
        i.batch_mtp = true;
        const auto r = bg::resolve(i);
        check(r.groups == 1 && r.serial == bg::Serial::None && r.would_be == 1, "auto would be 1 group anyway: no reason");
    }
    {   // explicit values are honoured
        bg::Input i = in(4, 1);
        i.set = true;
        i.given = 2;
        i.batch_mtp = true;
        i.adapt_async = true;
        const auto r = bg::resolve(i);
        check(r.groups == 2 && !r.from_auto && r.serial == bg::Serial::None, "--batch-groups 2 with --batch-mtp: 2");
        i.given = 1;
        check(bg::resolve(i).groups == 1, "--batch-groups 1: 1 (the opt-out)");
        i.given = 4;   // not checked here: generate.cpp refuses what does not divide
        check(bg::resolve(i).groups == 4, "--batch-groups 4 is returned as given");
    }
    {
        bg::Input i = in(2, 1);
        i.set = true;
        i.auto_given = true;
        i.batch_mtp = true;
        const auto r = bg::resolve(i);
        check(r.groups == 2 && r.from_auto && r.serial == bg::Serial::None,
              "--batch-groups auto with --batch-mtp: auto asked for the pipeline, 2 groups");
    }
    {
        bg::Input i = in(2, 0);
        i.set = true;
        i.given = 2;
        check(bg::resolve(i).groups == 2, "an explicit number without a split is returned (the engine warns)");
    }

    std::printf("%s\n", g_fail == 0 ? "batch_groups_test: all passed" : "batch_groups_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
