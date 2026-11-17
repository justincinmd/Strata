// The queue budget (src/hip/queue_budget.hpp): GPU_MAX_HW_QUEUES from the cards' hardware compute queue slots, a set variable left alone, and the
// KFD files' reading (a made-up tree in a temporary directory). CPU only.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "../src/hip/queue_budget.hpp"

static int bad = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static void put(const std::string& path, const std::string& text) { std::ofstream(path) << text; }

int main() {
    using namespace fnx;
    // slots - 1, at most the runtime's default, at least 1; unknown: the default
    CHECK(queue_budget(4) == 3 && queue_budget(6) == 4 && queue_budget(8) == 4 && queue_budget(2) == 1 && queue_budget(1) == 1 && queue_budget(0) == 4);
    // not set: set here when it differs from the default
    QueueBudget b = queue_budget_decide(4, nullptr);
    CHECK(b.value == 3 && b.set_here && b.line == "GPU_MAX_HW_QUEUES=3 (set here: a card has 4 hardware compute queue slots)");
    b = queue_budget_decide(6, nullptr);
    CHECK(b.value == 4 && !b.set_here && b.line.empty());
    b = queue_budget_decide(0, nullptr);   // no KFD: nothing
    CHECK(b.value == 4 && !b.set_here && b.line.empty());
    // set: left alone; a line when it is over the budget
    b = queue_budget_decide(4, "4");
    CHECK(b.value == 4 && !b.set_here && b.line.rfind("GPU_MAX_HW_QUEUES=4 is set: more than 3 ", 0) == 0);
    b = queue_budget_decide(4, "2");
    CHECK(b.value == 2 && !b.set_here && b.line.empty());
    b = queue_budget_decide(4, "");   // an empty value is not a setting
    CHECK(b.value == 3 && b.set_here);

    // the files
    char tmpl[] = "/tmp/fnx-qb-XXXXXX";
    const char* dir = mkdtemp(tmpl);
    CHECK(dir != nullptr);
    if (dir) {
        const std::string root = dir;
        CHECK(kfd_slots(root).empty() && kfd_min_slots(root) == 0 && queue_count_line(root, 7).empty());
        for (const char* d : {"/topology", "/topology/nodes", "/topology/nodes/0", "/topology/nodes/1", "/topology/nodes/2", "/proc", "/proc/7", "/proc/7/queues",
                              "/proc/7/queues/0", "/proc/7/queues/1", "/proc/7/queues/2", "/proc/7/queues/3"})
            mkdir((root + d).c_str(), 0700);
        put(root + "/topology/nodes/0/gpu_id", "0\n"), put(root + "/topology/nodes/0/properties", "cpu_cores_count 32\nnum_cp_queues 0\n");
        put(root + "/topology/nodes/1/gpu_id", "34440\n"), put(root + "/topology/nodes/1/properties", "simd_count 64\nnum_cp_queues 4\nnum_sdma_engines 2\n");
        put(root + "/topology/nodes/2/gpu_id", "16290\n"), put(root + "/topology/nodes/2/properties", "num_cp_queues 6\n");
        put(root + "/proc/7/queues/0/gpuid", "34440\n"), put(root + "/proc/7/queues/0/type", "0\n");
        put(root + "/proc/7/queues/1/gpuid", "34440\n"), put(root + "/proc/7/queues/1/type", "1\n");   // a copy engine queue
        put(root + "/proc/7/queues/2/gpuid", "34440\n"), put(root + "/proc/7/queues/2/type", "0\n");
        put(root + "/proc/7/queues/3/gpuid", "16290\n"), put(root + "/proc/7/queues/3/type", "0\n");
        const auto sl = kfd_slots(root);
        CHECK(sl.size() == 2 && sl[0].first == 16290 && sl[0].second == 6 && sl[1].first == 34440 && sl[1].second == 4);
        CHECK(kfd_min_slots(root) == 4);
        const auto q = kfd_compute_queues(root, 7);
        CHECK(q.size() == 2 && q.at(34440) == 2 && q.at(16290) == 1);
        CHECK(queue_count_line(root, 7) == "gpu 16290: 1 / 6, gpu 34440: 2 / 4");
        CHECK(kfd_compute_queues(root, 8).empty() && queue_count_line(root, 8) == "gpu 16290: 0 / 6, gpu 34440: 0 / 4");
        if (std::system(("rm -rf '" + root + "'").c_str()) != 0) std::printf("(could not remove %s)\n", root.c_str());
    }
    if (bad) return 1;
    std::printf("queue_budget_test: ok\n");
    return 0;
}
