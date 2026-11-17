// queue_budget.hpp: how many compute queues the HIP runtime may make for this process on a card.
// A card has a fixed number of hardware compute queue slots for user queues (the KFD topology's num_cp_queues: 4 on an R9700; the driver keeps the
// others for its own rings). The HIP runtime makes up to GPU_MAX_HW_QUEUES (its default: 4) queues for the normal-priority streams, one more for
// the device and one for each other stream priority in use (the engine's prefetcher has one): 6 a card by default. With more than slots + 1
// queues the firmware maps and unmaps them, and after the first decode one card starts every prompt kernel ~16 us late (prompt reads -14 %,
// measured 2026-11-16). slots - 1 normal queues (3 on an R9700: 5 queues) keep the reads fast; fewer make streams share a queue (2: reads -7 %).
// A FAST build (STRATA_SPLIT_FAST) sets GPU_MAX_HW_QUEUES to that before its first HIP call when the variable is not set (a set variable is left
// alone, with a line when it is over the budget); a build without the option leaves the runtime's own number. Pure C++ (the files' roots are arguments: tools/queue_budget_test.cpp).
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace fnx {

constexpr int HIP_DEFAULT_HW_QUEUES = 4;   // ROCclr's GPU_MAX_HW_QUEUES

// the normal-priority queues a card with `slots` hardware slots takes (0 or less: unknown, the runtime's default)
inline int queue_budget(int slots) { return slots > 0 ? std::min(HIP_DEFAULT_HW_QUEUES, std::max(1, slots - 1)) : HIP_DEFAULT_HW_QUEUES; }

// each GPU node's (gpu_id, num_cp_queues) from the KFD topology (root: /sys/class/kfd/kfd). A node with gpu_id 0 is the CPU's. Empty: no KFD
inline std::vector<std::pair<long, int>> kfd_slots(const std::string& root) {
    std::vector<std::pair<long, int>> out;
    const std::string nodes = root + "/topology/nodes";
    DIR* d = opendir(nodes.c_str());
    if (!d) return out;
    while (const dirent* e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        long gid = 0;
        if (!(std::ifstream(nodes + "/" + e->d_name + "/gpu_id") >> gid) || gid == 0) continue;
        std::ifstream props(nodes + "/" + e->d_name + "/properties");
        std::string key;
        long long val = 0;
        while (props >> key >> val)
            if (key == "num_cp_queues") out.push_back({gid, (int) val});
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}
inline int kfd_min_slots(const std::string& root) {
    int m = 0;
    for (const auto& g : kfd_slots(root)) m = m == 0 ? g.second : std::min(m, g.second);
    return m;
}

// the process's compute queues a card right now (root/proc/<pid>/queues/<n>/{gpuid,type}; type 0 = compute), by gpu_id
inline std::map<long, int> kfd_compute_queues(const std::string& root, long pid) {
    std::map<long, int> out;
    const std::string qs = root + "/proc/" + std::to_string(pid) + "/queues";
    DIR* d = opendir(qs.c_str());
    if (!d) return out;
    while (const dirent* e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        long gid = 0, type = -1;
        if ((std::ifstream(qs + "/" + e->d_name + "/gpuid") >> gid) && (std::ifstream(qs + "/" + e->d_name + "/type") >> type) && type == 0) ++out[gid];
    }
    closedir(d);
    return out;
}

struct QueueBudget {
    int slots = 0;        // the least num_cp_queues of the GPU nodes (0: unknown)
    int value = 0;        // GPU_MAX_HW_QUEUES for this run
    bool set_here = false;
    std::string line;     // for stderr
};
// the decision (cur: the variable's value, or nullptr when it is not set)
inline QueueBudget queue_budget_decide(int slots, const char* cur) {
    QueueBudget b;
    b.slots = slots;
    const int want = queue_budget(slots);
    if (cur && *cur) {
        b.value = std::atoi(cur);
        if (slots > 0 && b.value > want)
            b.line = "GPU_MAX_HW_QUEUES=" + std::string(cur) + " is set: more than " + std::to_string(want) + " (a card has " + std::to_string(slots) +
                     " hardware compute queue slots); prompt reads can lose ~14 % after the first decode";
        return b;
    }
    b.value = want, b.set_here = slots > 0 && want != HIP_DEFAULT_HW_QUEUES;
    if (b.set_here)
        b.line = "GPU_MAX_HW_QUEUES=" + std::to_string(want) + " (set here: a card has " + std::to_string(slots) + " hardware compute queue slots)";
    return b;
}
// at the engine's start, before the first HIP call
// set = false: the decision is not applied (the runtime keeps its own number, or the variable's): value says what the run has
inline QueueBudget queue_budget_apply(const std::string& root = "/sys/class/kfd/kfd", bool set = true) {
    QueueBudget b = queue_budget_decide(kfd_min_slots(root), std::getenv("GPU_MAX_HW_QUEUES"));
    if (b.set_here && !set) b.value = HIP_DEFAULT_HW_QUEUES, b.set_here = false, b.line.clear();
    if (b.set_here) setenv("GPU_MAX_HW_QUEUES", std::to_string(b.value).c_str(), 1);
    return b;
}
// "gpu <id>: <queues> / <slots>, ..." for the process's compute queues now
inline std::string queue_count_line(const std::string& root, long pid) {
    const auto have = kfd_compute_queues(root, pid);
    std::string s;
    for (const auto& g : kfd_slots(root)) {
        const auto it = have.find(g.first);
        s += (s.empty() ? "" : ", ") + std::string("gpu ") + std::to_string(g.first) + ": " + std::to_string(it == have.end() ? 0 : it->second) + " / " + std::to_string(g.second);
    }
    return s;
}

}  // namespace fnx
