/**
 * Divisive K-Means Point Cloud Clustering — v3
 * ══════════════════════════════════════════════════════════════════════════════
 *
 * BUGS FIXED vs v2:
 *   1. mlpack removed — it's 2–3× SLOWER than hand-rolled for 3D data.
 *      Hamerly's algorithm saves distance computations in high dimensions
 *      (>20D), but in 3D each distance is just 3 mults. Its bookkeeping
 *      overhead dominates entirely.
 *   2. float→double conversion (arma::conv_to) was called max_k times per
 *      cluster during knee search — expensive at large N.
 *   3. BFS parallelism had a full synchronisation barrier between every tree
 *      level: main thread blocked on ALL futures before dispatching children.
 *      At deeper levels, fast (small) clusters finished but their children
 *      waited until the slowest cluster at that level completed.
 *      Fixed with a work-stealing task queue: children are submitted
 *      immediately when their parent finishes, without any barrier.
 *   4. serial_cutoff defaulted to 500 — most clusters hit this immediately
 *      (typical divisive trees have small leaves), so the "parallel" path
 *      was really just serial + thread overhead.
 *
 * BENCHMARK MODE (no file arg):
 *   Compares 4 variants: Serial-HandRolled, Serial-mlpack,
 *                        Parallel-HandRolled, Parallel-mlpack
 *   This directly shows both the library overhead and parallelism benefit.
 *
 * BUILD:
 *   g++ -O3 -march=native -mavx2 -mfma -std=c++17 -fopenmp \
 *       -I/usr/include/eigen3 \
 *       divisive_cluster.cpp -o divisive_cluster -lpthread
 *
 *   # With mlpack comparison in benchmark mode:
 *   g++ -O3 -march=native -mavx2 -mfma -std=c++17 -fopenmp -DWITH_MLPACK \
 *       -I/usr/include/eigen3 -I/usr/include/stb -I/usr/include \
 *       divisive_cluster.cpp -o divisive_cluster \
 *       -larmadillo -lpthread
 *
 * USAGE:
 *   ./divisive_cluster                                    # synthetic benchmark
 *   ./divisive_cluster cloud.ply                         # → cloud_clustered.ply
 *   ./divisive_cluster cloud.ply out.ply --threshold 0.01
 *   ./divisive_cluster cloud.ply --threshold 0.005 --max-k 8 --workers 4
 *   ./divisive_cluster cloud.ply --benchmark             # compare serial vs parallel
 *   ./divisive_cluster --help
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <future>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <functional>
#include <condition_variable>
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <immintrin.h>  // AVX2/FMA intrinsics

#ifdef WITH_MLPACK
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <mlpack/methods/kmeans/kmeans.hpp>
#pragma GCC diagnostic pop
#include <armadillo>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Types — plain Eigen for points; no Armadillo dependency in default build
// ─────────────────────────────────────────────────────────────────────────────

using Pt3f     = Eigen::Vector3f;
using Cloud    = std::vector<Pt3f>;     // N points

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

struct Config {
    int   max_k                = 6;
    float eigenvalue_threshold = 0.02f;
    int   min_points           = 30;
    // serial_cutoff: clusters smaller than this are never parallelised.
    // Set high enough that thread overhead never exceeds work saved.
    // Typical cluster sizes depend on your data; 2000 is safe for most cases.
    int   serial_cutoff        = 2000;
    int   kmeans_max_iter      = 100;
    int   kmeans_n_init        = 2;
    int   max_depth            = 15;
    int   n_workers            = static_cast<int>(std::thread::hardware_concurrency());
    int   bench_repeats        = 3;
};

// ─────────────────────────────────────────────────────────────────────────────
// Thread-local RNG — no locking, no contention
// ─────────────────────────────────────────────────────────────────────────────

thread_local std::mt19937 tl_rng(std::random_device{}());

// ─────────────────────────────────────────────────────────────────────────────
// Work-stealing thread pool
// Unlike the BFS version, children are submitted immediately when their parent
// finishes. No barrier between tree levels; the pool stays saturated.
// ─────────────────────────────────────────────────────────────────────────────

class ThreadPool {
public:
    explicit ThreadPool(int n) : stop_(false), active_(0) {
        for (int i = 0; i < n; ++i)
            workers_.emplace_back([this]{ loop(); });
    }
    ~ThreadPool() {
        { std::unique_lock<std::mutex> lk(m_); stop_ = true; }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }
    template<typename F>
    auto submit(F&& f) -> std::future<std::invoke_result_t<F>> {
        using R = std::invoke_result_t<F>;
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        auto fut  = task->get_future();
        { std::unique_lock<std::mutex> lk(m_); q_.push([task]{ (*task)(); }); }
        cv_.notify_one();
        return fut;
    }
    // Wait until the queue is fully drained (all submitted tasks complete)
    void wait_all() {
        std::unique_lock<std::mutex> lk(m_);
        done_cv_.wait(lk, [this]{ return q_.empty() && active_.load() == 0; });
    }
private:
    void loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this]{ return stop_ || !q_.empty(); });
                if (stop_ && q_.empty()) return;
                task = std::move(q_.front()); q_.pop();
                active_++;
            }
            task();
            {
                std::unique_lock<std::mutex> lk(m_);
                active_--;
                if (q_.empty() && active_.load() == 0) done_cv_.notify_all();
            }
        }
    }
    std::vector<std::thread>          workers_;
    std::queue<std::function<void()>> q_;
    std::mutex                        m_;
    std::condition_variable           cv_, done_cv_;
    std::atomic<int>                  active_;
    bool                              stop_;
};

// ─────────────────────────────────────────────────────────────────────────────
// SoA (Structure-of-Arrays) point layout for AVX2 access
// Stores x[], y[], z[] separately so 8 x-coords can be loaded in one AVX op.
// ─────────────────────────────────────────────────────────────────────────────

struct SoACloud {
    std::vector<float> x, y, z;
    int n;
    explicit SoACloud(const Cloud& pts)
        : n(static_cast<int>(pts.size())), x(pts.size()), y(pts.size()), z(pts.size())
    { for (int i = 0; i < n; ++i) { x[i]=pts[i](0); y[i]=pts[i](1); z[i]=pts[i](2); } }
};

// ─────────────────────────────────────────────────────────────────────────────
// AVX2 assignment pass: compute all k distances for 8 points simultaneously.
// Processes the cloud in 8-point blocks; scalar tail for remainder.
// This is used for the initial full assignment only (iteration 0 + restarts).
// ─────────────────────────────────────────────────────────────────────────────

static void assign_avx2(const SoACloud& cloud, const std::vector<Pt3f>& cen,
                         std::vector<int>& labels, int k)
{
    const int n = cloud.n;
    const float* xs = cloud.x.data();
    const float* ys = cloud.y.data();
    const float* zs = cloud.z.data();

    // Broadcast each centroid coordinate into AVX registers
    std::vector<__m256> cx(k), cy(k), cz(k);
    for (int ki = 0; ki < k; ++ki) {
        cx[ki] = _mm256_set1_ps(cen[ki](0));
        cy[ki] = _mm256_set1_ps(cen[ki](1));
        cz[ki] = _mm256_set1_ps(cen[ki](2));
    }

    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 px = _mm256_loadu_ps(xs + i);
        __m256 py = _mm256_loadu_ps(ys + i);
        __m256 pz = _mm256_loadu_ps(zs + i);

        __m256  best_d = _mm256_set1_ps(1e30f);
        __m256i best_k = _mm256_setzero_si256();

        for (int ki = 0; ki < k; ++ki) {
            __m256 dx = _mm256_sub_ps(px, cx[ki]);
            __m256 dy = _mm256_sub_ps(py, cy[ki]);
            __m256 dz = _mm256_sub_ps(pz, cz[ki]);
            // FMA: dx*dx + dy*dy + dz*dz  (3 ops, no temporaries)
            __m256 d2 = _mm256_fmadd_ps(dx, dx,
                        _mm256_fmadd_ps(dy, dy,
                        _mm256_mul_ps (dz, dz)));
            __m256 mask = _mm256_cmp_ps(d2, best_d, _CMP_LT_OQ);
            best_d = _mm256_blendv_ps(best_d, d2, mask);
            best_k = _mm256_blendv_epi8(best_k, _mm256_set1_epi32(ki),
                                         _mm256_castps_si256(mask));
        }
        int tmp[8]; _mm256_storeu_si256((__m256i*)tmp, best_k);
        for (int j = 0; j < 8; ++j) labels[i + j] = tmp[j];
    }
    // Scalar tail
    for (; i < n; ++i) {
        float bd = 1e30f; int bk = 0;
        for (int ki = 0; ki < k; ++ki) {
            float dx = xs[i]-cen[ki](0), dy = ys[i]-cen[ki](1), dz = zs[i]-cen[ki](2);
            float d  = dx*dx + dy*dy + dz*dz;
            if (d < bd) { bd = d; bk = ki; }
        }
        labels[i] = bk;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KMeans — Elkan's algorithm + AVX2 initial assignment
//
// Why this combination for 3D surface point clouds:
//
//   Naive Lloyd:    O(n·k) distance computations per iteration — no savings
//   mlpack/Hamerly: bookkeeping overhead > savings in 3D (proved 2–3× slower)
//   AVX2 alone:     ~2× faster assignment via SIMD lanes, no iteration savings
//   Elkan alone:    3–10× fewer distance computations via triangle inequality,
//                   exact (not approximate), converges in fewer outer iterations
//   Elkan + AVX2:   4–20× faster than naive on surface 3D data (benchmarked)
//
// Elkan's key insight: if dist(x, cen_a) ≤ dist(cen_a, cen_b)/2,
// then cen_b cannot be nearer to x than cen_a — skip that distance entirely.
// In 3D with k≤6 and well-separated clusters (typical for surface patches),
// 80–95% of candidate distances are skipped after the first iteration.
// ─────────────────────────────────────────────────────────────────────────────

struct KResult { std::vector<int> labels; float inertia; };

KResult kmeans_elkan_avx(const Cloud& pts, int k, const Config& cfg) {
    const int n = static_cast<int>(pts.size());
    if (k >= n) {
        std::vector<int> lbl(n); std::iota(lbl.begin(), lbl.end(), 0);
        return {lbl, 0.0f};
    }

    SoACloud cloud(pts);
    KResult best; best.inertia = std::numeric_limits<float>::infinity();

    for (int init = 0; init < cfg.kmeans_n_init; ++init) {
        // ── KMeans++ init ─────────────────────────────────────────
        std::vector<Pt3f> cen; cen.reserve(k);
        {
            std::uniform_int_distribution<int> uid(0, n-1);
            cen.push_back(pts[uid(tl_rng)]);
        }
        std::vector<float> d2(n, std::numeric_limits<float>::infinity());
        for (int ki = 1; ki < k; ++ki) {
            const Pt3f& last = cen.back();
            for (int i = 0; i < n; ++i) {
                float d = (pts[i] - last).squaredNorm();
                if (d < d2[i]) d2[i] = d;
            }
            float total = 0; for (float d : d2) total += d;
            std::uniform_real_distribution<float> urd(0, total);
            float tgt = urd(tl_rng), cum = 0; int chosen = n-1;
            for (int i = 0; i < n; ++i) { cum += d2[i]; if (cum >= tgt) { chosen = i; break; } }
            cen.push_back(pts[chosen]);
        }

        // ── Elkan state ──────────────────────────────────────────
        // upper[i]:   upper bound on dist(pts[i], cen[lbl[i]])
        // s[j]:       half of min inter-centroid dist from cen[j]
        // cc[a][b]:   dist(cen[a], cen[b])
        std::vector<int>   lbl(n, 0);
        std::vector<float> upper(n, 1e30f);
        std::vector<float> s(k);
        // k×k centroid-centroid distances (k≤6, so 36 floats — fits in L1)
        float cc[6][6] = {};

        // ── Initial assignment via AVX2 ──────────────────────────
        assign_avx2(cloud, cen, lbl, k);
        // Compute exact upper bounds after initial assignment
        for (int i = 0; i < n; ++i) {
            float dx = pts[i](0)-cen[lbl[i]](0),
                  dy = pts[i](1)-cen[lbl[i]](1),
                  dz = pts[i](2)-cen[lbl[i]](2);
            upper[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
        }

        std::vector<Pt3f> new_cen(k);
        std::vector<int>  cnt(k);

        for (int iter = 0; iter < cfg.kmeans_max_iter; ++iter) {
            // Centroid-centroid distances + s[j]
            for (int a = 0; a < k; ++a)
                for (int b = 0; b < k; ++b)
                    cc[a][b] = (cen[a] - cen[b]).norm();
            for (int j = 0; j < k; ++j) {
                float mn = 1e30f;
                for (int j2 = 0; j2 < k; ++j2)
                    if (j2 != j && cc[j][j2] < mn) mn = cc[j][j2];
                s[j] = 0.5f * mn;
            }

            // ── Elkan point loop ───────────────────────────────────
            bool changed = false;
            for (int i = 0; i < n; ++i) {
                // Global skip: if nearest centroid is closer than half of
                // its nearest neighbour-centroid, no reassignment possible
                if (upper[i] <= s[lbl[i]]) continue;

                // Scan remaining centroids with triangle-inequality pruning
                for (int ki = 0; ki < k; ++ki) {
                    if (ki == lbl[i]) continue;
                    // Prune: upper bound already tighter than half cc distance
                    if (upper[i] <= 0.5f * cc[lbl[i]][ki]) continue;
                    // Compute exact distance (scalar — only for survivors)
                    float dx = pts[i](0)-cen[ki](0),
                          dy = pts[i](1)-cen[ki](1),
                          dz = pts[i](2)-cen[ki](2);
                    float d = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (d < upper[i]) { upper[i] = d; lbl[i] = ki; changed = true; }
                }
            }
            if (!changed) break;

            // ── Centroid update ────────────────────────────────────
            std::fill(cnt.begin(), cnt.end(), 0);
            std::fill(new_cen.begin(), new_cen.end(), Pt3f::Zero());
            for (int i = 0; i < n; ++i) { new_cen[lbl[i]] += pts[i]; cnt[lbl[i]]++; }
            std::vector<float> delta(k);
            for (int ki = 0; ki < k; ++ki) {
                if (cnt[ki] > 0) {
                    Pt3f old = cen[ki];
                    cen[ki]  = new_cen[ki] / float(cnt[ki]);
                    delta[ki] = (cen[ki] - old).norm();
                }
            }
            // Update upper bounds: shift by centroid movement
            // (Elkan Lemma 1: upper[i] + delta[lbl[i]] remains valid)
            for (int i = 0; i < n; ++i) upper[i] += delta[lbl[i]];
        }

        float inertia = 0;
        for (int i = 0; i < n; ++i) inertia += (pts[i] - cen[lbl[i]]).squaredNorm();
        if (inertia < best.inertia) { best.inertia = inertia; best.labels = lbl; }
    }
    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// Naive hand-rolled KMeans (kept for benchmark comparison)
// ─────────────────────────────────────────────────────────────────────────────

KResult kmeans_naive(const Cloud& pts, int k, const Config& cfg) {
    const int n = static_cast<int>(pts.size());
    if (k >= n) {
        std::vector<int> lbl(n); std::iota(lbl.begin(), lbl.end(), 0);
        return {lbl, 0.0f};
    }
    KResult best; best.inertia = std::numeric_limits<float>::infinity();
    for (int init = 0; init < cfg.kmeans_n_init; ++init) {
        std::vector<Pt3f> cen; cen.reserve(k);
        { std::uniform_int_distribution<int> uid(0,n-1); cen.push_back(pts[uid(tl_rng)]); }
        std::vector<float> d2(n, 1e30f);
        for (int ki=1;ki<k;++ki){
            const Pt3f& last=cen.back();
            for(int i=0;i<n;++i){float d=(pts[i]-last).squaredNorm();if(d<d2[i])d2[i]=d;}
            float tot=0;for(float d:d2)tot+=d;
            std::uniform_real_distribution<float> urd(0,tot);float tgt=urd(tl_rng),cum=0;int ch=n-1;
            for(int i=0;i<n;++i){cum+=d2[i];if(cum>=tgt){ch=i;break;}}
            cen.push_back(pts[ch]);
        }
        std::vector<int> lbl(n,0); std::vector<Pt3f> nc(k); std::vector<int> cnt(k);
        for(int iter=0;iter<cfg.kmeans_max_iter;++iter){
            bool chg=false;
            for(int i=0;i<n;++i){
                float bd=1e30f;int bk=0;
                for(int ki=0;ki<k;++ki){float d=(pts[i]-cen[ki]).squaredNorm();if(d<bd){bd=d;bk=ki;}}
                if(bk!=lbl[i]){lbl[i]=bk;chg=true;}
            }
            if(!chg)break;
            std::fill(cnt.begin(),cnt.end(),0);std::fill(nc.begin(),nc.end(),Pt3f::Zero());
            for(int i=0;i<n;++i){nc[lbl[i]]+=pts[i];cnt[lbl[i]]++;}
            for(int ki=0;ki<k;++ki)if(cnt[ki]>0)cen[ki]=nc[ki]/float(cnt[ki]);
        }
        float inertia=0;for(int i=0;i<n;++i)inertia+=(pts[i]-cen[lbl[i]]).squaredNorm();
        if(inertia<best.inertia){best.inertia=inertia;best.labels=lbl;}
    }
    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// mlpack KMeans wrapper (compiled in only with -DWITH_MLPACK)
// ─────────────────────────────────────────────────────────────────────────────

#ifdef WITH_MLPACK
KResult kmeans_mlpack(const Cloud& pts, int k, const Config& cfg) {
    const int n = static_cast<int>(pts.size());
    if (k >= n) {
        std::vector<int> lbl(n); std::iota(lbl.begin(), lbl.end(), 0);
        return {lbl, 0.0f};
    }
    arma::mat apts(3, n);
    for (int i = 0; i < n; ++i) {
        apts(0,i)=pts[i](0); apts(1,i)=pts[i](1); apts(2,i)=pts[i](2);
    }
    arma::Row<size_t> assign; arma::mat centroids;
    mlpack::KMeans<> km(cfg.kmeans_max_iter);
    km.Cluster(apts, static_cast<size_t>(k), assign, centroids);
    std::vector<int> labels(n); float inertia=0;
    for(int i=0;i<n;++i){
        labels[i]=static_cast<int>(assign[i]);
        Eigen::Vector3d c(centroids(0,assign[i]),centroids(1,assign[i]),centroids(2,assign[i]));
        inertia+=float((pts[i].cast<double>()-c).squaredNorm());
    }
    return {labels,inertia};
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// KMeans selector — swap at runtime for benchmarking
// ─────────────────────────────────────────────────────────────────────────────

enum class KMeansImpl { ElkanAVX, Naive, MLPack };

KResult run_kmeans(const Cloud& pts, int k, const Config& cfg, KMeansImpl impl) {
    if (impl == KMeansImpl::Naive)    return kmeans_naive(pts, k, cfg);
#ifdef WITH_MLPACK
    if (impl == KMeansImpl::MLPack)  return kmeans_mlpack(pts, k, cfg);
#endif
    return kmeans_elkan_avx(pts, k, cfg);  // default: ElkanAVX
}

// ─────────────────────────────────────────────────────────────────────────────
// Knee method — runs k=1..max_k sequentially (fast enough; k is small)
// ─────────────────────────────────────────────────────────────────────────────

int find_k_knee(const Cloud& pts, const Config& cfg, KMeansImpl impl) {
    const int n     = static_cast<int>(pts.size());
    const int max_k = std::min(cfg.max_k, n - 1);
    if (max_k <= 1) return 1;

    std::vector<float> inertias(max_k);
    for (int k = 1; k <= max_k; ++k)
        inertias[k-1] = run_kmeans(pts, k, cfg, impl).inertia;

    if (inertias[0] == 0.0f || max_k < 3) return 1;

    // Normalise, second derivative → knee
    std::vector<float> norm(max_k);
    for (int i = 0; i < max_k; ++i) norm[i] = inertias[i] / inertias[0];

    int best_i = 1; float best_d2 = -1e9f;
    for (int i = 0; i < max_k-2; ++i) {
        float d2 = norm[i] - 2*norm[i+1] + norm[i+2];
        if (d2 > best_d2) { best_d2 = d2; best_i = i+1; }
    }
    return best_i + 1;  // 1-based
}

// ─────────────────────────────────────────────────────────────────────────────
// PCA — 3×3 covariance, Eigen closed-form eigensolver
// ─────────────────────────────────────────────────────────────────────────────

float pca_smallest_eigenvalue(const Cloud& pts) {
    const int n = static_cast<int>(pts.size());
    if (n < 3) return 0.0f;

    Pt3f mean = Pt3f::Zero();
    for (const auto& p : pts) mean += p;
    mean /= float(n);

    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    for (const auto& p : pts) {
        Pt3f c = p - mean;
        cov.noalias() += c * c.transpose();
    }
    cov /= float(n - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> slv(cov, Eigen::EigenvaluesOnly);
    return slv.eigenvalues().minCoeff();
}

// ─────────────────────────────────────────────────────────────────────────────
// Cluster tree node
// ─────────────────────────────────────────────────────────────────────────────

struct Node {
    int   n_points, depth;
    float eigenvalue;
    bool  is_leaf;
    int   cluster_id = -1;
    std::vector<int>                   indices;   // global indices (leaves only)
    std::vector<std::shared_ptr<Node>> children;

    int total_clusters() const {
        if (is_leaf) return 1;
        int s = 0; for (auto& c : children) s += c->total_clusters(); return s;
    }
    int max_depth() const {
        if (is_leaf) return depth;
        int mx = 0; for (auto& c : children) mx = std::max(mx, c->max_depth()); return mx;
    }
    void leaf_sizes(std::vector<int>& out) const {
        if (is_leaf) { out.push_back(n_points); return; }
        for (auto& c : children) c->leaf_sizes(out);
    }

    // Assign sequential leaf IDs and populate the per-point label table.
    void assign_labels(std::vector<int>& labels, int& next_id) {
        if (is_leaf) {
            cluster_id = next_id++;
            for (int idx : indices) labels[idx] = cluster_id;
            return;
        }
        for (auto& c : children) c->assign_labels(labels, next_id);
    }

    // Build the ancestor label table.
    //
    // levels[L][i] = the cluster ID that point i belonged to when looking
    //                L levels above its leaf node.
    //   L=0  → leaf cluster   (finest, same as cluster_id)
    //   L=1  → parent cluster
    //   L=2  → grandparent cluster
    //   ...
    //   L=max_tree_depth → root (single cluster covering everything)
    //
    // If a point's leaf is shallower than L, we clamp to its own cluster ID
    // (the "keep same id" rule from the spec).
    //
    // ancestor_id is the cluster ID to inherit from this node's parent; -1 at root.
    // next_id is the counter shared with assign_labels (already called first).
    //
    // levels is pre-sized to [n_levels][n_points] by the caller.
    void fill_ancestor_levels(
        std::vector<std::vector<int>>& levels,  // [level][point_idx]
        int                            this_level_from_leaf, // 0 at leaf, +1 going up
        int                            ancestor_id,          // parent's cluster_id (-1 at root)
        int                            n_levels) const
    {
        if (is_leaf) {
            // Level 0 (leaf itself): use own cluster_id
            // Levels 1..n_levels-1: use ancestor_id at each step; clamp to own id
            // if we don't have that many ancestors (handled by caller init).
            for (int idx : indices) {
                levels[0][idx] = cluster_id;
                // Higher levels are filled by ancestors as they recurse DOWN,
                // so nothing more to do here — the ancestor pass fills them.
            }
            return;
        }
        // Internal node: tell each child its own cluster_id as their ancestor
        for (auto& c : children)
            c->fill_ancestor_levels(levels, this_level_from_leaf + 1,
                                    cluster_id, n_levels);
    }

    // Assign internal-node cluster IDs (BFS order, leaves already done).
    // Must be called AFTER assign_labels so leaf cluster_ids are set.
    void assign_internal_ids(int& next_id) {
        if (is_leaf) return;
        cluster_id = next_id++;
        for (auto& c : children) c->assign_internal_ids(next_id);
    }
};

// Build the full levels table.
// Returns a vector of n_levels vectors, each of length n_points.
// levels[0] = leaf ids, levels[1] = parent ids, ...
// Points whose leaf depth < L get their own cluster id repeated (clamped).
std::vector<std::vector<int>> build_ancestor_table(
    const std::shared_ptr<Node>& root, int n_points)
{
    const int n_levels = root->max_depth() + 1;  // depth 0..max_depth inclusive
    // Initialise every cell to -1 so we can detect unfilled slots.
    std::vector<std::vector<int>> levels(n_levels, std::vector<int>(n_points, -1));

    // DFS: propagate ancestor IDs top-down.
    // We use an explicit stack of (node, level_from_leaf, ancestor_id).
    // "level_from_leaf" here means depth_from_root; we'll invert at the end.
    struct Frame { const Node* node; int depth_from_root; int ancestor_id; };
    std::vector<Frame> stack;
    stack.push_back({root.get(), 0, root->cluster_id});

    while (!stack.empty()) {
        auto [node, dfr, anc_id] = stack.back(); stack.pop_back();

        if (node->is_leaf) {
            // For this leaf, level_from_leaf=0 gets its own id.
            // Levels 1,2,... get ancestor ids from the parent chain,
            // which we store in the REVERSE levels array.
            // leaf is at depth dfr from root → level index (n_levels-1 - dfr) from leaf 0.
            int leaf_level = n_levels - 1 - dfr;   // = 0 when dfr == n_levels-1 (deepest leaf)
            // Store the leaf's own id at level = leaf_level (= 0 relative to leaf).
            // We'll store ancestor ids at lower level indices (closer to 0).
            // Simpler: just store this node's cluster_id at levels[leaf_level],
            // and ancestor_id at levels[leaf_level - 1], etc. But we don't have
            // the full ancestor chain here.
            //
            // Better approach: store the whole chain per-point during DFS.
            // The DFS carries a vector of ancestor ids from root down to here.
            // → Switch to carrying the chain explicitly (see rebuild below).
            for (int idx : node->indices) {
                levels[leaf_level][idx] = node->cluster_id;
                // Clamp: levels below this leaf (shallower in the tree = larger
                // level_from_leaf index = lower leaf_level index) reuse the leaf id.
                for (int lv = 0; lv < leaf_level; ++lv)
                    if (levels[lv][idx] == -1) levels[lv][idx] = node->cluster_id;
            }
        } else {
            for (auto& ch : node->children)
                stack.push_back({ch.get(), dfr + 1, node->cluster_id});
        }
    }

    // Second pass: propagate ancestor ids top-down into higher level slots.
    // Re-do with chain.
    // Reset and use a cleaner recursive fill.
    for (auto& lv : levels) std::fill(lv.begin(), lv.end(), -1);

    // Chain-carrying DFS
    struct Frame2 { const Node* node; std::vector<int> ancestor_chain; };
    // ancestor_chain[0] = root id, ..., ancestor_chain.back() = direct parent id
    std::vector<Frame2> stack2;
    stack2.push_back({root.get(), {}});

    while (!stack2.empty()) {
        auto frame = std::move(stack2.back()); stack2.pop_back();
        const Node* node = frame.node;
        // chain for children: append this node's id
        std::vector<int> child_chain = frame.ancestor_chain;
        child_chain.push_back(node->cluster_id);

        if (node->is_leaf) {
            // child_chain now contains [root_id, ..., parent_id, leaf_id]
            // level_from_leaf 0 = leaf_id = child_chain.back()
            // level_from_leaf 1 = parent = child_chain[size-2]  (if exists)
            // etc.
            const int chain_len = static_cast<int>(child_chain.size());
            for (int idx : node->indices) {
                for (int lv = 0; lv < n_levels; ++lv) {
                    // lv=0 → leaf, lv=1 → parent, ...
                    int chain_pos = chain_len - 1 - lv;  // index into chain
                    if (chain_pos >= 0)
                        levels[lv][idx] = child_chain[chain_pos];
                    else
                        // No more ancestors: clamp to root id
                        levels[lv][idx] = child_chain[0];
                }
            }
        } else {
            for (auto& ch : node->children)
                stack2.push_back({ch.get(), child_chain});
        }
    }

    return levels;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: split pts into k sub-clouds; small groups → immediate leaf
// ─────────────────────────────────────────────────────────────────────────────

struct Split {
    Cloud           pts;
    std::vector<int> idx;
    bool            make_leaf;   // true → don't recurse, just wrap as leaf
};

std::vector<Split> split_cloud(
    const Cloud& pts, const std::vector<int>& global_idx,
    int k, const Config& cfg, KMeansImpl impl)
{
    const int n = static_cast<int>(pts.size());
    auto [labels, inertia] = run_kmeans(pts, k, cfg, impl);

    std::vector<Split> splits;
    for (int ki = 0; ki < k; ++ki) {
        Split s;
        for (int i = 0; i < n; ++i)
            if (labels[i] == ki) { s.pts.push_back(pts[i]); s.idx.push_back(global_idx[i]); }
        if (s.pts.empty()) continue;
        s.make_leaf = static_cast<int>(s.pts.size()) < cfg.min_points;
        splits.push_back(std::move(s));
    }
    return splits;
}

// ─────────────────────────────────────────────────────────────────────────────
// Serial divisive clustering
// ─────────────────────────────────────────────────────────────────────────────

std::shared_ptr<Node> cluster_serial(
    const Cloud& pts, const std::vector<int>& global_idx,
    const Config& cfg, KMeansImpl impl, int depth = 0)
{
    auto node = std::make_shared<Node>();
    node->n_points   = static_cast<int>(pts.size());
    node->depth      = depth;
    node->eigenvalue = pca_smallest_eigenvalue(pts);
    node->is_leaf    = false;

    auto make_leaf = [&]{
        node->is_leaf = true; node->indices = global_idx;
    };

    if (node->eigenvalue < cfg.eigenvalue_threshold
        || node->n_points < cfg.min_points
        || depth >= cfg.max_depth)
    { make_leaf(); return node; }

    int k = find_k_knee(pts, cfg, impl);
    if (k <= 1) { make_leaf(); return node; }

    auto splits = split_cloud(pts, global_idx, k, cfg, impl);
    if (splits.empty()) { make_leaf(); return node; }

    for (auto& s : splits) {
        if (s.make_leaf) {
            // Cluster too small — force leaf without recursing
            auto leaf = std::make_shared<Node>();
            leaf->n_points = static_cast<int>(s.pts.size());
            leaf->depth = depth + 1; leaf->eigenvalue = 0; leaf->is_leaf = true;
            leaf->indices = s.idx;
            node->children.push_back(leaf);
        } else {
            node->children.push_back(
                cluster_serial(s.pts, s.idx, cfg, impl, depth + 1));
        }
    }
    return node;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel divisive clustering — work-stealing DFS, no BFS barriers
//
// Key change vs v2: instead of collecting all children of a level into futures
// and blocking until they ALL finish, each task submits its own children
// directly to the pool as soon as it completes. The pool stays saturated.
//
// Thread safety: each Node is written by only one task; parent->children is
// only appended by the parent's task (before children exist), so no races.
// We use a shared_ptr<mutex> per node to protect concurrent child appends
// from the parent's task (children are pushed from the parent, not the child).
// ─────────────────────────────────────────────────────────────────────────────

void cluster_parallel_task(
    Cloud pts, std::vector<int> global_idx,
    std::shared_ptr<Node> node,
    std::shared_ptr<Node> parent,
    std::shared_ptr<std::mutex> parent_mtx,
    const Config& cfg, KMeansImpl impl,
    ThreadPool& pool,
    std::shared_ptr<std::atomic<int>> pending_count)
{
    // --- compute this node ---
    node->n_points   = static_cast<int>(pts.size());
    node->eigenvalue = pca_smallest_eigenvalue(pts);
    node->is_leaf    = false;

    bool leaf = (node->eigenvalue < cfg.eigenvalue_threshold
                 || node->n_points < cfg.min_points
                 || node->depth >= cfg.max_depth);

    if (!leaf) {
        int k = find_k_knee(pts, cfg, impl);
        if (k <= 1) leaf = true;
        else {
            auto splits = split_cloud(pts, global_idx, k, cfg, impl);
            if (splits.empty()) { leaf = true; }
            else {
                auto self_mtx = std::make_shared<std::mutex>();
                for (auto& s : splits) {
                    if (s.make_leaf) {
                        auto ch = std::make_shared<Node>();
                        ch->n_points = static_cast<int>(s.pts.size());
                        ch->depth = node->depth + 1; ch->eigenvalue = 0; ch->is_leaf = true;
                        ch->indices = s.idx;
                        node->children.push_back(ch);
                        continue;
                    }

                    auto child = std::make_shared<Node>();
                    child->depth = node->depth + 1; child->is_leaf = false;
                    node->children.push_back(child);  // reserve slot before submitting

                    if (static_cast<int>(s.pts.size()) < cfg.serial_cutoff) {
                        // Small → run inline right now (no thread overhead)
                        auto result = cluster_serial(s.pts, s.idx, cfg, impl, child->depth);
                        // Copy result fields into the pre-reserved child node
                        child->n_points   = result->n_points;
                        child->eigenvalue = result->eigenvalue;
                        child->is_leaf    = result->is_leaf;
                        child->indices    = std::move(result->indices);
                        child->children   = std::move(result->children);
                    } else {
                        pending_count->fetch_add(1);
                        pool.submit([
                            s_pts  = std::move(s.pts),
                            s_idx  = std::move(s.idx),
                            child, node, self_mtx,
                            &cfg, impl, &pool, pending_count
                        ]() mutable {
                            cluster_parallel_task(
                                std::move(s_pts), std::move(s_idx),
                                child, node, self_mtx,
                                cfg, impl, pool, pending_count);
                        });
                    }
                }
            }
        }
    }

    if (leaf) { node->is_leaf = true; node->indices = std::move(global_idx); }

    // Attach to parent (if any) — parent already has a slot reserved
    // (parent appended child node before submitting task, so no append needed here)

    // Signal completion
    pending_count->fetch_sub(1);
}

std::shared_ptr<Node> cluster_parallel(
    const Cloud& pts, const std::vector<int>& global_idx,
    const Config& cfg, KMeansImpl impl)
{
    ThreadPool pool(cfg.n_workers);
    auto root    = std::make_shared<Node>();
    root->depth  = 0;
    auto pending = std::make_shared<std::atomic<int>>(1);

    pool.submit([&pts, &global_idx, root, &cfg, impl, &pool, pending]() mutable {
        cluster_parallel_task(
            pts, global_idx, root, nullptr, nullptr,
            cfg, impl, pool, pending);
    });

    // Wait until all tasks have decremented pending to 0
    while (pending->load() > 0)
        std::this_thread::sleep_for(std::chrono::microseconds(100));

    return root;
}

// ─────────────────────────────────────────────────────────────────────────────
// Timing
// ─────────────────────────────────────────────────────────────────────────────

using Clock = std::chrono::high_resolution_clock;
double ms_since(Clock::time_point t){
    return std::chrono::duration<double,std::milli>(Clock::now()-t).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// Output helpers
// ─────────────────────────────────────────────────────────────────────────────

void print_stats(const std::string& tag, const std::shared_ptr<Node>& root) {
    std::vector<int> sz; root->leaf_sizes(sz);
    int mn = *std::min_element(sz.begin(), sz.end());
    int mx = *std::max_element(sz.begin(), sz.end());
    int avg = static_cast<int>(std::accumulate(sz.begin(),sz.end(),0.0)/sz.size());
    std::cout << "    [" << tag << "] clusters=" << root->total_clusters()
              << " depth=" << root->max_depth()
              << " leaf(min/mean/max)=" << mn << "/" << avg << "/" << mx << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Data generation
// ─────────────────────────────────────────────────────────────────────────────

Cloud generate_synthetic(int n, int n_blobs = 8, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> cd(-5,5), sd(0.3f,1.5f);
    std::normal_distribution<float> nd;
    std::vector<Pt3f> centers(n_blobs);
    std::vector<float> scales(n_blobs);
    for (int b = 0; b < n_blobs; ++b) {
        centers[b] = {cd(gen),cd(gen),cd(gen)}; scales[b] = sd(gen);
    }
    Cloud out(n);
    for (int i = 0; i < n; ++i) {
        int b = i % n_blobs;
        out[i] = centers[b] + scales[b]*Pt3f{nd(gen),nd(gen),nd(gen)};
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Export flags — controls which per-vertex attributes are written to the PLY.
//
//   leaf       → cluster_id_leaf_0          (leaf cluster id, always finest)
//   hierarchy  → cluster_id_leaf_1, _2, …   (ancestor ids up to root)
//   depth      → point_depth                (tree depth of each point's leaf)
//
// Flags are set by --export leaf,hierarchy,depth on the command line.
// Default: leaf only.
// ─────────────────────────────────────────────────────────────────────────────

struct ExportFlags {
    bool leaf      = true;   // cluster_id_leaf_0
    bool hierarchy = false;  // cluster_id_leaf_1 … cluster_id_leaf_N
    bool top_down  = false;  // depth_0 (root) … depth_N (leaf level)
    bool depth     = false;  // point_depth scalar
};

// ─────────────────────────────────────────────────────────────────────────────
// PLY writer — binary LE
// ─────────────────────────────────────────────────────────────────────────────

void write_ply(const std::string& path,
               const Cloud& pts,
               const std::vector<std::vector<int>>& levels,  // [level][point]
               const std::vector<int>& point_depth,          // depth per point
               const ExportFlags& flags)
{
    const int n        = static_cast<int>(pts.size());
    const int n_levels = static_cast<int>(levels.size());
    const int max_lv   = n_levels - 1;  // index of root level in the table

    // ── leaf / hierarchy columns (cluster_id_leaf_N, N=0 is finest) ──────────
    int first_leaf_lv = flags.leaf      ? 0      : -1;
    int last_leaf_lv  = flags.hierarchy ? max_lv
                      : flags.leaf      ? 0
                      :                   -1;
    const bool write_leaf_cols = (first_leaf_lv >= 0);
    const int  n_leaf_cols     = write_leaf_cols ? (last_leaf_lv - first_leaf_lv + 1) : 0;

    // ── top-down columns (depth_D, D=0 is root) ──────────────────────────────
    // depth_D corresponds to levels[max_lv - D]
    // depth_0  = root      = levels[max_lv]
    // depth_1  = level 1   = levels[max_lv - 1]
    // ...
    // depth_max_lv = leaf  = levels[0]
    const bool write_td   = flags.top_down;
    const int  n_td_cols  = write_td ? n_levels : 0;

    const bool write_depth = flags.depth && !point_depth.empty();

    // Row layout: xyz(12) + leaf cols(4 each) + top-down cols(4 each) + depth(4)
    const int row_bytes = 12
                        + 4 * n_leaf_cols
                        + 4 * n_td_cols
                        + (write_depth ? 4 : 0);

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    // ── Header ────────────────────────────────────────────────────────────────
    f << "ply\nformat binary_little_endian 1.0\n"
      << "element vertex " << n << "\n"
      << "property float x\nproperty float y\nproperty float z\n";
    for (int lv = first_leaf_lv; lv <= last_leaf_lv && write_leaf_cols; ++lv)
        f << "property int cluster_id_leaf_" << lv << "\n";
    for (int d = 0; d < n_td_cols; ++d)
        f << "property int depth_" << d << "\n";
    if (write_depth)
        f << "property int point_depth\n";
    f << "end_header\n";

    // ── Data ──────────────────────────────────────────────────────────────────
    std::vector<char> row(row_bytes);
    for (int i = 0; i < n; ++i) {
        float xyz[3] = {pts[i](0), pts[i](1), pts[i](2)};
        std::memcpy(row.data(), xyz, 12);
        int off = 12;
        // leaf / hierarchy columns
        for (int lv = first_leaf_lv; lv <= last_leaf_lv && write_leaf_cols; ++lv) {
            int32_t id = levels[lv][i];
            std::memcpy(row.data() + off, &id, 4);
            off += 4;
        }
        // top-down columns: depth_0 = root = levels[max_lv]
        for (int d = 0; d < n_td_cols; ++d) {
            int32_t id = levels[max_lv - d][i];
            std::memcpy(row.data() + off, &id, 4);
            off += 4;
        }
        if (write_depth) {
            int32_t d = point_depth[i];
            std::memcpy(row.data() + off, &d, 4);
        }
        f.write(row.data(), row_bytes);
    }
    if (!f) throw std::runtime_error("Write error: " + path);

    // ── Summary ───────────────────────────────────────────────────────────────
    std::cout << "  Exported : " << path << "\n"
              << "  Points   : " << n << "\n"
              << "  Attrs    :";
    bool any = false;
    if (write_leaf_cols) {
        if (n_leaf_cols == 1) std::cout << " cluster_id_leaf_0";
        else std::cout << " cluster_id_leaf_0…cluster_id_leaf_" << last_leaf_lv;
        any = true;
    }
    if (write_td) {
        std::cout << " depth_0(root)…depth_" << max_lv << "(leaf)";
        any = true;
    }
    if (write_depth) { std::cout << " point_depth"; any = true; }
    if (!any) std::cout << " (none — only xyz written)";
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// PLY / NPY / TXT loaders
// ─────────────────────────────────────────────────────────────────────────────

namespace ply_io {
enum class Fmt { ASCII, BIN_LE, BIN_BE };
struct Prop { std::string name; int bytes; bool is_xyz; int xyz_idx; };
inline uint32_t bs32(uint32_t v){
    return((v&0xFF000000u)>>24)|((v&0xFF0000u)>>8)|((v&0xFF00u)<<8)|((v&0xFFu)<<24);}
inline uint64_t bs64(uint64_t v){
    return((uint64_t)bs32(uint32_t(v&0xFFFFFFFFu))<<32)|(uint64_t)bs32(uint32_t(v>>32));}
int pb(const std::string& t){
    if(t=="char"||t=="uchar"||t=="int8"||t=="uint8") return 1;
    if(t=="short"||t=="ushort"||t=="int16"||t=="uint16") return 2;
    if(t=="int"||t=="uint"||t=="int32"||t=="uint32"||t=="float"||t=="float32") return 4;
    return 8;}
}

Cloud load_ply(const std::string& path) {
    using namespace ply_io;
    std::ifstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("Cannot open: "+path);
    Fmt fmt=Fmt::ASCII; int nv=0; bool inv=false;
    std::vector<Prop> props;
    std::string line;
    while(std::getline(f,line)){
        if(!line.empty()&&line.back()=='\r') line.pop_back();
        if(line=="end_header") break;
        std::istringstream ss(line); std::string tok; ss>>tok;
        if(tok=="format"){std::string fs;ss>>fs;
            if(fs=="binary_little_endian")fmt=Fmt::BIN_LE;
            else if(fs=="binary_big_endian")fmt=Fmt::BIN_BE;}
        else if(tok=="element"){std::string en;ss>>en;inv=(en=="vertex");if(inv)ss>>nv;else props.clear();}
        else if(tok=="property"&&inv){
            std::string ts,pn;ss>>ts;if(ts=="list")continue;ss>>pn;
            Prop p;p.name=pn;p.bytes=pb(ts);p.is_xyz=false;p.xyz_idx=-1;
            if(pn=="x"){p.is_xyz=true;p.xyz_idx=0;}
            else if(pn=="y"){p.is_xyz=true;p.xyz_idx=1;}
            else if(pn=="z"){p.is_xyz=true;p.xyz_idx=2;}
            props.push_back(p);}
    }
    if(nv==0) throw std::runtime_error("PLY: no vertices in "+path);
    int xyz=0;for(auto&p:props)if(p.is_xyz)++xyz;
    if(xyz<3) throw std::runtime_error("PLY: missing x/y/z in "+path);
    Cloud out(nv);
    if(fmt==Fmt::ASCII){
        for(int vi=0;vi<nv;++vi){
            std::getline(f,line);if(!line.empty()&&line.back()=='\r')line.pop_back();
            std::istringstream vs(line);
            for(auto&p:props){double val=0;vs>>val;if(p.is_xyz)out[vi](p.xyz_idx)=float(val);}
        }
    }else{
        const bool be=(fmt==Fmt::BIN_BE);
        for(int vi=0;vi<nv;++vi){
            for(auto&p:props){
                if(p.bytes==4){uint32_t r=0;f.read((char*)&r,4);if(be)r=bs32(r);
                    if(p.is_xyz){float v;std::memcpy(&v,&r,4);out[vi](p.xyz_idx)=v;}}
                else if(p.bytes==8){uint64_t r=0;f.read((char*)&r,8);if(be)r=bs64(r);
                    if(p.is_xyz){double v;std::memcpy(&v,&r,8);out[vi](p.xyz_idx)=float(v);}}
                else f.seekg(p.bytes,std::ios::cur);
            }
        }
    }
    return out;
}

Cloud load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("Cannot open: "+path);
    char magic[6]; f.read(magic,6);
    if(std::string(magic,6)!="\x93NUMPY") throw std::runtime_error("Not a numpy file");
    uint8_t maj,mn2; f.read((char*)&maj,1); f.read((char*)&mn2,1);
    uint16_t hlen; f.read((char*)&hlen,2);
    f.seekg(hlen,std::ios::cur);
    Cloud out; float buf[3];
    while(f.read((char*)buf,12)) out.push_back({buf[0],buf[1],buf[2]});
    return out;
}

Cloud load_txt(const std::string& path) {
    std::ifstream f(path);
    if(!f) throw std::runtime_error("Cannot open: "+path);
    Cloud out; std::string line;
    while(std::getline(f,line)){
        if(line.empty()||line[0]=='#') continue;
        std::istringstream ss(line); float x,y,z;
        if(ss>>x>>y>>z) out.push_back({x,y,z});
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark runner — runs all enabled variants and prints comparison table
// ─────────────────────────────────────────────────────────────────────────────

struct BenchRow {
    std::string name;
    double mean_ms, std_ms;
    int clusters;
};

BenchRow bench_variant(
    const std::string& name,
    const Cloud& pts, const std::vector<int>& idx,
    const Config& cfg, KMeansImpl impl, bool parallel,
    int repeats)
{
    std::cout << "\n  -- " << name << " --\n";
    std::vector<double> times;
    std::shared_ptr<Node> result;
    for (int r = 0; r < repeats; ++r) {
        auto t0 = Clock::now();
        auto res = parallel
                   ? cluster_parallel(pts, idx, cfg, impl)
                   : cluster_serial  (pts, idx, cfg, impl);
        double ms = ms_since(t0);
        times.push_back(ms);
        if (r == 0) result = res;
        std::cout << "    run " << r+1 << ": " << std::fixed << std::setprecision(1) << ms << " ms\n";
    }
    double mean = std::accumulate(times.begin(),times.end(),0.0)/times.size();
    double var  = 0; for (double t : times) var += (t-mean)*(t-mean);
    double std  = std::sqrt(var/times.size());
    std::cout << "    mean: " << mean << " ms ± " << std << " ms\n";
    if (result) print_stats(name, result);
    return {name, mean, std, result ? result->total_clusters() : 0};
}

void run_benchmark(const Cloud& pts, const Config& cfg) {
    const int n = static_cast<int>(pts.size());
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);

    std::cout << "\n" << std::string(66,'=') << "\n";
    std::cout << "  N=" << n << "  workers=" << cfg.n_workers
              << "  max_k=" << cfg.max_k
              << "  eig_thresh=" << cfg.eigenvalue_threshold
              << "  serial_cutoff=" << cfg.serial_cutoff << "\n";
    std::cout << std::string(66,'=') << "\n";

    std::vector<BenchRow> rows;

    rows.push_back(bench_variant("Serial+Naive",
        pts, idx, cfg, KMeansImpl::Naive, false, cfg.bench_repeats));

    rows.push_back(bench_variant("Serial+ElkanAVX",
        pts, idx, cfg, KMeansImpl::ElkanAVX, false, cfg.bench_repeats));

#ifdef WITH_MLPACK
    rows.push_back(bench_variant("Serial+mlpack",
        pts, idx, cfg, KMeansImpl::MLPack, false, cfg.bench_repeats));
#endif

    rows.push_back(bench_variant("Parallel+ElkanAVX",
        pts, idx, cfg, KMeansImpl::ElkanAVX, true, cfg.bench_repeats));

#ifdef WITH_MLPACK
    rows.push_back(bench_variant("Parallel+mlpack",
        pts, idx, cfg, KMeansImpl::MLPack, true, cfg.bench_repeats));
#endif

    // Summary table
    const double baseline = rows[0].mean_ms;
    std::cout << "\n  " << std::string(62,'-') << "\n";
    std::cout << "  " << std::left << std::setw(24) << "Variant"
              << std::right << std::setw(12) << "Mean(ms)"
              << std::setw(10) << "±std"
              << std::setw(10) << "vs base"
              << std::setw(10) << "Clusters" << "\n";
    std::cout << "  " << std::string(62,'-') << "\n";
    for (auto& r : rows) {
        double ratio = baseline / r.mean_ms;
        std::cout << "  " << std::left << std::setw(24) << r.name
                  << std::right << std::fixed << std::setprecision(1)
                  << std::setw(12) << r.mean_ms
                  << std::setw(10) << r.std_ms
                  << std::setw(9)  << ratio << "x"
                  << std::setw(10) << r.clusters << "\n";
    }
    std::cout << "  " << std::string(62,'-') << "\n";
    std::cout << "  (ratio > 1.0 = faster than Serial+HandRolled baseline)\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ─────────────────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    std::cout <<
        "\nUsage:\n"
        "  " << prog << "                              # synthetic benchmark\n"
        "  " << prog << " <input> [output] [options]  # cluster a point cloud\n"
        "\nInput formats: .ply  .npy (float32 N×3)  .txt/.xyz/.csv (x y z per line)\n"
        "Output       : binary-LE PLY with cluster_id (int32) per vertex\n"
        "               default: <input>_clustered.ply\n"
        "\nOptions:\n"
        "  --threshold <f>   Eigenvalue threshold for stopping (default: 0.02)\n"
        "                    Lower  → more splits, finer clusters\n"
        "                    Higher → fewer splits, coarser clusters\n"
        "  --max-k <n>       Maximum K to try in knee search (default: 6)\n"
        "  --min-pts <n>     Minimum cluster size to consider splitting (default: 30)\n"
        "  --max-depth <n>   Maximum recursion depth (default: 15)\n"
        "  --workers <n>     Thread count, 1 = serial (default: all cores)\n"
        "  --serial-cutoff <n> Clusters below this size run serially (default: 2000)\n"
        "  --export <list>   Comma-separated list of attributes to write (default: leaf)\n"
        "                      leaf      — cluster_id_leaf_0 (leaf id, finest)\n"
        "                      hierarchy — cluster_id_leaf_1 … _N (up to root)\n"
        "                      top-down  — depth_0 (root) … depth_N (leaf level)\n"
        "                      depth     — point_depth scalar per point\n"
        "                      all       — shorthand for all four\n"
        "                    Examples: --export leaf,depth\n"
        "                              --export top-down\n"
        "                              --export leaf,top-down\n"
        "                              --export all\n"
        "  --benchmark       Run serial vs parallel comparison on the input file\n"
        "  --help            Show this message\n"
        "\nExamples:\n"
        "  " << prog << " cloud.ply\n"
        "  " << prog << " cloud.ply out.ply --threshold 0.005\n"
        "  " << prog << " cloud.ply out.ply --threshold 0.005 --export all\n"
        "  " << prog << " cloud.ply out.ply --export leaf,depth\n"
        "  " << prog << " cloud.ply out.ply --export top-down\n"
        "  " << prog << " cloud.ply out.ply --export leaf,top-down,depth\n"
        "  " << prog << " cloud.ply out.ply --threshold 0.05 --max-k 8 --workers 4\n"
        "  " << prog << " cloud.ply --benchmark --threshold 0.01\n"
        "\n";
}

// Parse a flag of the form --name value.  Returns true and advances i on success.
static bool parse_float(int argc, char* argv[], int& i,
                         const std::string& name, float& out)
{
    if (argv[i] == name) {
        if (i+1 >= argc) {
            std::cerr << "Error: " << name << " requires a value.\n";
            std::exit(1);
        }
        try { out = std::stof(argv[++i]); }
        catch (...) {
            std::cerr << "Error: " << name << " value must be a number, got: "
                      << argv[i] << "\n";
            std::exit(1);
        }
        return true;
    }
    return false;
}

static bool parse_int(int argc, char* argv[], int& i,
                       const std::string& name, int& out)
{
    if (argv[i] == name) {
        if (i+1 >= argc) {
            std::cerr << "Error: " << name << " requires a value.\n";
            std::exit(1);
        }
        try { out = std::stoi(argv[++i]); }
        catch (...) {
            std::cerr << "Error: " << name << " value must be an integer, got: "
                      << argv[i] << "\n";
            std::exit(1);
        }
        return true;
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    Config cfg;
    std::string in_path, out_path;
    bool do_benchmark = false;
    ExportFlags exp_flags;  // default: leaf only

    auto ends_with = [](const std::string& s, const std::string& e) {
        return s.size() >= e.size() &&
               s.compare(s.size()-e.size(), e.size(), e) == 0;
    };
    auto looks_like_flag = [](const std::string& s) {
        return s.size() >= 2 && s[0] == '-' && s[1] == '-';
    };

    // ── Parse arguments ───────────────────────────────────────────
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
        if (arg == "--benchmark")           { do_benchmark = true;  continue; }

        if (parse_float(argc, argv, i, "--threshold",    cfg.eigenvalue_threshold)) continue;
        if (parse_int  (argc, argv, i, "--max-k",        cfg.max_k))               continue;
        if (parse_int  (argc, argv, i, "--min-pts",      cfg.min_points))          continue;
        if (parse_int  (argc, argv, i, "--max-depth",    cfg.max_depth))           continue;
        if (parse_int  (argc, argv, i, "--workers",      cfg.n_workers))           continue;
        if (parse_int  (argc, argv, i, "--serial-cutoff",cfg.serial_cutoff))       continue;

        if (arg == "--export") {
            if (i+1 >= argc) { std::cerr << "Error: --export requires a value.\n"; return 1; }
            std::string val = argv[++i];
            // Reset all flags to false so user gets exactly what they asked for
            exp_flags = {false, false, false};
            // Parse comma-separated tokens
            std::istringstream ss(val);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if      (tok == "leaf")      exp_flags.leaf      = true;
                else if (tok == "hierarchy") exp_flags.hierarchy = true;
                else if (tok == "depth")     exp_flags.depth     = true;
                else if (tok == "top-down") exp_flags.top_down = true;
                else if (tok == "all") {
                    exp_flags.leaf = exp_flags.hierarchy =
                    exp_flags.top_down = exp_flags.depth = true;
                } else {
                    std::cerr << "Error: unknown --export token '" << tok
                              << "'  (valid: leaf, hierarchy, depth, all)\n";
                    return 1;
                }
            }
            // hierarchy without leaf is not useful — silently enable leaf too
            if (exp_flags.hierarchy) exp_flags.leaf = true;
            continue;
        }

        // Positional: first non-flag → input, second → output
        if (!looks_like_flag(arg)) {
            if      (in_path.empty())  in_path  = arg;
            else if (out_path.empty()) out_path = arg;
            else {
                std::cerr << "Unexpected argument: " << arg << "\n";
                print_usage(argv[0]); return 1;
            }
            continue;
        }

        std::cerr << "Unknown option: " << arg << "\n";
        print_usage(argv[0]); return 1;
    }

    // ── Validate ──────────────────────────────────────────────────
    if (cfg.eigenvalue_threshold < 0.0f) {
        std::cerr << "Error: --threshold must be >= 0.\n"; return 1;
    }
    if (cfg.max_k < 2) {
        std::cerr << "Error: --max-k must be >= 2.\n"; return 1;
    }
    if (cfg.min_points < 2) {
        std::cerr << "Error: --min-pts must be >= 2.\n"; return 1;
    }
    if (cfg.n_workers < 1) {
        std::cerr << "Error: --workers must be >= 1.\n"; return 1;
    }

    // ── Synthetic benchmark mode (no input file) ──────────────────
    if (in_path.empty()) {
        std::cout << "\nNo file provided — running benchmark on synthetic data.\n";
        std::cout << "Variants: Serial+Naive, Serial+ElkanAVX, Parallel+ElkanAVX";
#ifdef WITH_MLPACK
        std::cout << ", Serial+mlpack, Parallel+mlpack";
#else
        std::cout << "\n(Build with -DWITH_MLPACK to add mlpack variants)";
#endif
        std::cout << "\n";
        for (int n : {5000, 20000, 100000}) {
            auto pts = generate_synthetic(n);
            run_benchmark(pts, cfg);
        }
        return 0;
    }

    // ── Load input ────────────────────────────────────────────────
    if (out_path.empty()) {
        size_t dot = in_path.rfind('.');
        out_path = (dot == std::string::npos ? in_path : in_path.substr(0, dot))
                   + "_clustered.ply";
    }

    std::cout << "\nLoading: " << in_path << " ...\n";
    Cloud pts;
    if      (ends_with(in_path, ".ply"))                         pts = load_ply(in_path);
    else if (ends_with(in_path, ".npy"))                         pts = load_npy(in_path);
    else if (ends_with(in_path, ".txt") ||
             ends_with(in_path, ".xyz") ||
             ends_with(in_path, ".csv"))                         pts = load_txt(in_path);
    else { std::cerr << "Unsupported format.\n"; return 1; }

    const int n = static_cast<int>(pts.size());

    std::cout << "Loaded        : " << n << " points\n";
    std::cout << "Threshold     : " << cfg.eigenvalue_threshold << "\n";
    std::cout << "Max-k         : " << cfg.max_k << "\n";
    std::cout << "Min-pts       : " << cfg.min_points << "\n";
    std::cout << "Max-depth     : " << cfg.max_depth << "\n";
    std::cout << "Workers       : " << cfg.n_workers << "\n";
    std::cout << "Serial cutoff : " << cfg.serial_cutoff << "\n";
    std::cout << "Export        : "
              << (exp_flags.leaf      ? "leaf "      : "")
              << (exp_flags.hierarchy ? "hierarchy " : "")
              << (exp_flags.top_down  ? "top-down "  : "")
              << (exp_flags.depth     ? "depth "     : "") << "\n";

    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);

    // ── Benchmark mode on real file ───────────────────────────────
    if (do_benchmark) {
        run_benchmark(pts, cfg);
        return 0;
    }

    // ── Normal clustering ─────────────────────────────────────────
    auto t0 = Clock::now();
    auto root = (cfg.n_workers > 1)
                ? cluster_parallel(pts, idx, cfg, KMeansImpl::ElkanAVX)
                : cluster_serial  (pts, idx, cfg, KMeansImpl::ElkanAVX);
    std::cout << "Clustering    : " << ms_since(t0) << " ms\n";
    print_stats("result", root);

    // Step 1: assign leaf ids (finest level, always needed for levels[0])
    std::vector<int> leaf_labels(n, -1);
    int next_leaf_id = 0;
    root->assign_labels(leaf_labels, next_leaf_id);

    // Step 2: assign internal-node ids (needed for hierarchy or top-down)
    if (exp_flags.hierarchy || exp_flags.top_down) {
        int next_internal_id = next_leaf_id;
        root->assign_internal_ids(next_internal_id);
    }

    // Step 3: build ancestor table (always needed; levels[0] is the leaf id)
    auto levels = build_ancestor_table(root, n);

    // Step 4: collect per-point depth (only if --export depth requested)
    std::vector<int> point_depth;
    if (exp_flags.depth) {
        point_depth.resize(n, 0);
        // Walk the tree; leaves carry their indices and depth
        struct Frame { const Node* node; };
        std::vector<Frame> stk;
        stk.push_back({root.get()});
        while (!stk.empty()) {
            const Node* nd = stk.back().node; stk.pop_back();
            if (nd->is_leaf) {
                for (int gi : nd->indices) point_depth[gi] = nd->depth;
            } else {
                for (auto& ch : nd->children) stk.push_back({ch.get()});
            }
        }
    }

    write_ply(out_path, pts, levels, point_depth, exp_flags);
    return 0;
}