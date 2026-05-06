# Cleave

Divisive K-Means clustering for 3D point clouds, designed for surface geometry. Cleave recursively splits a point cloud into clusters and stops when each cluster is geometrically flat — measured by the smallest eigenvalue of its local PCA covariance. The result is a tree of surface patches exported as a PLY file with per-vertex cluster attributes at every level of the hierarchy.

## How it works

Cleave runs a divisive (top-down) clustering loop:

1. **K selection** — runs K-Means for k = 1 … max\_k and picks the elbow of the inertia curve via second derivative
2. **K-Means** — Elkan's triangle-inequality algorithm with an AVX2-accelerated initial assignment pass, optimised for 3D data
3. **PCA stop condition** — computes the 3×3 covariance of each cluster and checks its smallest eigenvalue; if it falls below `--threshold`, the cluster is flat enough and splitting stops
4. **Recursion** — each sub-cluster is processed independently, in parallel across a thread pool

The output PLY carries the original point coordinates plus integer cluster-ID attributes at each level of the tree, so you can inspect the hierarchy at any granularity in CloudCompare, MeshLab, or your own tooling.

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| GCC or Clang | C++17 or later | |
| [Eigen3](https://eigen.tuxfamily.org) | 3.3+ | Header-only |
| CPU with AVX2 + FMA | — | Most x86 CPUs since 2013 |

**Ubuntu / Debian**

```bash
sudo apt install libeigen3-dev
```

**macOS (Homebrew)**

```bash
brew install eigen
```

## Getting the repository

```bash
git clone https://github.com/<your-username>/cleave.git
cd cleave
```

## Building

```bash
cd src
make
```

The Makefile auto-detects the Eigen3 include path via `pkg-config` and falls back to common system locations. To build manually:

```bash
g++ -O3 -march=native -mavx2 -mfma -std=c++17 -fopenmp \
    -I/usr/include/eigen3 \
    divisive_cluster.cpp -o cleave -lpthread
```

> **macOS note:** replace `-fopenmp` with `-Xpreprocessor -fopenmp -lomp` if using Apple Clang, or install `g++` via Homebrew and use that instead.

## Installing

Install system-wide (requires write access to `/usr/local/bin`):

```bash
cd src
sudo make install
```

Install to a custom prefix — no sudo needed if you own the directory:

```bash
make install PREFIX=~/.local        # → ~/.local/bin/cleave
make install PREFIX=/opt/cleave     # → /opt/cleave/bin/cleave
```

After installing to `~/.local`, make sure `~/.local/bin` is on your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc or ~/.zshrc
```

Then run from anywhere:

```bash
cleave cloud.ply --threshold 0.01
```

To remove:

```bash
sudo make uninstall                     # from /usr/local/bin
make uninstall PREFIX=~/.local          # from ~/.local/bin
```

### Optional: mlpack benchmark variant

Building with `MLPACK=1` adds mlpack's KMeans as an additional variant in `--benchmark` mode, letting you compare it against the hand-rolled implementation on your own data. It has no effect on normal clustering.

```bash
sudo apt install libmlpack-dev libarmadillo-dev libensmallen-dev
make MLPACK=1
sudo make install MLPACK=1
```

## Running

```bash
# Cluster a point cloud with default settings → cloud_clustered.ply
./cleave cloud.ply

# Explicit output path
./cleave cloud.ply out.ply

# Tune the stop threshold (lower = finer, more clusters)
./cleave cloud.ply out.ply --threshold 0.005

# Export the full hierarchy and depth attributes
./cleave cloud.ply out.ply --threshold 0.01 --export all

# Run the built-in benchmark on your file (serial vs parallel)
./cleave cloud.ply --benchmark

# Run the synthetic scaling benchmark (no input needed)
./cleave
```

## Options

```
--threshold <f>     Eigenvalue threshold for the PCA stop condition (default: 0.02)
                    Controls the flatness required before a cluster stops splitting.
                    Lower  → more splits, finer and more numerous clusters
                    Higher → fewer splits, coarser clusters

--max-k <n>         Maximum K to try during knee-method K selection (default: 6)

--min-pts <n>       Minimum number of points for a cluster to be eligible
                    for splitting (default: 30)

--max-depth <n>     Hard cap on recursion depth (default: 15)

--workers <n>       Number of threads (default: all logical cores)
                    Set to 1 to force serial execution

--serial-cutoff <n> Clusters smaller than this run serially regardless of
                    --workers, avoiding thread-dispatch overhead (default: 2000)

--export <list>     Comma-separated list of per-vertex attributes to include
                    in the output PLY (default: leaf). Tokens:

                      leaf       cluster_id_leaf_0
                                 Leaf-level cluster ID, the finest granularity.

                      hierarchy  cluster_id_leaf_1 … cluster_id_leaf_N
                                 One attribute per level going up from the leaf
                                 toward the root. Useful for traversing from a
                                 leaf node to its ancestors.
                                 Implies leaf.

                      top-down   depth_0 … depth_N
                                 One attribute per level going down from the root
                                 toward the leaves. depth_0 is the root (all points
                                 share one ID); depth_N is the leaf level.
                                 Useful for visualising the tree top-down.

                      depth      point_depth
                                 The tree depth of each point's leaf node as a
                                 scalar integer. Useful for analysing tree balance.

                      all        Shorthand for leaf,hierarchy,top-down,depth

--benchmark         Run a serial-vs-parallel timing comparison on the input file
                    instead of writing output. Useful for tuning --workers and
                    --serial-cutoff on your hardware.

--help              Print usage and exit
```

## Output format

Cleave writes a binary little-endian PLY file. All original coordinates are preserved as `float` (x, y, z). Cluster attributes are written as `int32` properties. The set of properties depends on `--export`:

```
property float x
property float y
property float z
property int cluster_id_leaf_0    # always present (--export leaf, the default)
property int cluster_id_leaf_1    # --export hierarchy
property int cluster_id_leaf_2
...
property int depth_0              # --export top-down  (root, 1 unique id)
property int depth_1
...
property int depth_N              # (leaf level, same ids as cluster_id_leaf_0)
property int point_depth          # --export depth
```

### Reading the attributes

**`cluster_id_leaf_N`** — indexed from the leaf upward. `leaf_0` is the finest clustering; each step up is a coarser grouping of those clusters. All points that share a `leaf_1` id are siblings of the same parent node.

**`depth_N`** — indexed from the root downward. `depth_0` is the root (every point gets the same id). Each step down adds one level of splits. `depth_N` at the deepest level is identical to `cluster_id_leaf_0`.

Both schemes use globally unique IDs across all levels, so you can cross-reference them without collisions.

### Viewing in CloudCompare

Open the output PLY in CloudCompare. Under **Properties → Color Scale**, select any of the `cluster_id_leaf_N` or `depth_N` scalar fields and apply a colour ramp. Switching between fields lets you interactively explore the hierarchy at different granularities.

## Supported input formats

| Extension | Format |
|---|---|
| `.ply` | ASCII, binary little-endian, binary big-endian; `float` or `double` xyz; extra properties (normals, colour, intensity) are skipped |
| `.npy` | NumPy binary, `float32`, shape `(N, 3)` |
| `.txt` / `.xyz` / `.csv` | One point per line: `x y z` |

## Choosing a threshold

The `--threshold` value is the smallest eigenvalue of the 3×3 PCA covariance of a cluster — roughly the variance of the cluster in its thinnest spatial direction. For surface point clouds this has a natural interpretation:

- A **flat wall patch** sampled at 1 cm resolution might have a smallest eigenvalue around 0.001–0.005
- A **curved surface** or **edge region** will have a larger smallest eigenvalue and keep splitting
- The **right threshold** depends on your point density and the scale of features you care about

A practical approach is to start with the default `0.02` and look at the result in CloudCompare, then halve or double the threshold until the cluster granularity matches what you need.

## License

Cleave is released under the [GNU General Public License v3.0](LICENSE).