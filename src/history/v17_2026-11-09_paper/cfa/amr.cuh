// Adaptive mesh: NVB refinement forest, marking, coarsening, look-ahead, transfer
// Part of cylinder_fem_amr; included in this order by cylinder_fem_amr.cu.
#pragma once

// ============================================================================
// Adaptive Mesh Refinement (AMR): Newest-Vertex Bisection Forest with Coarsening,
// Curved Cylinder Projection & Solution Transfer
// ============================================================================
struct AmrParams {
    int    max_level      = 4;       // Max AMR levels above the base mesh (1 level = 2 bisections)
    int    min_level      = 2;       // Every element stays at least this level (never coarsened below).
                                     // The indicators measure the disturbance of the Poiseuille
                                     // inflow, so without a floor the channel stays at the base
                                     // mesh, whose P1 approximation of the approach flow biased
                                     // C_D by -0.6 % (2D-2) and -2.2 % (2D-3)
    double tol_vort       = 0.1;     // Disturbance-vorticity criterion tolerance [m/s] (<= 0: off)
    double tol_interp     = 0.01;    // Disturbance interpolation-error criterion tolerance [m/s] (<= 0: off)
    double coarsen_safety = 0.35;    // Coarsen only if the merged parent's predicted eta < safety * tol
    int    buffer_layers  = 1;       // Vertex-neighbour layers added around refinement-marked elements
    int    max_elems      = 200000;  // Element budget (soft: closure may exceed it slightly)
    double lookahead_time = 0.0;     // Advect error indicators this far downstream [s] (0 = off)
    double keep_radius    = 0.25;    // No coarsening of elements whose centroid lies within this
                                     // distance of the cylinder centre [m] (0 = off)
    double tol_radius     = 0.0;     // > 0: both tolerances grow as (d / tol_radius)^tol_power beyond
    double tol_power      = 1.0;     // this distance d from the cylinder centre [m]: the forces are
                                     // insensitive to the far wake, which a uniform tolerance
                                     // resolves as finely as the cylinder (0 = off). Off by default:
                                     // in 2D-3 it costs C_L,max accuracy and brings back late remesh
                                     // glitches (HANDOVER 11.16)
};

struct AmrStats {
    int    leaves_before      = 0;
    int    leaves_after       = 0;
    int    marked_refine      = 0;
    int    marked_coarsen     = 0;
    int    bisections         = 0;   // Bisections of marked elements
    int    closure_bisections = 0;   // Bisections required for conformity
    int    removed_vertices   = 0;   // Coarsening: newest vertices removed
    double elapsed_ms         = 0.0;
    // Phase breakdown of elapsed_ms
    double ms_lookahead       = 0.0;   // advective look-ahead
    double ms_mark            = 0.0;   // normalisation, sort, tolerance marking
    double ms_neighbours      = 0.0;   // edge adjacency, hole filling, buffer layers
    double ms_coarsen         = 0.0;
    double ms_refine          = 0.0;   // bisection + conformity closure
};

struct AmrElem {
    int  v[3];                 // v[0] = newest vertex, (v[1], v[2]) = refinement edge, CCW
    int  gen      = 0;         // Bisection generation
    int  parent   = -1;
    int  child[2] = {-1, -1};
    bool alive    = false;
    bool is_leaf() const { return alive && child[0] < 0; }
};

class AdaptiveMesh {
public:
    // Accumulated sub-phase times [ms] for profiling (printed after [Timing])
    enum { D_ADJ, D_TRACE, D_MERGE, D_COLLECT, D_KEYS, D_SORT, D_COMPACT, D_FILL, D_COUNT };
    double dbg_ms[D_COUNT] = {0, 0, 0, 0, 0, 0, 0, 0};
private:
    std::chrono::steady_clock::time_point dbg_t;
    void dbg_start() { dbg_t = std::chrono::steady_clock::now(); }
    void dbg_lap(int k) {
        auto now = std::chrono::steady_clock::now();
        dbg_ms[k] += std::chrono::duration<double, std::milli>(now - dbg_t).count();
        dbg_t = now;
    }
public:
    AdaptiveMesh(const Mesh2D& base,
                 const std::vector<double>& u0,
                 const std::vector<double>& v0,
                 const std::vector<double>& p0) {
        x = base.x;  y = base.y;  bc_tag = base.bc_tag;
        u = u0;      v = v0;      p = p0;
        uo = u0;     vo = v0;
        int nv = base.num_nodes;
        v_alive.assign(nv, 1);
        v_parents.assign(nv, {-1, -1});
        v_edge.assign(nv, 0);

        elems.resize(base.num_elems);
        for (int e = 0; e < base.num_elems; ++e) {
            for (int k = 0; k < 3; ++k) elems[e].v[k] = base.elems[e].v[k];
            elems[e].gen   = 0;
            elems[e].alive = true;
        }
    }

    int num_leaves() const { return (int)leaf_ids.size(); }
    // Compact node of the last extract() -> persistent vertex slot (slots of removed vertices are recycled)
    const std::vector<int>& node_vertices() const { return vert_ids; }
    int num_vertex_slots() const { return (int)x.size(); }

    double inlet_umax = SimConfig::U_MAX;   // peak inflow velocity imposed on new inlet midpoints

    /**
     * Copies the solver solution (compact numbering of the last extract()) back onto
     * the persistent vertices of the hierarchy. (h_uo, h_vo) is the previous time level
     * u^{n-1}, transferred with the same rules so BDF2 keeps its history across remeshing.
     */
    void scatter_solution(const std::vector<double>& h_u,
                          const std::vector<double>& h_v,
                          const std::vector<double>& h_p,
                          const std::vector<double>& h_uo,
                          const std::vector<double>& h_vo) {
        for (size_t i = 0; i < vert_ids.size(); ++i) {
            u[vert_ids[i]]  = h_u[i];
            v[vert_ids[i]]  = h_v[i];
            p[vert_ids[i]]  = h_p[i];
            uo[vert_ids[i]] = h_uo[i];
            vo[vert_ids[i]] = h_vo[i];
        }
    }

    /**
     * Repeats the solution transfer of the last adapt() for a later solution on the mesh it started
     * from (--amr-overlap: the mesh is built from the state at step n while the GPU steps on, and the
     * state at step n + k is moved to it). old_vids: node -> vertex slot of that mesh, h_*: its
     * solution; the result is on the nodes of the last extract(). The rules are those of adapt():
     * kept vertices keep their values (coarsening is nodal injection), midpoints created by adapt()
     * get set_midpoint_values in creation order (with the current inlet_umax). Coarsening runs before
     * refinement, so a slot freed and reused within one adapt() is in the creation log.
     */
    void retransfer(const std::vector<int>& old_vids,
                    const std::vector<double>& h_u, const std::vector<double>& h_v, const std::vector<double>& h_p,
                    const std::vector<double>& h_uo, const std::vector<double>& h_vo,
                    std::vector<double>& n_u, std::vector<double>& n_v, std::vector<double>& n_p,
                    std::vector<double>& n_uo, std::vector<double>& n_vo) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < (int)old_vids.size(); ++j) {
            int s = old_vids[j];
            u[s] = h_u[j];  v[s] = h_v[j];  p[s] = h_p[j];  uo[s] = h_uo[j];  vo[s] = h_vo[j];
        }
        for (const auto& c : created_log) set_midpoint_values(c[0], c[1], c[2]);
        const int nn = (int)vert_ids.size();
        n_u.resize(nn);  n_v.resize(nn);  n_p.resize(nn);  n_uo.resize(nn);  n_vo.resize(nn);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nn; ++i) {
            int s = vert_ids[i];
            n_u[i] = u[s];  n_v[i] = v[s];  n_p[i] = p[s];  n_uo[i] = uo[s];  n_vo[i] = vo[s];
        }
    }

    /**
     * Builds the compact solver mesh (leaf elements, referenced vertices) and solution.
     * Elements are ordered along a Morton (Z-order) curve and vertices by first use,
     * which keeps neighbouring unknowns close in memory for the GPU SpMV and assembly.
     */
    void extract(Mesh2D& mesh,
                 std::vector<double>& h_u,
                 std::vector<double>& h_v,
                 std::vector<double>& h_p,
                 std::vector<double>& h_uo,
                 std::vector<double>& h_vo) {
        dbg_start();
        leaf_ids.clear();
        for (int e = 0; e < (int)elems.size(); ++e) {
            if (elems[e].is_leaf()) leaf_ids.push_back(e);
        }
        dbg_lap(D_COLLECT);

        // Keys stored next to the ids: the same comparisons as an indirect sort, without the
        // cache misses
        std::vector<std::pair<uint32_t, int>> keyed(leaf_ids.size());
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)leaf_ids.size(); ++i) {
            const AmrElem& T = elems[leaf_ids[i]];
            double xc = (x[T.v[0]] + x[T.v[1]] + x[T.v[2]]) / 3.0;
            double yc = (y[T.v[0]] + y[T.v[1]] + y[T.v[2]]) / 3.0;
            keyed[i] = {morton_code(xc / SimConfig::L_CHANNEL, yc / SimConfig::H_CHANNEL), leaf_ids[i]};
        }
        dbg_lap(D_KEYS);
        // Ties are broken by forest id (ids are ascending in the input), which is a stable sort and
        // does not depend on the thread count
        auto by_key = [](const std::pair<uint32_t, int>& a, const std::pair<uint32_t, int>& b) {
            return a.first != b.first ? a.first < b.first : a.second < b.second;
        };
#ifdef _OPENMP
        __gnu_parallel::sort(keyed.begin(), keyed.end(), by_key);
#else
        std::sort(keyed.begin(), keyed.end(), by_key);
#endif
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)keyed.size(); ++i) leaf_ids[i] = keyed[i].second;
        dbg_lap(D_SORT);

        std::vector<int> compact(x.size(), -1);
        vert_ids.clear();
        for (int e : leaf_ids) {
            for (int k = 0; k < 3; ++k) {
                int vid = elems[e].v[k];
                if (compact[vid] < 0) {
                    compact[vid] = (int)vert_ids.size();
                    vert_ids.push_back(vid);
                }
            }
        }

        dbg_lap(D_COMPACT);
        int nn = (int)vert_ids.size();
        mesh.x.resize(nn);  mesh.y.resize(nn);  mesh.bc_tag.resize(nn);
        h_u.resize(nn);     h_v.resize(nn);     h_p.resize(nn);
        h_uo.resize(nn);    h_vo.resize(nn);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nn; ++i) {
            int vid = vert_ids[i];
            mesh.x[i] = x[vid];  mesh.y[i] = y[vid];  mesh.bc_tag[i] = bc_tag[vid];
            h_u[i] = u[vid];     h_v[i] = v[vid];     h_p[i] = p[vid];
            h_uo[i] = uo[vid];   h_vo[i] = vo[vid];
        }

        mesh.elems.resize(leaf_ids.size());
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)leaf_ids.size(); ++i) {
            const AmrElem& T = elems[leaf_ids[i]];
            Triangle tri;
            for (int k = 0; k < 3; ++k) tri.v[k] = compact[T.v[k]];
            tri.gen   = T.gen;
            tri.level = T.gen / 2;
            mesh.elems[i] = tri;
        }
        mesh.num_nodes = nn;
        mesh.num_elems = (int)leaf_ids.size();
        dbg_lap(D_FILL);
    }

    /**
     * One adaptation cycle (compact numbering of the last extract(); call extract() afterwards).
     * `eta` holds 2 * L raw indicators [m/s]: [0, L) disturbance vorticity, [L, 2L) disturbance
     * interpolation error. Each is normalised by its tolerance, a_c = eta_c / tol_c, and an
     * element is refined if any criterion exceeds 1 (multi-criterion tagging). Criterion c
     * scales as h^q_c (q = 1 vorticity, q = 2 interpolation): one level (h/2) divides a_c by
     * 2^q_c, one bisection by 2^(q_c/2).
     */
    AmrStats adapt(const std::vector<double>& eta, const AmrParams& prm) {
        auto t_start = std::chrono::steady_clock::now();
        AmrStats st;
        created_log.clear();
        const int max_gen = 2 * prm.max_level;
        const int L = (int)leaf_ids.size();
        st.leaves_before = L;

        constexpr double Q_VORT = 1.0, Q_INTERP = 2.0;
        const bool use_v = prm.tol_vort > 0.0;
        const bool use_i = prm.tol_interp > 0.0;

        // ---- 1. Normalised indicators on forest element ids ----
        std::vector<double> av(elems.size(), 0.0), ai(elems.size(), 0.0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < L; ++i) {
            double w = 1.0;   // tolerance growth with the distance from the cylinder (tol_radius)
            if (prm.tol_radius > 0.0) {
                const AmrElem& T = elems[leaf_ids[i]];
                double xc = (x[T.v[0]] + x[T.v[1]] + x[T.v[2]]) / 3.0 - SimConfig::X_CYL;
                double yc = (y[T.v[0]] + y[T.v[1]] + y[T.v[2]]) / 3.0 - SimConfig::Y_CYL;
                double d = std::sqrt(xc * xc + yc * yc);
                if (d > prm.tol_radius) w = std::pow(d / prm.tol_radius, prm.tol_power);
            }
            if (use_v) av[leaf_ids[i]] = eta[i] / (prm.tol_vort * w);
            if (use_i) ai[leaf_ids[i]] = eta[L + i] / (prm.tol_interp * w);
        }

        // Edge neighbours of every leaf, shared by the look-ahead walk and hole filling
        auto t_look = std::chrono::steady_clock::now();
        dbg_start();
        build_leaf_adjacency();
        dbg_lap(D_ADJ);
        buckets_built = false;

        // ---- 2. Advective look-ahead: features above tolerance are carried downstream so
        //         the mesh is ready where they will be at the next adaptation ----
        if (prm.lookahead_time > 0.0) {
            apply_lookahead(use_v ? &av : nullptr, use_i ? &ai : nullptr, prm.lookahead_time, Q_VORT, Q_INTERP);
        }
        st.ms_lookahead = ms_since(t_look);

        auto score = [&](int e) { return std::max(av[e], ai[e]); };
        // Predicted normalised indicator of the parent of a bisection (h * sqrt(2))
        auto parent_score = [&](int e) {
            return std::max(std::sqrt(2.0) * av[e], 2.0 * ai[e]);
        };

        // Leaves above tolerance, largest score first (ties in leaf order). Only these are
        // visited by the marking loop, so the rest is not sorted.
        std::vector<std::pair<double, int>> keyed;
        for (int e : leaf_ids) {
            if (score(e) > 1.0) keyed.push_back({score(e), e});
        }
        std::stable_sort(keyed.begin(), keyed.end(),
                         [](const std::pair<double, int>& a, const std::pair<double, int>& b) { return a.first > b.first; });
        std::vector<int> order(keyed.size());
        for (size_t i = 0; i < keyed.size(); ++i) order[i] = keyed[i].second;

        // ---- 3. Tolerance-based refinement marking, largest first within the budget. Each
        //         element gets the number of levels its worst criterion needs to meet tol ----
        std::vector<int> target(elems.size(), -1);   // target generation of refined leaves
        std::vector<char> refine(elems.size(), 0);
        long long predicted = L;
        auto mark = [&](int e, int tgt) {
            int gen = elems[e].gen;
            tgt = std::min(max_gen, tgt);
            if (tgt <= gen) return false;
            long long added = (1LL << (tgt - gen)) - 1;
            if (predicted + added > prm.max_elems) return false;
            predicted += added;
            refine[e] = 1;
            target[e] = tgt;
            ++st.marked_refine;
            return true;
        };
        for (int e : order) {
            if (score(e) <= 1.0) break;
            int levels = 1;
            if (av[e] > 1.0) levels = std::max(levels, (int)std::ceil(std::log2(av[e]) / Q_VORT));
            if (ai[e] > 1.0) levels = std::max(levels, (int)std::ceil(std::log2(ai[e]) / Q_INTERP));
            mark(e, elems[e].gen + 2 * levels);
        }
        if (prm.min_level > 0) {
            for (int e : leaf_ids) {
                if (!refine[e] && elems[e].gen < 2 * prm.min_level) mark(e, 2 * prm.min_level);
            }
        }

        st.ms_mark = ms_since(t_start) - st.ms_lookahead;
        auto t_phase = std::chrono::steady_clock::now();

        // ---- 4. Hole filling: a leaf that is coarser than at least two of its edge
        //         neighbours, and not far below tolerance, is refined one level. This removes
        //         isolated coarse cells (speckles) inside resolved regions ----
        auto planned_level = [&](int e) { return std::max(elems[e].gen, target[e]) / 2; };
        auto finer_neighbours = [&](int e) {
            const int* nb = &leaf_nbr[3 * leaf_pos[e]];
            int lvl = planned_level(e), count = 0;
            for (int k = 0; k < 3; ++k) {
                if (nb[k] >= 0 && planned_level(leaf_ids[nb[k]]) > lvl) ++count;
            }
            return count;
        };
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<int> fill;
            for (int e : leaf_ids) {
                if (refine[e] || elems[e].gen >= max_gen) continue;
                if (score(e) > prm.coarsen_safety && finer_neighbours(e) >= 2) fill.push_back(e);
            }
            for (int e : fill) mark(e, elems[e].gen + 2);
        }

        // ---- 5. Buffer layers: vertex neighbours get at least one level ----
        std::vector<char> vflag(x.size(), 0);
        for (int layer = 0; layer < prm.buffer_layers; ++layer) {
            std::fill(vflag.begin(), vflag.end(), 0);
            for (int e : leaf_ids) {
                if (!refine[e]) continue;
                for (int k = 0; k < 3; ++k) vflag[elems[e].v[k]] = 1;
            }
            for (int e : leaf_ids) {
                const AmrElem& T = elems[e];
                if (refine[e] || T.gen >= max_gen) continue;
                if (vflag[T.v[0]] || vflag[T.v[1]] || vflag[T.v[2]]) mark(e, T.gen + 2);
            }
        }

        // One more layer is protected from coarsening
        std::fill(vflag.begin(), vflag.end(), 0);
        for (int e : leaf_ids) {
            if (!refine[e]) continue;
            for (int k = 0; k < 3; ++k) vflag[elems[e].v[k]] = 1;
        }

        st.ms_neighbours = ms_since(t_phase);
        t_phase = std::chrono::steady_clock::now();

        // ---- 6. Coarsening: undo bisections where the merged parent is predicted below
        //         coarsen_safety on every criterion. The band between coarsen_safety and 1 is
        //         the hysteresis that prevents refine/coarsen cycling. Leaves that are already
        //         coarser than two neighbours are kept (no new holes) ----
        std::vector<char> coarsen(elems.size(), 0);
        std::vector<int> coarsen_leaves;
        for (int e : leaf_ids) {
            const AmrElem& T = elems[e];
            if (refine[e] || T.gen <= 2 * prm.min_level) continue;
            if (vflag[T.v[0]] || vflag[T.v[1]] || vflag[T.v[2]]) continue;
            if (prm.keep_radius > 0.0) {
                double xc = (x[T.v[0]] + x[T.v[1]] + x[T.v[2]]) / 3.0 - SimConfig::X_CYL;
                double yc = (y[T.v[0]] + y[T.v[1]] + y[T.v[2]]) / 3.0 - SimConfig::Y_CYL;
                if (xc * xc + yc * yc < prm.keep_radius * prm.keep_radius) continue;
            }
            if (parent_score(e) < prm.coarsen_safety && finer_neighbours(e) < 2) {
                coarsen[e] = 1;
                coarsen_leaves.push_back(e);
                ++st.marked_coarsen;
            }
        }

        // A removable vertex m is the newest vertex v[0] of coarsening-marked leaves (the
        // children of the elements bisected through m), so only those vertices are examined
        std::vector<char> seen(x.size(), 0);
        for (int sweep = 0; sweep < 2; ++sweep) {
            std::vector<int> candidates, examined;
            for (int e : coarsen_leaves) {
                if (!elems[e].is_leaf() || elems[e].gen == 0) continue;
                int m = elems[e].v[0];
                if (seen[m]) continue;
                seen[m] = 1;
                examined.push_back(m);
                bool ok = false;
                for (int s = 0; s < 2; ++s) {
                    int pe = v_parents[m][s];
                    if (pe < 0) continue;
                    const AmrElem& P = elems[pe];
                    int c0 = P.child[0], c1 = P.child[1];
                    if (!P.alive || c0 < 0 ||
                        !elems[c0].is_leaf() || !elems[c1].is_leaf() ||
                        !coarsen[c0] || !coarsen[c1] ||
                        std::max(parent_score(c0), parent_score(c1)) >= prm.coarsen_safety) {
                        ok = false;
                        break;
                    }
                    ok = true;
                }
                if (ok) candidates.push_back(m);
            }
            for (int m : examined) seen[m] = 0;
            std::sort(candidates.begin(), candidates.end());
            coarsen_leaves.clear();
            for (int m : candidates) {
                for (int pe : v_parents[m]) if (pe >= 0) coarsen_leaves.push_back(pe);   // merged parents
                collapse_vertex(m, coarsen, av, ai);
                ++st.removed_vertices;
            }
            if (candidates.empty()) break;
        }

        st.ms_coarsen = ms_since(t_phase);
        t_phase = std::chrono::steady_clock::now();

        // ---- 7. Refinement: NVB bisection of marked leaves (and their descendants) until
        //         each reaches its target generation; children inherit the target. Each round
        //         bisects the leaves that still have a target, then closes the mesh. Only marked
        //         leaves and their children can carry a target, and only leaves next to a new
        //         bisection can become hanging, so neither step rescans the forest; both bisect
        //         in ascending element id, the order of a full scan ----
        refine_to_targets(target, st);

        st.ms_refine = ms_since(t_phase);

        int leaves = 0;
        for (const auto& T : elems) leaves += T.is_leaf() ? 1 : 0;
        st.leaves_after = leaves;
        st.elapsed_ms = ms_since(t_start);
        return st;
    }

    /**
     * Uniform refinement: every leaf of the current mesh (last extract()) is refined by
     * `levels` levels (2 bisections each). Call extract() afterwards.
     */
    AmrStats refine_uniform(int levels) {
        AmrStats st;
        st.leaves_before = (int)leaf_ids.size();
        std::vector<int> target(elems.size(), -1);
        for (int e : leaf_ids) target[e] = elems[e].gen + 2 * levels;
        refine_to_targets(target, st);
        int leaves = 0;
        for (const auto& T : elems) leaves += T.is_leaf() ? 1 : 0;
        st.leaves_after = leaves;
        return st;
    }

private:
    /**
     * Bisects the leaves of the last extract() with target[e] > gen (and their children)
     * until each reaches its target generation, keeping the mesh conforming.
     */
    void refine_to_targets(std::vector<int>& target, AmrStats& st) {
        build_vertex_leaves();
        std::vector<int> work, new_children, next;
        std::vector<std::pair<int, int>> new_edges;
        for (int e : leaf_ids) {
            if (target[e] > elems[e].gen) work.push_back(e);   // marked leaves are never coarsened
        }
        std::sort(work.begin(), work.end());
        while (!work.empty()) {
            new_children.clear();
            new_edges.clear();
            for (int e : work) {
                bisect(e, &new_children, &new_edges);
                ++st.bisections;
                target.resize(elems.size(), -1);
                target[elems[e].child[0]] = target[e];
                target[elems[e].child[1]] = target[e];
            }
            st.closure_bisections += make_conforming(new_children, new_edges);
            target.resize(elems.size(), -1);
            next.clear();
            for (int e : work) {
                for (int c : elems[e].child) {
                    if (elems[c].is_leaf() && target[c] > elems[c].gen) next.push_back(c);
                }
            }
            std::sort(next.begin(), next.end());
            work.swap(next);
        }

    }

    // Persistent vertex storage (slots of removed vertices are recycled)
    std::vector<double> x, y, u, v, p;
    std::vector<double> uo, vo;                  // previous time level u^{n-1}, v^{n-1}
    std::vector<int>    bc_tag;
    std::vector<char>   v_alive;
    std::vector<std::array<int, 2>> v_parents;   // Elements bisected through this (newest) vertex
    std::vector<uint64_t> v_edge;                // Bisected edge key of a midpoint vertex
    std::vector<int> free_verts;

    // Refinement forest (slots of removed elements are recycled)
    std::vector<AmrElem> elems;
    std::vector<int> free_elems;

    // Bisected edge -> midpoint vertex
    std::unordered_map<uint64_t, int> edge_mid;
    // Midpoints created by the current adapt(), in creation order: {m, a, b} (retransfer)
    std::vector<std::array<int, 3>> created_log;

    // Compact numbering of the last extract()
    std::vector<int> leaf_ids;   // compact element -> forest element
    std::vector<int> vert_ids;   // compact node    -> persistent vertex

    // Leaf adjacency of the current adapt() (build_leaf_adjacency)
    std::vector<int> leaf_pos;   // forest element -> compact element (-1: not a leaf)
    std::vector<int> leaf_nbr;   // [3 * compact]: compact neighbour across the edge opposite v[k], -1 on the boundary
    std::vector<int> inc_ptr, inc_list;   // vertex -> incident leaves (CSR, compact ids)

    // Vertex -> leaves (forest ids) at the start of the refinement phase of adapt()
    std::vector<int> vleaf_ptr, vleaf_list;

    // Uniform bucket grid over the leaves, built on first use in an adapt() (look-ahead fallback)
    static constexpr double BUCKET_CELL = 0.01;
    bool buckets_built = false;
    int bucket_nx = 0, bucket_ny = 0;
    std::vector<int> bucket_ptr, bucket_items;

    static uint32_t spread_bits(uint32_t n) {
        n &= 0x0000ffffu;
        n = (n | (n << 8)) & 0x00ff00ffu;
        n = (n | (n << 4)) & 0x0f0f0f0fu;
        n = (n | (n << 2)) & 0x33333333u;
        n = (n | (n << 1)) & 0x55555555u;
        return n;
    }

    static uint32_t morton_code(double sx, double sy) {
        auto quantize = [](double s) {
            return (uint32_t)std::min(65535.0, std::max(0.0, s * 65535.0));
        };
        return spread_bits(quantize(sx)) | (spread_bits(quantize(sy)) << 1);
    }

    double leaf_area(int e) const {
        const AmrElem& T = elems[e];
        return 0.5 * std::fabs((x[T.v[1]] - x[T.v[0]]) * (y[T.v[2]] - y[T.v[0]]) -
                               (x[T.v[2]] - x[T.v[0]]) * (y[T.v[1]] - y[T.v[0]]));
    }

    /**
     * Edge neighbours of every leaf of the (conforming) mesh of the last extract(), from the
     * vertex -> leaf incidence: leaf_nbr[3 i + k] is the leaf sharing the edge opposite v[k].
     */
    void build_leaf_adjacency() {
        const int L = (int)leaf_ids.size();
        leaf_pos.assign(elems.size(), -1);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < L; ++i) leaf_pos[leaf_ids[i]] = i;

        // Vertex -> leaf incidence. The order inside a vertex's list does not matter: on a conforming
        // mesh exactly one other leaf shares an edge.
        inc_ptr.assign(x.size() + 1, 0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < L; ++i)
            for (int k = 0; k < 3; ++k) {
                #pragma omp atomic
                inc_ptr[elems[leaf_ids[i]].v[k] + 1]++;
            }
        for (size_t i = 0; i < x.size(); ++i) inc_ptr[i + 1] += inc_ptr[i];
        inc_list.resize(inc_ptr[x.size()]);
        std::vector<int> fill(inc_ptr.begin(), inc_ptr.end() - 1);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < L; ++i)
            for (int k = 0; k < 3; ++k) {
                int slot;
                #pragma omp atomic capture
                slot = fill[elems[leaf_ids[i]].v[k]]++;
                inc_list[slot] = i;
            }

        leaf_nbr.assign(3 * L, -1);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < L; ++i) {
            const AmrElem& T = elems[leaf_ids[i]];
            for (int k = 0; k < 3; ++k) {
                int a = T.v[(k + 1) % 3], b = T.v[(k + 2) % 3];
                for (int s = inc_ptr[a]; s < inc_ptr[a + 1]; ++s) {
                    int j = inc_list[s];
                    if (j == i) continue;
                    const int* w = elems[leaf_ids[j]].v;
                    if (w[0] == b || w[1] == b || w[2] == b) { leaf_nbr[3 * i + k] = j; break; }
                }
            }
        }
    }

    // Barycentric coordinates of (px, py) in leaf e; true if inside (with a 1e-12 tolerance)
    bool barycentric(int e, double px, double py, double l[3]) const {
        const AmrElem& T = elems[e];
        double x0 = x[T.v[0]], y0 = y[T.v[0]];
        double d  = (x[T.v[1]] - x0) * (y[T.v[2]] - y0) - (x[T.v[2]] - x0) * (y[T.v[1]] - y0);
        l[1] = ((px - x0) * (y[T.v[2]] - y0) - (x[T.v[2]] - x0) * (py - y0)) / d;
        l[2] = ((x[T.v[1]] - x0) * (py - y0) - (px - x0) * (y[T.v[1]] - y0)) / d;
        l[0] = 1.0 - l[1] - l[2];
        return l[1] >= -1.0e-12 && l[2] >= -1.0e-12 && l[1] + l[2] <= 1.0 + 1.0e-12;
    }

    void build_buckets() {
        const double cell = BUCKET_CELL;
        bucket_nx = (int)std::ceil(SimConfig::L_CHANNEL / cell);
        bucket_ny = (int)std::ceil(SimConfig::H_CHANNEL / cell);
        const int nx = bucket_nx, ny = bucket_ny;
        auto clampi = [](int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); };
        auto bucket_range = [&](int e, int& i0, int& i1, int& j0, int& j1) {
            const AmrElem& T = elems[e];
            double xmin = std::min({x[T.v[0]], x[T.v[1]], x[T.v[2]]});
            double xmax = std::max({x[T.v[0]], x[T.v[1]], x[T.v[2]]});
            double ymin = std::min({y[T.v[0]], y[T.v[1]], y[T.v[2]]});
            double ymax = std::max({y[T.v[0]], y[T.v[1]], y[T.v[2]]});
            i0 = clampi((int)(xmin / cell), 0, nx - 1);  i1 = clampi((int)(xmax / cell), 0, nx - 1);
            j0 = clampi((int)(ymin / cell), 0, ny - 1);  j1 = clampi((int)(ymax / cell), 0, ny - 1);
        };
        bucket_ptr.assign(nx * ny + 1, 0);
        for (int e : leaf_ids) {
            int i0, i1, j0, j1;
            bucket_range(e, i0, i1, j0, j1);
            for (int j = j0; j <= j1; ++j)
                for (int i = i0; i <= i1; ++i) bucket_ptr[j * nx + i + 1]++;
        }
        for (int b = 0; b < nx * ny; ++b) bucket_ptr[b + 1] += bucket_ptr[b];
        bucket_items.resize(bucket_ptr[nx * ny]);
        std::vector<int> fill(bucket_ptr.begin(), bucket_ptr.end() - 1);
        for (int e : leaf_ids) {
            int i0, i1, j0, j1;
            bucket_range(e, i0, i1, j0, j1);
            for (int j = j0; j <= j1; ++j)
                for (int i = i0; i <= i1; ++i) bucket_items[fill[j * nx + i]++] = e;
        }
        #pragma omp atomic write seq_cst
        buckets_built = true;
    }

    // Leaf containing (px, py) from the bucket grid, -1 outside the fluid domain
    int locate_bucket(double px, double py) {
        if (px < 0.0 || px > SimConfig::L_CHANNEL || py < 0.0 || py > SimConfig::H_CHANNEL) return -1;
        bool built;
        #pragma omp atomic read seq_cst
        built = buckets_built;
        if (!built) {
            #pragma omp critical(amr_buckets)
            {
                if (!buckets_built) build_buckets();
            }
        }
        auto clampi = [](int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); };
        int b = clampi((int)(py / BUCKET_CELL), 0, bucket_ny - 1) * bucket_nx
              + clampi((int)(px / BUCKET_CELL), 0, bucket_nx - 1);
        double l[3];
        for (int k = bucket_ptr[b]; k < bucket_ptr[b + 1]; ++k) {
            if (barycentric(bucket_items[k], px, py, l)) return bucket_items[k];
        }
        return -1;
    }

    /**
     * Leaf containing (px, py), found by walking across edges from leaf `start` towards the
     * point (to the neighbour opposite the most negative barycentric coordinate). The look-ahead
     * moves less than an element per sample, so this is usually zero or one hop. A walk that
     * leaves the mesh (cylinder, channel walls) or does not settle falls back to the bucket grid.
     */
    int locate_from(int start, double px, double py) {
        int e = start;
        double l[3];
        for (int hop = 0; hop < 256; ++hop) {
            if (barycentric(e, px, py, l)) return e;
            int k = (l[0] < l[1]) ? (l[0] < l[2] ? 0 : 2) : (l[1] < l[2] ? 1 : 2);
            int nb = leaf_nbr[3 * leaf_pos[e] + k];
            if (nb < 0) break;
            e = leaf_ids[nb];
        }
        return locate_bucket(px, py);
    }

    /**
     * Advective look-ahead. Every leaf whose indicator exceeds tol (1, the indicators are
     * normalised) is traced downstream for `horizon` seconds with its centroid velocity; each leaf
     * crossed by the path receives the indicator that feature would produce on it,
     * eta_K * (h_T / h_K)^q (eta ~ h^q), separately for the vorticity (q_v) and interpolation (q_i)
     * criteria. The path does not depend on the criterion, so it is traced once for both. Leaves
     * are located by walking along the path (locate_from). This keeps vortices inside the
     * refined region between adaptations instead of letting them run into coarse cells.
     */
    void apply_lookahead(std::vector<double>* av, std::vector<double>* ai, double horizon, double q_v, double q_i) {
        dbg_start();
        const int L = (int)leaf_ids.size();
        // Sources are the indicators before any update, per compact leaf
        std::vector<double> src_v(av ? L : 0), src_i(ai ? L : 0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < L; ++i) {
            if (av) src_v[i] = (*av)[leaf_ids[i]];
            if (ai) src_i[i] = (*ai)[leaf_ids[i]];
        }
        // Traces are independent. A leaf keeps the maximum of its own value and every trace that
        // crosses it; a maximum does not depend on the order, so the result is the same for any
        // thread count.
        auto update_max = [](double* slot, double val) {
            if (*slot >= val) return;   // cheap test first; the atomic below re-checks
            #pragma omp atomic compare
            if (*slot < val) { *slot = val; }
        };
        #pragma omp parallel for schedule(dynamic, 256)
        for (int i = 0; i < L; ++i) {
            const bool act_v = av && src_v[i] > 1.0;
            const bool act_i = ai && src_i[i] > 1.0;
            if (!act_v && !act_i) continue;
            int e = leaf_ids[i];
            const AmrElem& T = elems[e];
            double xc = (x[T.v[0]] + x[T.v[1]] + x[T.v[2]]) / 3.0;
            double yc = (y[T.v[0]] + y[T.v[1]] + y[T.v[2]]) / 3.0;
            double uc = (u[T.v[0]] + u[T.v[1]] + u[T.v[2]]) / 3.0;
            double vc = (v[T.v[0]] + v[T.v[1]] + v[T.v[2]]) / 3.0;
            double hK = std::sqrt(2.0 * leaf_area(e));
            double dist = std::hypot(uc, vc) * horizon;
            if (dist < 0.5 * hK) continue;
            int n = std::min(64, (int)std::ceil(dist / (0.5 * hK)));
            int last = e;
            for (int s = 1; s <= n; ++s) {
                double f = horizon * (double)s / n;
                double px = xc + uc * f, py = yc + vc * f;
                if (px < 0.0 || px > SimConfig::L_CHANNEL || py < 0.0 || py > SimConfig::H_CHANNEL) break;
                int t = locate_from(last, px, py);
                if (t < 0) break;
                if (t == last) continue;
                last = t;
                double hT = std::sqrt(2.0 * leaf_area(t));
                if (act_v) update_max(&(*av)[t], src_v[i] * std::pow(hT / hK, q_v));
                if (act_i) update_max(&(*ai)[t], src_i[i] * std::pow(hT / hK, q_i));
            }
        }
        dbg_lap(D_TRACE);
    }

    int alloc_vertex() {
        if (!free_verts.empty()) {
            int id = free_verts.back();
            free_verts.pop_back();
            return id;
        }
        x.push_back(0.0);  y.push_back(0.0);
        u.push_back(0.0);  v.push_back(0.0);  p.push_back(0.0);
        uo.push_back(0.0); vo.push_back(0.0);
        bc_tag.push_back(BC_INTERIOR);
        v_alive.push_back(0);
        v_parents.push_back({-1, -1});
        v_edge.push_back(0);
        return (int)x.size() - 1;
    }

    int alloc_elem() {
        if (!free_elems.empty()) {
            int id = free_elems.back();
            free_elems.pop_back();
            return id;
        }
        elems.emplace_back();
        return (int)elems.size() - 1;
    }

    void free_elem(int e) {
        elems[e] = AmrElem();
        free_elems.push_back(e);
    }

    /**
     * Midpoint of edge (a, b), created on first use. Cylinder edges get their midpoint
     * projected onto the exact circle; u, v, p are prolongated by P1 interpolation and
     * Dirichlet data are imposed exactly on boundary midpoints.
     */
    int get_or_create_midpoint(int a, int b) {
        uint64_t key = edge_key(a, b);
        auto it = edge_mid.find(key);
        if (it != edge_mid.end()) return it->second;

        double mx = 0.5 * (x[a] + x[b]);
        double my = 0.5 * (y[a] + y[b]);
        int mid_tag;
        if (bc_tag[a] == BC_CYLINDER && bc_tag[b] == BC_CYLINDER) {
            // Cylinder nodes are only ever connected by boundary edges, so this edge lies on the arc
            double theta = std::atan2(my - SimConfig::Y_CYL, mx - SimConfig::X_CYL);
            mx = SimConfig::X_CYL + SimConfig::R_CYL * std::cos(theta);
            my = SimConfig::Y_CYL + SimConfig::R_CYL * std::sin(theta);
            mid_tag = BC_CYLINDER;
        } else {
            // Straight channel boundaries are detected geometrically. A cylinder tag is
            // only valid via the branch above: deep in the boundary layer an interior
            // midpoint can come within the cylinder classification tolerance.
            mid_tag = classify_boundary_node(mx, my);
            if (mid_tag == BC_CYLINDER) mid_tag = BC_INTERIOR;
        }

        int m = alloc_vertex();
        x[m] = mx;  y[m] = my;  bc_tag[m] = mid_tag;
        set_midpoint_values(m, a, b);
        created_log.push_back({m, a, b});
        v_alive[m]   = 1;
        v_parents[m] = {-1, -1};
        v_edge[m]    = key;
        edge_mid[key] = m;
        return m;
    }

    // u, v, p (both time levels) of midpoint m of edge (a, b): P1 interpolation, Dirichlet data on
    // boundary midpoints
    void set_midpoint_values(int m, int a, int b) {
        u[m]  = 0.5 * (u[a] + u[b]);   v[m]  = 0.5 * (v[a] + v[b]);   p[m] = 0.5 * (p[a] + p[b]);
        uo[m] = 0.5 * (uo[a] + uo[b]); vo[m] = 0.5 * (vo[a] + vo[b]);
        if (bc_tag[m] == BC_CYLINDER || bc_tag[m] == BC_WALL) {
            u[m] = v[m] = uo[m] = vo[m] = 0.0;
        } else if (bc_tag[m] == BC_INLET) {
            u[m] = uo[m] = 4.0 * inlet_umax * y[m] * (SimConfig::H_CHANNEL - y[m])
                         / (SimConfig::H_CHANNEL * SimConfig::H_CHANNEL);
            v[m] = vo[m] = 0.0;
        }
    }

    /**
     * Newest-vertex bisection of leaf e = (c, a, b) through its refinement edge (a, b):
     *   children (m, c, a) and (m, b, c); m becomes the newest vertex of both, and the
     *   parent's two other edges become the children's refinement edges. Orientation is
     *   preserved.
     */
    void bisect(int e, std::vector<int>* new_children = nullptr,
                std::vector<std::pair<int, int>>* new_edges = nullptr) {
        int c = elems[e].v[0], a = elems[e].v[1], b = elems[e].v[2];
        int gen = elems[e].gen;
        if (new_edges && !edge_mid.count(edge_key(a, b))) new_edges->push_back({a, b});
        int m = get_or_create_midpoint(a, b);

        int c1 = alloc_elem();
        int c2 = alloc_elem();
        AmrElem& T1 = elems[c1];
        T1.v[0] = m; T1.v[1] = c; T1.v[2] = a;
        T1.gen = gen + 1;  T1.parent = e;  T1.alive = true;
        AmrElem& T2 = elems[c2];
        T2.v[0] = m; T2.v[1] = b; T2.v[2] = c;
        T2.gen = gen + 1;  T2.parent = e;  T2.alive = true;

        elems[e].child[0] = c1;
        elems[e].child[1] = c2;
        if (new_children) { new_children->push_back(c1); new_children->push_back(c2); }

        auto& par = v_parents[m];
        if (par[0] < 0)      par[0] = e;
        else if (par[1] < 0) par[1] = e;
        else {
            fprintf(stderr, "[AMR] internal error: edge bisected by more than two elements\n");
            exit(EXIT_FAILURE);
        }
    }

    // Vertex -> incident leaves of the current forest (call before refinement starts)
    void build_vertex_leaves() {
        vleaf_ptr.assign(x.size() + 1, 0);
        for (const AmrElem& T : elems) {
            if (T.is_leaf()) for (int k = 0; k < 3; ++k) vleaf_ptr[T.v[k] + 1]++;
        }
        for (size_t i = 0; i < x.size(); ++i) vleaf_ptr[i + 1] += vleaf_ptr[i];
        vleaf_list.resize(vleaf_ptr[x.size()]);
        std::vector<int> fill(vleaf_ptr.begin(), vleaf_ptr.end() - 1);
        for (int e = 0; e < (int)elems.size(); ++e) {
            if (elems[e].is_leaf()) for (int k = 0; k < 3; ++k) vleaf_list[fill[elems[e].v[k]]++] = e;
        }
    }

    bool has_vertex(int e, int a) const {
        const int* w = elems[e].v;
        return w[0] == a || w[1] == a || w[2] == a;
    }

    /**
     * Appends the current leaves that contain edge (a, b). Every leaf containing vertex a
     * descends from a leaf of build_vertex_leaves() (if a existed then) or from an element
     * bisected through a (a midpoint created since); the search follows children containing a.
     */
    void leaves_with_edge(int a, int b, std::vector<int>& out) {
        std::vector<int>& stack = edge_search_stack;
        stack.clear();
        if (a + 1 < (int)vleaf_ptr.size() && vleaf_ptr[a + 1] > vleaf_ptr[a]) {
            for (int s = vleaf_ptr[a]; s < vleaf_ptr[a + 1]; ++s) stack.push_back(vleaf_list[s]);
        } else {
            for (int pe : v_parents[a]) {
                if (pe >= 0) { stack.push_back(elems[pe].child[0]); stack.push_back(elems[pe].child[1]); }
            }
        }
        while (!stack.empty()) {
            int e = stack.back();
            stack.pop_back();
            if (!has_vertex(e, a)) continue;
            if (elems[e].is_leaf()) {
                if (has_vertex(e, b)) out.push_back(e);
            } else {
                stack.push_back(elems[e].child[0]);
                stack.push_back(elems[e].child[1]);
            }
        }
    }
    std::vector<int> edge_search_stack;

    bool is_hanging(int e) const {
        const AmrElem& T = elems[e];
        for (int k = 0; k < 3; ++k) {
            if (edge_mid.count(edge_key(T.v[k], T.v[(k + 1) % 3]))) return true;
        }
        return false;
    }

    /**
     * Conformity closure: bisect every leaf that has a hanging node on one of its edges
     * until none remain. Under NVB a hanging node on a non-refinement edge lands on a
     * child's refinement edge, so this terminates; with a matching base labelling only
     * coarser elements are ever bisected. Starting from a conforming mesh to which the
     * bisections that produced `children` and the new midpoints on `edges` were applied, the
     * only leaves that can hang are those children and the other leaves on those edges, so
     * each pass examines just the neighbourhood of the previous pass. Each pass bisects its
     * hanging leaves in ascending element id. Consumes `children` and `edges`.
     */
    int make_conforming(std::vector<int>& children, std::vector<std::pair<int, int>>& edges) {
        int count = 0;
        std::vector<int> hanging;
        while (true) {
            hanging.clear();
            for (int e : children) {
                if (elems[e].is_leaf()) hanging.push_back(e);
            }
            for (const auto& ab : edges) leaves_with_edge(ab.first, ab.second, hanging);
            std::sort(hanging.begin(), hanging.end());
            hanging.erase(std::unique(hanging.begin(), hanging.end()), hanging.end());
            hanging.erase(std::remove_if(hanging.begin(), hanging.end(),
                                         [&](int e) { return !is_hanging(e); }),
                          hanging.end());
            if (hanging.empty()) break;
            children.clear();
            edges.clear();
            for (int e : hanging) {
                bisect(e, &children, &edges);
                ++count;
            }
        }
        return count;
    }

    /**
     * Removes newest vertex m by merging the children of every element bisected through it
     * (two parents for an interior edge, one on the boundary). The merged parents inherit
     * the coarsening flag and the predicted indicator so a second sweep can continue upward.
     * Nodal injection is the exact P1 transfer to the nested coarse space.
     */
    void collapse_vertex(int m, std::vector<char>& coarsen,
                         std::vector<double>& av, std::vector<double>& ai) {
        for (int s = 0; s < 2; ++s) {
            int pe = v_parents[m][s];
            if (pe < 0) continue;
            int c0 = elems[pe].child[0], c1 = elems[pe].child[1];
            double merged_v = std::sqrt(2.0) * std::max(av[c0], av[c1]);
            double merged_i = 2.0 * std::max(ai[c0], ai[c1]);
            free_elem(c0);
            free_elem(c1);
            elems[pe].child[0] = elems[pe].child[1] = -1;
            coarsen[pe] = 1;
            av[pe] = merged_v;
            ai[pe] = merged_i;
        }
        edge_mid.erase(v_edge[m]);
        v_alive[m]   = 0;
        v_parents[m] = {-1, -1};
        free_verts.push_back(m);
    }
};
