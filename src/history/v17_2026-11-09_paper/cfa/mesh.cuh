// Compact solver mesh: base-mesh generator, CSR graph, validation
// Part of cylinder_fem_amr; included in this order by cylinder_fem_amr.cu.
#pragma once

// ============================================================================
// Unstructured Triangular Mesh Data Structure (compact, solver-facing)
// ============================================================================
struct Triangle {
    int v[3];     // Counter-clockwise vertex indices; v[0] = newest vertex, (v[1], v[2]) = refinement edge
    int level;    // AMR refinement level (0 = initial mesh); one level = h halved = 2 bisections
    int gen = 0;  // Number of bisections separating this element from its base-mesh ancestor
};

// Wall-clock milliseconds since t0
static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static inline uint64_t edge_key(int a, int b) {
    if (a > b) std::swap(a, b);
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
}

struct Mesh2D {
    int num_nodes = 0;
    int num_elems = 0;

    std::vector<double> x;          // Node X coordinates [num_nodes]
    std::vector<double> y;          // Node Y coordinates [num_nodes]
    std::vector<int>    bc_tag;     // Node boundary tag [num_nodes]
    std::vector<Triangle> elems;    // Triangular elements [num_elems]

    // CSR Sparse Matrix Pattern (Node-to-Node Adjacency)
    int nnz = 0;
    std::vector<int> csr_row_ptr;   // [num_nodes + 1]
    std::vector<int> csr_col_ind;   // [nnz]

    // Boundary edges on the cylinder (pressure probes on the cylinder surface)
    std::vector<std::pair<int, int>> cylinder_edges;
    std::vector<int> cylinder_edge_elem;   // the element owning each cylinder edge
};

// ============================================================================
// Initial Body-Fitted Unstructured Triangular Mesh Generator Around Cylinder
// ============================================================================
double signed_area(const Mesh2D& mesh, const Triangle& tri) {
    double x0 = mesh.x[tri.v[0]], y0 = mesh.y[tri.v[0]];
    double x1 = mesh.x[tri.v[1]], y1 = mesh.y[tri.v[1]];
    double x2 = mesh.x[tri.v[2]], y2 = mesh.y[tri.v[2]];
    return 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
}

// Swapping v[1] and v[2] keeps v[0] and the refinement edge (v[1], v[2]) intact.
void ensure_ccw(Mesh2D& mesh) {
    for (auto& tri : mesh.elems) {
        if (signed_area(mesh, tri) < 0.0) {
            std::swap(tri.v[1], tri.v[2]);
        }
    }
}

int classify_boundary_node(double px, double py) {
    const double eps = 1.0e-6;
    double dist_cyl = std::hypot(px - SimConfig::X_CYL, py - SimConfig::Y_CYL);
    if (std::abs(dist_cyl - SimConfig::R_CYL) < 1.0e-4) return BC_CYLINDER;
    if ( std::abs(py) < eps || std::abs(py - SimConfig::H_CHANNEL) < eps ) return BC_WALL;
    if ( std::abs(px) < eps ) return BC_INLET;
    if ( std::abs(px - SimConfig::L_CHANNEL) < eps ) return BC_OUTLET;
    return BC_INTERIOR;
}

/**
 * Generates a body-fitted multi-block O-grid + wake channel triangular mesh:
 *   - Block 1: Square box [0, 0.41] x [0, 0.41] containing the circular cylinder at (0.2, 0.2)
 *     discretized via transfinite radial interpolation between the circle (r = R_CYL) and the
 *     bounding square boundary.
 *   - Block 2: Downstream channel wake region [0.41, 2.20] x [0, 0.41] matching the eastern
 *     interface of Block 1 seamlessly.
 * Every quad is split along its diagonal and BOTH triangles get that diagonal as their NVB
 * refinement edge (newest vertex opposite to it). Refinement edges therefore pair up across
 * every quad, which is Stevenson's matching condition for newest-vertex bisection.
 */
void generate_cylinder_mesh(Mesh2D& mesh, int n_quarter = 8, int n_radial = 7, int n_wake_x = 32) {
    mesh.x.clear();
    mesh.y.clear();
    mesh.bc_tag.clear();
    mesh.elems.clear();

    const double L_box = SimConfig::H_CHANNEL; // 0.41 m square around cylinder
    const int N_azi = 4 * n_quarter;           // Perimeter nodes on cylinder & box

    // Deduplicated vertex insertion helper
    auto add_or_find_node = [&](double px, double py, int forced_tag = -1) -> int {
        const double tol = 1.0e-7;
        for (int i = 0; i < (int)mesh.x.size(); ++i) {
            if (std::hypot(mesh.x[i] - px, mesh.y[i] - py) < tol) {
                return i;
            }
        }
        int id = (int)mesh.x.size();
        mesh.x.push_back(px);
        mesh.y.push_back(py);
        int tag = (forced_tag >= 0) ? forced_tag : classify_boundary_node(px, py);
        mesh.bc_tag.push_back(tag);
        return id;
    };

    // Quad (v00, v10, v11, v01) split along diagonal v00-v11, newest vertices opposite to it
    auto add_quad = [&](int v00, int v10, int v11, int v01) {
        mesh.elems.push_back({{v10, v11, v00}, 0, 0});
        mesh.elems.push_back({{v01, v00, v11}, 0, 0});
    };

    // 1. Construct perimeter points on the [0, L_box] x [0, L_box] square (CCW starting from East-South corner)
    //    Side 0 (East):   x = L_box, y in [0, L_box]
    //    Side 1 (North):  y = L_box, x in [L_box, 0]
    //    Side 2 (West):   x = 0,     y in [L_box, 0]
    //    Side 3 (South):  y = 0,     x in [0, L_box]
    std::vector<double> box_x(N_azi), box_y(N_azi);
    std::vector<double> cyl_x(N_azi), cyl_y(N_azi);

    for (int q = 0; q < n_quarter; ++q) {
        double s = (double)q / n_quarter;
        // East edge: (L_box, 0) -> (L_box, L_box)
        box_x[0 * n_quarter + q] = L_box;
        box_y[0 * n_quarter + q] = s * L_box;
        // North edge: (L_box, L_box) -> (0, L_box)
        box_x[1 * n_quarter + q] = (1.0 - s) * L_box;
        box_y[1 * n_quarter + q] = L_box;
        // West edge: (0, L_box) -> (0, 0)
        box_x[2 * n_quarter + q] = 0.0;
        box_y[2 * n_quarter + q] = (1.0 - s) * L_box;
        // South edge: (0, 0) -> (L_box, 0)
        box_x[3 * n_quarter + q] = s * L_box;
        box_y[3 * n_quarter + q] = 0.0;
    }

    for (int i = 0; i < N_azi; ++i) {
        // Angle matching the ray from cylinder center to the outer box node
        double dx = box_x[i] - SimConfig::X_CYL;
        double dy = box_y[i] - SimConfig::Y_CYL;
        double theta = std::atan2(dy, dx);
        cyl_x[i] = SimConfig::X_CYL + SimConfig::R_CYL * std::cos(theta);
        cyl_y[i] = SimConfig::Y_CYL + SimConfig::R_CYL * std::sin(theta);
    }

    // Build O-grid nodes [n_radial + 1][N_azi]
    std::vector<std::vector<int>> ogrid_ids(n_radial + 1, std::vector<int>(N_azi));
    for (int r = 0; r <= n_radial; ++r) {
        // Graded radial stretching toward the cylinder boundary layer
        double xi = (double)r / n_radial;
        double weight = 0.65 * xi + 0.35 * xi * xi;
        for (int i = 0; i < N_azi; ++i) {
            double px = (1.0 - weight) * cyl_x[i] + weight * box_x[i];
            double py = (1.0 - weight) * cyl_y[i] + weight * box_y[i];
            int forced = (r == 0) ? BC_CYLINDER : -1;
            ogrid_ids[r][i] = add_or_find_node(px, py, forced);
        }
    }

    // Triangulate O-grid block
    for (int r = 0; r < n_radial; ++r) {
        for (int i = 0; i < N_azi; ++i) {
            int inext = (i + 1) % N_azi;
            add_quad(ogrid_ids[r][i], ogrid_ids[r + 1][i], ogrid_ids[r + 1][inext], ogrid_ids[r][inext]);
        }
    }

    // 2. Downstream wake block [L_box, L_CHANNEL] x [0, H_CHANNEL]
    //    Matches the n_quarter + 1 vertical nodes on the East edge of the O-grid box
    int ny_wake = n_quarter;
    std::vector<std::vector<int>> wake_ids(n_wake_x + 1, std::vector<int>(ny_wake + 1));

    for (int ix = 0; ix <= n_wake_x; ++ix) {
        double s = (double)ix / n_wake_x;
        // Slight stretching downstream so wake near cylinder has finer resolution
        double x_coord = L_box + (SimConfig::L_CHANNEL - L_box) * (0.75 * s + 0.25 * s * s);
        for (int iy = 0; iy <= ny_wake; ++iy) {
            double y_coord = ((double)iy / ny_wake) * SimConfig::H_CHANNEL;
            wake_ids[ix][iy] = add_or_find_node(x_coord, y_coord);
        }
    }

    for (int ix = 0; ix < n_wake_x; ++ix) {
        for (int iy = 0; iy < ny_wake; ++iy) {
            add_quad(wake_ids[ix][iy], wake_ids[ix + 1][iy], wake_ids[ix + 1][iy + 1], wake_ids[ix][iy + 1]);
        }
    }

    mesh.num_nodes = (int)mesh.x.size();
    mesh.num_elems = (int)mesh.elems.size();
    ensure_ccw(mesh);
}

// ============================================================================
// CSR Sparse Matrix Graph & Cylinder Boundary Edge Extraction
// ============================================================================
void build_csr_and_boundary_topology(Mesh2D& mesh) {
    mesh.num_nodes = (int)mesh.x.size();
    mesh.num_elems = (int)mesh.elems.size();
    const int n = mesh.num_nodes;

    // Node -> incident elements (the order inside a node's list does not matter: rows are sorted)
    std::vector<int> inc_ptr(n + 1, 0), inc(3 * (size_t)mesh.num_elems);
    const int ne = mesh.num_elems;
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < ne; ++e)
        for (int k = 0; k < 3; ++k) {
            #pragma omp atomic
            inc_ptr[mesh.elems[e].v[k] + 1]++;
        }
    for (int i = 0; i < n; ++i) inc_ptr[i + 1] += inc_ptr[i];
    {
        std::vector<int> fill(inc_ptr.begin(), inc_ptr.end() - 1);
        #pragma omp parallel for schedule(static)
        for (int e = 0; e < ne; ++e)
            for (int k = 0; k < 3; ++k) {
                int slot;
                #pragma omp atomic capture
                slot = fill[mesh.elems[e].v[k]]++;
                inc[slot] = e;
            }
    }

    // Row i: i and the vertices of its incident elements, sorted and unique. Each row is built in
    // a slot sized for its worst case (1 + 2 per incident element), then the rows are packed.
    std::vector<int> cap_ptr(n + 1, 0), len(n);
    for (int i = 0; i < n; ++i) cap_ptr[i + 1] = cap_ptr[i] + 1 + 2 * (inc_ptr[i + 1] - inc_ptr[i]);
    std::vector<int> tmp(cap_ptr[n]);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        int* row = tmp.data() + cap_ptr[i];
        int m = 0;
        row[m++] = i;
        for (int s = inc_ptr[i]; s < inc_ptr[i + 1]; ++s) {
            const Triangle& tri = mesh.elems[inc[s]];
            for (int k = 0; k < 3; ++k) if (tri.v[k] != i) row[m++] = tri.v[k];
        }
        std::sort(row, row + m);
        len[i] = (int)(std::unique(row, row + m) - row);
    }
    mesh.csr_row_ptr.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) mesh.csr_row_ptr[i + 1] = mesh.csr_row_ptr[i] + len[i];
    mesh.csr_col_ind.resize(mesh.csr_row_ptr[n]);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        std::copy(tmp.begin() + cap_ptr[i], tmp.begin() + cap_ptr[i] + len[i],
                  mesh.csr_col_ind.begin() + mesh.csr_row_ptr[i]);
    mesh.nnz = (int)mesh.csr_col_ind.size();

    // True boundary edges on the cylinder belong to exactly 1 triangle
    std::unordered_map<uint64_t, int> cyl_counts, cyl_elem;
    std::vector<uint64_t> cyl_order;
    for (int e = 0; e < mesh.num_elems; ++e) {
        const Triangle& tri = mesh.elems[e];
        for (int a = 0; a < 3; ++a) {
            int u = tri.v[a], w = tri.v[(a + 1) % 3];
            if (mesh.bc_tag[u] != BC_CYLINDER || mesh.bc_tag[w] != BC_CYLINDER) continue;
            if (cyl_counts[edge_key(u, w)]++ == 0) cyl_order.push_back(edge_key(u, w));
            cyl_elem[edge_key(u, w)] = e;
        }
    }
    mesh.cylinder_edges.clear();
    mesh.cylinder_edge_elem.clear();
    for (uint64_t key : cyl_order) {
        if (cyl_counts[key] != 1) continue;
        mesh.cylinder_edges.push_back({(int)(key >> 32), (int)(key & 0xffffffffu)});
        mesh.cylinder_edge_elem.push_back(cyl_elem[key]);
    }
}

// ============================================================================
// Mesh Integrity Validation (run after every adaptation)
// ============================================================================
struct MeshQuality {
    double min_angle_deg = 180.0;
    double min_area      = 1.0e300;
    double total_area    = 0.0;
    int    max_level     = 0;
    int    max_gen       = 0;
};

static bool on_same_physical_boundary(const Mesh2D& mesh, int a, int b) {
    const double eps = 1.0e-9;
    int ta = mesh.bc_tag[a], tb = mesh.bc_tag[b];
    if (ta == BC_INTERIOR || tb == BC_INTERIOR) return false;
    if (ta == BC_CYLINDER || tb == BC_CYLINDER) return ta == BC_CYLINDER && tb == BC_CYLINDER;
    double xa = mesh.x[a], ya = mesh.y[a], xb = mesh.x[b], yb = mesh.y[b];
    if (std::abs(ya) < eps && std::abs(yb) < eps) return true;
    if (std::abs(ya - SimConfig::H_CHANNEL) < eps && std::abs(yb - SimConfig::H_CHANNEL) < eps) return true;
    if (std::abs(xa) < eps && std::abs(xb) < eps) return true;
    if (std::abs(xa - SimConfig::L_CHANNEL) < eps && std::abs(xb - SimConfig::L_CHANNEL) < eps) return true;
    return false;
}

/**
 * Verifies that the mesh is a valid conforming triangulation: every element is positively
 * oriented, every edge is shared by at most two elements, and every edge with a single
 * element lies on the physical boundary (a free interior edge means a hanging node).
 * Edges are counted on the CSR graph, so build_csr_and_boundary_topology must run first.
 */
bool validate_mesh(const Mesh2D& mesh, MeshQuality& q) {
    q = MeshQuality();
    bool ok = true;
    if ((int)mesh.csr_row_ptr.size() != mesh.num_nodes + 1) {
        fprintf(stderr, "[Mesh] CSR graph does not match the mesh\n");
        return false;
    }
    // Edge (a < b) multiplicities, stored at the CSR position of b in row a
    std::vector<int> edge_count(mesh.nnz, 0);
    double max_cos = -1.0, min_area = 1.0e300, total_area = 0.0;
    int max_level = 0, max_gen = 0;

    #pragma omp parallel for schedule(static) reduction(max : max_cos, max_level, max_gen) \
                                              reduction(min : min_area) reduction(+ : total_area)
    for (int e = 0; e < mesh.num_elems; ++e) {
        const Triangle& tri = mesh.elems[e];
        double area = signed_area(mesh, tri);
        if (!(area > 0.0)) {
            #pragma omp critical(mesh_report)
            {
                fprintf(stderr, "[Mesh] element %d has non-positive area %.3e\n", e, area);
                ok = false;
            }
        }
        min_area = std::min(min_area, area);
        total_area += area;
        max_level = std::max(max_level, tri.level);
        max_gen   = std::max(max_gen, tri.gen);

        for (int k = 0; k < 3; ++k) {
            int a = tri.v[k], b = tri.v[(k + 1) % 3], c = tri.v[(k + 2) % 3];
            int lo = std::min(a, b), hi = std::max(a, b);
            const int* first = mesh.csr_col_ind.data() + mesh.csr_row_ptr[lo];
            const int* last  = mesh.csr_col_ind.data() + mesh.csr_row_ptr[lo + 1];
            const int* pos   = std::lower_bound(first, last, hi);
            if (pos == last || *pos != hi) {
                #pragma omp critical(mesh_report)
                {
                    fprintf(stderr, "[Mesh] edge (%d, %d) missing from the CSR graph\n", lo, hi);
                    ok = false;
                }
            } else {
                #pragma omp atomic
                edge_count[pos - mesh.csr_col_ind.data()]++;
            }
            double ux = mesh.x[b] - mesh.x[a], uy = mesh.y[b] - mesh.y[a];
            double wx = mesh.x[c] - mesh.x[a], wy = mesh.y[c] - mesh.y[a];
            double cosang = (ux * wx + uy * wy) /
                            (std::sqrt((ux * ux + uy * uy) * (wx * wx + wy * wy)) + 1.0e-300);
            max_cos = std::max(max_cos, cosang);
        }
    }
    q.min_area = min_area;  q.total_area = total_area;  q.max_level = max_level;  q.max_gen = max_gen;
    q.min_angle_deg = std::acos(std::max(-1.0, std::min(1.0, max_cos))) * 180.0 / SimConfig::PI;

    int bad_edges = 0;
    for (int a = 0; a < mesh.num_nodes; ++a) {
        for (int idx = mesh.csr_row_ptr[a]; idx < mesh.csr_row_ptr[a + 1]; ++idx) {
            int b = mesh.csr_col_ind[idx], count = edge_count[idx];
            if (b <= a || count == 0) continue;
            if (count > 2 || (count == 1 && !on_same_physical_boundary(mesh, a, b))) {
                if (bad_edges < 10) {
                    fprintf(stderr, "[Mesh] non-conforming edge (%d, %d) shared by %d element(s)\n",
                            a, b, count);
                }
                ++bad_edges;
            }
        }
    }
    if (bad_edges > 0) {
        fprintf(stderr, "[Mesh] %d non-conforming edges\n", bad_edges);
        ok = false;
    }
    return ok;
}
