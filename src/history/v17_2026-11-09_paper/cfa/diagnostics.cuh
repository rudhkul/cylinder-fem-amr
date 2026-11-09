// Pressure probes, force-history statistics and VTK output
// Part of cylinder_fem_amr; included in this order by cylinder_fem_amr.cu.
#pragma once

// ============================================================================
// Hydrodynamic Diagnostics: Surface Pressure Probes & Force History Statistics
// ============================================================================
struct PressureProbe {
    int a = 0, b = 0;
    double w = 0.0;   // p = (1 - w) p_a + w p_b
};

static double wrap_angle(double a) {
    while (a <= -SimConfig::PI) a += 2.0 * SimConfig::PI;
    while (a >   SimConfig::PI) a -= 2.0 * SimConfig::PI;
    return a;
}

/**
 * Locates the cylinder boundary edge spanning polar angle theta and returns the linear
 * interpolation weights of the surface point at that angle.
 */
PressureProbe find_cylinder_probe(const Mesh2D& mesh, double theta) {
    PressureProbe best;
    double best_dist = 1.0e300;
    for (const auto& edge : mesh.cylinder_edges) {
        int a = edge.first, b = edge.second;
        double ta = std::atan2(mesh.y[a] - SimConfig::Y_CYL, mesh.x[a] - SimConfig::X_CYL);
        double tb = std::atan2(mesh.y[b] - SimConfig::Y_CYL, mesh.x[b] - SimConfig::X_CYL);
        double span = wrap_angle(tb - ta);
        double s    = wrap_angle(theta - ta);
        if (std::abs(span) < 1.0e-14) continue;
        double w = s / span;
        if (w >= -1.0e-12 && w <= 1.0 + 1.0e-12) {
            best = {a, b, std::min(1.0, std::max(0.0, w))};
            return best;
        }
        double dist = std::min(std::abs(s), std::abs(wrap_angle(theta - tb)));
        if (dist < best_dist) {
            best_dist = dist;
            best = {a, b, (std::abs(s) < std::abs(wrap_angle(theta - tb))) ? 0.0 : 1.0};
        }
    }
    return best;
}

struct ForceHistory {
    std::vector<double> t, cd, cl, dp;

    void add(double ti, double cdi, double cli, double dpi) {
        t.push_back(ti);  cd.push_back(cdi);  cl.push_back(cli);  dp.push_back(dpi);
    }

    // Parabola through samples i-1, i, i+1: (time, value) of the extremum
    static std::pair<double, double> refine_peak(const std::vector<double>& tt, const std::vector<double>& y, size_t i) {
        double a = y[i - 1], b = y[i], c = y[i + 1];
        double denom = a - 2.0 * b + c;
        double sft = (denom != 0.0) ? 0.5 * (a - c) / denom : 0.0;
        sft = std::max(-0.5, std::min(0.5, sft));
        return { tt[i] + sft * (tt[i + 1] - tt[i]), b - 0.25 * (a - c) * sft };
    }

    static double interp(const std::vector<double>& tt, const std::vector<double>& y, double tq) {
        auto it = std::lower_bound(tt.begin(), tt.end(), tq);
        if (it == tt.begin()) return y.front();
        if (it == tt.end()) return y.back();
        size_t k = it - tt.begin();
        double w = (tq - tt[k - 1]) / (tt[k] - tt[k - 1]);
        return (1.0 - w) * y[k - 1] + w * y[k];
    }

    /**
     * DFG 2D-2 quantities over the last `window` seconds after the start-up transient, as
     * defined by Schaefer & Turek (1996): each lift period [t0, t0 + 1/f] starts at a maximum
     * of C_L; per period C_D,max, C_D,min, C_L,max, C_L,min, St = D f / U_mean and
     * delta_p(t0 + 1/(2f)). Extrema are refined with a parabola through three samples. The
     * mean over the complete periods is printed with the spread (max - min) across periods.
     * Periods are delimited by the upward crossings of the mean lift, so isolated glitches
     * cannot split a period. Also written to <prefix>_stats.csv by the caller.
     */
    struct Stats { int periods = 0; double v[6] = {0}, spread[6] = {0}; };

    Stats compute(double window, size_t& i0) const {
        Stats S;
        const double t_startup = 2.0;
        i0 = 0;
        if (t.empty() || t.back() < t_startup + 0.5) return S;
        while (i0 < t.size() && t[i0] < std::max(t.back() - window, t_startup)) ++i0;
        if (t.size() - i0 < 5) return S;

        double cl_mean = 0.0;
        for (size_t i = i0; i < t.size(); ++i) cl_mean += cl[i];
        cl_mean /= (double)(t.size() - i0);
        std::vector<size_t> up;   // first sample after each upward crossing of the mean lift
        for (size_t i = i0 + 1; i < t.size(); ++i)
            if (cl[i - 1] < cl_mean && cl[i] >= cl_mean) up.push_back(i);
        // lift maximum between consecutive upward crossings
        std::vector<std::pair<double, double>> lmax;
        for (size_t k = 0; k + 1 < up.size(); ++k) {
            size_t best = up[k];
            for (size_t i = up[k]; i < up[k + 1]; ++i) if (cl[i] > cl[best]) best = i;
            if (best > 0 && best + 1 < t.size()) lmax.push_back(refine_peak(t, cl, best));
        }
        std::vector<double> per[6];
        for (size_t k = 0; k + 1 < lmax.size(); ++k) {
            double t0 = lmax[k].first, t1 = lmax[k + 1].first, f = 1.0 / (t1 - t0);
            double dmax = -1e300, dmin = 1e300, lmin = 1e300;
            for (size_t i = 1; i + 1 < t.size(); ++i) {
                if (t[i] < t0 || t[i] >= t1) continue;
                if (cd[i] >= cd[i - 1] && cd[i] > cd[i + 1]) dmax = std::max(dmax, refine_peak(t, cd, i).second);
                if (cd[i] <= cd[i - 1] && cd[i] < cd[i + 1]) dmin = std::min(dmin, refine_peak(t, cd, i).second);
                if (cl[i] <= cl[i - 1] && cl[i] < cl[i + 1]) lmin = std::min(lmin, refine_peak(t, cl, i).second);
            }
            if (dmax < -1e299 || dmin > 1e299 || lmin > 1e299) continue;
            per[0].push_back(dmax);
            per[1].push_back(dmin);
            per[2].push_back(lmax[k].second);
            per[3].push_back(lmin);
            per[4].push_back(2.0 * SimConfig::R_CYL * f / SimConfig::U_MEAN);
            per[5].push_back(interp(t, dp, t0 + 0.5 / f));
        }
        S.periods = (int)per[0].size();
        for (int q = 0; q < 6 && S.periods > 0; ++q) {
            double sum = 0.0, lo = 1e300, hi = -1e300;
            for (double x : per[q]) { sum += x; lo = std::min(lo, x); hi = std::max(hi, x); }
            S.v[q] = sum / S.periods;
            S.spread[q] = hi - lo;
        }
        return S;
    }

    /**
     * DFG 2D-3 quantities (Schaefer & Turek 1996; reference values John 2004, IJNMF 44:777):
     * C_D,max and its time, C_L,max and its time (parabolic refinement of the discrete
     * maxima) and delta_p at t = 8 s.
     */
    void report_2d3(const std::string& csv_path) const {
        std::cout << "------------------------------------------------------------------------\n";
        if (t.size() < 3 || t.back() < 8.0 - 1.0e-9) {
            std::cout << "  DFG 2D-3 statistics need t_end = 8 s\n";
            std::cout << "------------------------------------------------------------------------\n";
            return;
        }
        // skip the first steps (impulsive start transient of the pressure at t ~ 0)
        size_t i0 = 1;
        while (i0 < t.size() && t[i0] < 1.0) ++i0;
        size_t id = i0, il = i0;
        for (size_t i = i0; i + 1 < t.size(); ++i) {
            if (cd[i] > cd[id]) id = i;
            if (cl[i] > cl[il]) il = i;
        }
        auto pd = refine_peak(t, cd, id), pl = refine_peak(t, cl, il);
        double dp8 = interp(t, dp, 8.0);
        static const double ref[5] = {3.93625, 2.950921575, 5.693125, 0.47795, -0.1116};
        // delta_p = p(a1) - p(a2) as in the DFG definition (FEATFLOW level 6: -0.1115 at t = 8 s)
        double val[5] = {pd.first, pd.second, pl.first, pl.second, dp8};
        static const char* names[5] = {"t(C_D,max)", "C_D,max", "t(C_L,max)", "C_L,max", "dp(8 s)"};
        std::cout << std::fixed << std::setprecision(6) << "  DFG 2D-3 (reference: John 2004)\n";
        for (int q = 0; q < 5; ++q) {
            std::cout << "    " << std::left << std::setw(11) << names[q] << std::right << std::setw(12) << val[q]
                      << "   ref " << std::setw(12) << ref[q] << "   rel. error " << std::showpos
                      << std::setprecision(3) << 100.0 * (val[q] - ref[q]) / std::fabs(ref[q]) << " %"
                      << std::noshowpos << std::setprecision(6) << "\n";
        }
        std::cout << "------------------------------------------------------------------------\n";
        std::ofstream f(csv_path);
        if (f) {
            f << "quantity,value,reference\n" << std::setprecision(10);
            static const char* keys[5] = {"t_cd_max", "cd_max", "t_cl_max", "cl_max", "dp_8"};
            for (int q = 0; q < 5; ++q) f << keys[q] << "," << val[q] << "," << ref[q] << "\n";
        }
    }

    void report(double window, const std::string& csv_path) const {
        std::cout << "------------------------------------------------------------------------\n";
        size_t i0 = 0;
        Stats S = compute(window, i0);
        if (S.periods == 0) {
            std::cout << "  Force statistics: n/a (statistics start after the 2 s start-up transient and"
                         " need two complete lift periods; run to t >= 6 s for DFG 2D-2 values)\n";
            std::cout << "------------------------------------------------------------------------\n";
            return;
        }
        // Reference: FEATFLOW Q2/P1disc level 6, dt = 1/1600 (2014 data, our evaluation with
        // the same definitions, scripts/benchmark_stats.py); DFG 1996 ranges in brackets
        static const char* names[6] = {"C_D,max", "C_D,min", "C_L,max", "C_L,min", "St", "dp(t0+1/2f)"};
        static const double ref[6] = {3.22739, 3.16426, 0.98658, -1.02129, 0.30184, 2.48520};
        static const char* range[6] = {"[3.22, 3.24]", "", "[0.99, 1.01]", "", "[0.295, 0.305]", "[2.46, 2.50]"};
        std::cout << std::fixed << std::setprecision(5)
                  << "  Force statistics over t in [" << t[i0] << ", " << t.back() << "] s, "
                  << S.periods << " lift period(s)\n"
                  << "                 value     spread     reference   rel. error   DFG 1996\n";
        for (int q = 0; q < 6; ++q) {
            std::cout << "    " << std::left << std::setw(12) << names[q] << std::right
                      << std::setw(9) << S.v[q] << "  " << std::scientific << std::setprecision(1)
                      << std::setw(8) << S.spread[q] << "  " << std::fixed << std::setprecision(5)
                      << std::setw(10) << ref[q] << "  " << std::showpos << std::setprecision(3)
                      << std::setw(8) << 100.0 * (S.v[q] - ref[q]) / std::fabs(ref[q]) << " %"
                      << std::noshowpos << std::setprecision(5) << "   " << range[q] << "\n";
        }
        std::cout << "------------------------------------------------------------------------\n";
        std::ofstream f(csv_path);
        if (f) {
            f << "quantity,value,spread,reference\n" << std::setprecision(10);
            static const char* keys[6] = {"cd_max", "cd_min", "cl_max", "cl_min", "St", "dp_half"};
            for (int q = 0; q < 6; ++q) f << keys[q] << "," << S.v[q] << "," << S.spread[q] << "," << ref[q] << "\n";
            f << "periods," << S.periods << ",0,0\n";
        }
    }
};

// ============================================================================
// ParaView Legacy VTK UnstructuredGrid Exporter (.vtk)
// ============================================================================
void write_vtk_unstructured(
    const std::string& filename,
    const Mesh2D& mesh,
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& p,
    const std::vector<double>& vort,
    const std::vector<double>& elem_err,   // [vorticity | interpolation] indicators, 2 * num_elems
    const AmrParams& amr
) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        fprintf(stderr, "[Output] cannot open %s for writing\n", filename.c_str());
        return;
    }
    ofs << std::setprecision(10);

    ofs << "# vtk DataFile Version 3.0\n";
    ofs << "2D FEM-AMR Cylinder Flow Solution (CUDA)\n";
    ofs << "ASCII\n";
    ofs << "DATASET UNSTRUCTURED_GRID\n";
    ofs << "POINTS " << mesh.num_nodes << " double\n";
    for (int i = 0; i < mesh.num_nodes; ++i) {
        ofs << mesh.x[i] << " " << mesh.y[i] << " 0.0\n";
    }

    ofs << "\nCELLS " << mesh.num_elems << " " << (4 * mesh.num_elems) << "\n";
    for (const auto& tri : mesh.elems) {
        ofs << "3 " << tri.v[0] << " " << tri.v[1] << " " << tri.v[2] << "\n";
    }

    ofs << "\nCELL_TYPES " << mesh.num_elems << "\n";
    for (int e = 0; e < mesh.num_elems; ++e) {
        ofs << "5\n"; // VTK_TRIANGLE
    }

    ofs << "\nPOINT_DATA " << mesh.num_nodes << "\n";
    ofs << "VECTORS velocity double\n";
    for (int i = 0; i < mesh.num_nodes; ++i) {
        ofs << u[i] << " " << v[i] << " 0.0\n";
    }

    ofs << "\nSCALARS pressure double 1\nLOOKUP_TABLE default\n";
    for (int i = 0; i < mesh.num_nodes; ++i) {
        ofs << p[i] << "\n";
    }

    ofs << "\nSCALARS vorticity double 1\nLOOKUP_TABLE default\n";
    for (int i = 0; i < mesh.num_nodes; ++i) {
        ofs << vort[i] << "\n";
    }

    ofs << "\nCELL_DATA " << mesh.num_elems << "\n";
    ofs << "SCALARS amr_level int 1\nLOOKUP_TABLE default\n";
    for (const auto& tri : mesh.elems) {
        ofs << tri.level << "\n";
    }

    ofs << "\nSCALARS nvb_generation int 1\nLOOKUP_TABLE default\n";
    for (const auto& tri : mesh.elems) {
        ofs << tri.gen << "\n";
    }

    const int ne = mesh.num_elems;
    const bool have = (int)elem_err.size() >= 2 * ne;
    ofs << "\nSCALARS indicator_vorticity double 1\nLOOKUP_TABLE default\n";
    for (int e = 0; e < ne; ++e) ofs << (have ? elem_err[e] : 0.0) << "\n";
    ofs << "\nSCALARS indicator_interp double 1\nLOOKUP_TABLE default\n";
    for (int e = 0; e < ne; ++e) ofs << (have ? elem_err[ne + e] : 0.0) << "\n";
    // Combined score max(eta_v / tol_v, eta_i / tol_i): > 1 refines, < coarsen_safety coarsens
    ofs << "\nSCALARS error_indicator double 1\nLOOKUP_TABLE default\n";
    for (int e = 0; e < ne; ++e) {
        double sv = (have && amr.tol_vort > 0.0)   ? elem_err[e] / amr.tol_vort        : 0.0;
        double si = (have && amr.tol_interp > 0.0) ? elem_err[ne + e] / amr.tol_interp : 0.0;
        ofs << std::max(sv, si) << "\n";
    }
}
