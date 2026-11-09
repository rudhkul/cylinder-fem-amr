// Command-line options
// Part of cylinder_fem_amr; included in this order by cylinder_fem_amr.cu.
#pragma once

// ============================================================================
// Command Line Options
// ============================================================================
struct RunOptions {
    int    num_steps      = 100;
    double dt             = SimConfig::DT_DEFAULT;
    int    amr_interval   = 10;
    int    init_amr       = 2;
    int    print_interval = 10;
    int    vtk_interval   = 0;
    double stats_window   = 2.0;
    bool   rotational     = true;
    double lookahead      = 1.0;     // look-ahead horizon in units of amr_interval * dt
    bool   bdf2           = true;    // BDF2 (second order) or backward Euler time stepping
    double supg_tau_time  = 1.0e-3;  // time scale T in the SUPG tau [s]; 0 = use dt. A fixed T keeps
                                     // the spatial stabilization independent of dt, which restores
                                     // second-order temporal convergence of BDF2 (with T = dt the
                                     // shrinking tau adds an O(dt) spatial perturbation)
    bool   amg            = true;    // SA-AMG (true) or Jacobi (false) preconditioner for the PPE
    bool   momentum_ilu   = true;    // multicolour ILU(0) (true) or Jacobi (false) for the momentum BiCGSTAB
    bool   skew_convection = true;   // skew-symmetric (true) or convective (false) Galerkin convection
    int    amg_rebuild    = 50;      // rebuild the AMG hierarchy at least every N steps
    int    amr_overlap    = 5;       // > 0: build the new mesh in a worker thread while the GPU takes
                                     // this many more steps on the old one (see Step F); 0 = off
                                     // (default since 2026-11-08, HANDOVER 11.15/11.16)
    int    amg_remap      = 0;       // > 0: up to N remeshes in a row carry the AMG hierarchy over
                                     // (AmgPreconditioner::remap) before a full rebuild; 0 = rebuild
    int    uniform_level  = 0;       // > 0: uniformly refined fixed mesh, AMR off
    bool   mms            = false;   // manufactured-solution verification run (fixed mesh)
    bool   case_2d3       = false;   // DFG 2D-3: inflow U_MAX sin(pi t / 8), fluid at rest at t = 0
    bool   relative_tol   = true;    // 2D-3: AMR tolerances scale with the current inflow
    bool   rotational_from_dp = true; // rotational term from the pressure increment (see main)
    int    remesh_projection = 4;    // after a remesh re-project: 0 nothing, 1 u^n, 2 u^n and u^{n-1}
                                     // to zero divergence, 4 both with the scheme's own pressure step
    double pressure_stab  = 0.3;     // OSS pressure stabilization factor (0 = off), times Codina's
                                     // tau. 0.03 left a node-to-node pressure mode along the wall
                                     // (U3: dp error 3.9 %, 0.1: 1.4 %, 0.3: 0.4 %) with forces and
                                     // shedding unchanged; 1.0 diverges
    int    dump_systems   = 0;       // > 0: write the momentum and pressure systems of this step (solver studies)
    std::string prefix    = "cylinder_fem_amr";
    AmrParams amr;
};

void print_usage(const char* prog) {
    RunOptions d;
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "       " << prog << " [num_steps] [amr_interval] [max_amr_level] [amr_tol_vort]   (legacy)\n\n"
        << "  --steps N           number of time steps                          (" << d.num_steps << ")\n"
        << "  --dt S              time step [s]                                 (" << d.dt << ")\n"
        << "  --amr-interval N    adapt every N steps, 0 disables AMR           (" << d.amr_interval << ")\n"
        << "  --init-amr N        adaptation cycles on the initial condition    (" << d.init_amr << ")\n"
        << "  --max-level L       max AMR levels above the base mesh            (" << d.amr.max_level << ")\n"
        << "  --amr-tol-vort U    disturbance-vorticity criterion tolerance [m/s], 0 = off   (" << d.amr.tol_vort << ")\n"
        << "  --amr-tol-interp U  disturbance interpolation-error tolerance [m/s], 0 = off (" << d.amr.tol_interp << ")\n"
        << "  --coarsen-safety F  coarsen where merged score < F (refine at 1)  (" << d.amr.coarsen_safety << ")\n"
        << "  --lookahead F       advective look-ahead, in AMR intervals, 0=off (" << d.lookahead << ")\n"
        << "  --buffer N          refinement buffer layers                      (" << d.amr.buffer_layers << ")\n"
        << "  --max-elems N       element budget                                (" << d.amr.max_elems << ")\n"
        << "  --rotational-div increment|velocity     div u* in the rotational term from the pressure\n"
        << "                      increment (increment) or from u* directly (velocity, pre-2026-10)\n"
        << "  --case 2d2|2d3      DFG benchmark: periodic Re = 100 (2d2), or inflow 1.5 sin(pi t/8)\n"
        << "                      from rest over 0 <= t <= 8 s (2d3; use --steps 8000)\n"
        << "  --amr-tol-scale inflow|fixed            2d3: tolerances follow the inflow magnitude (inflow)\n"
        << "  --mms               manufactured-solution verification on the fixed (uniform) mesh;\n"
        << "                      prints the L2/H1 errors at the final time (<prefix>_mms.csv)\n"
        << "  --uniform-level L   uniformly refine the base mesh L levels and disable AMR (0)\n"
        << "  --min-level L       AMR never goes below level L (start from a uniform level-L mesh) (" << d.amr.min_level << ")\n"
        << "  --keep-radius R     never coarsen within R [m] of the cylinder centre, 0 = off (" << d.amr.keep_radius << ")\n"
        << "  --amr-overlap K     build each new mesh on the host while the GPU takes K more steps on\n"
        << "                      the old one, then transfer the state of that step; K < interval, 0 = off (" << d.amr_overlap << ")\n"
        << "  --amr-tol-radius R  tolerances grow as (d/R)^P beyond distance d > R [m] from the\n"
        << "                      cylinder centre, 0 = uniform tolerances (" << d.amr.tol_radius << ")\n"
        << "  --amr-tol-power P   exponent P of that growth (" << d.amr.tol_power << ")\n"
        << "  --print-interval N  console diagnostics every N steps             (" << d.print_interval << ")\n"
        << "  --vtk-interval N    write a VTK snapshot every N steps, 0 = final (" << d.vtk_interval << ")\n"
        << "  --pressure-update rotational|standard   incremental projection variant (rotational)\n"
        << "  --time-scheme bdf2|euler                time discretization (bdf2)\n"
        << "  --pressure-stab F   orthogonal-subscale pressure stabilization factor, 0 = off (0.3)\n"
        << "  --supg-tau-time T   fixed time scale in the SUPG tau [s], 0 = use dt (0.001)\n"
        << "  --pressure-precond amg|jacobi           PPE preconditioner (amg)\n"
        << "  --momentum-precond ilu|jacobi           momentum preconditioner: multicolour ILU(0) (ilu)\n"
        << "  --convection-form skew|convective       Galerkin convection operator (skew)\n"
        << "  --amg-rebuild N     rebuild the AMG hierarchy at least every N steps (50)\n"
        << "  --amg-remap N       after a remesh, carry the AMG hierarchy over to the new mesh for up\n"
        << "                      to N remeshes in a row before a full rebuild; 0 = always rebuild (0)\n"
        << "  --remesh-projection scheme|both|current|none\n"
        << "                      velocity re-projection after a remesh: u^n and u^{n-1} with the\n"
        << "                      scheme's pressure step (scheme), both / only u^n to zero discrete\n"
        << "                      divergence (both, current = pre-2026-10 behaviour), or none (scheme)\n"
        << "  --stats-window S    time window for force statistics [s]          (" << d.stats_window << ")\n"
        << "  --dump-systems N    write the momentum and pressure systems of step N to\n"
        << "                      <prefix>_momentum.bin and <prefix>_pressure.bin (solver studies)\n"
        << "  --prefix NAME       output file prefix                            (" << d.prefix << ")\n";
}

static double parse_double(const char* s, const char* name) {
    char* end = nullptr;
    double val = std::strtod(s, &end);
    if (end == s || *end != '\0') {
        fprintf(stderr, "Invalid value '%s' for %s\n", s, name);
        exit(EXIT_FAILURE);
    }
    return val;
}

static int parse_int(const char* s, const char* name) {
    char* end = nullptr;
    long val = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') {
        fprintf(stderr, "Invalid value '%s' for %s\n", s, name);
        exit(EXIT_FAILURE);
    }
    return (int)val;
}

// Returns false if the program should exit successfully (e.g. --help)
bool parse_options(int argc, char** argv, RunOptions& opt) {
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", a.c_str());
                exit(EXIT_FAILURE);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help")       { print_usage(argv[0]); return false; }
        else if (a == "--steps")              opt.num_steps          = parse_int(value(), "--steps");
        else if (a == "--dt")                 opt.dt                 = parse_double(value(), "--dt");
        else if (a == "--amr-interval")       opt.amr_interval       = parse_int(value(), "--amr-interval");
        else if (a == "--init-amr")           opt.init_amr           = parse_int(value(), "--init-amr");
        else if (a == "--max-level")          opt.amr.max_level      = parse_int(value(), "--max-level");
        else if (a == "--amr-tol-vort")       opt.amr.tol_vort       = parse_double(value(), "--amr-tol-vort");
        else if (a == "--amr-tol-interp")     opt.amr.tol_interp     = parse_double(value(), "--amr-tol-interp");
        else if (a == "--lookahead")          opt.lookahead          = parse_double(value(), "--lookahead");
        else if (a == "--coarsen-safety")     opt.amr.coarsen_safety = parse_double(value(), "--coarsen-safety");
        else if (a == "--buffer")             opt.amr.buffer_layers  = parse_int(value(), "--buffer");
        else if (a == "--max-elems")          opt.amr.max_elems      = parse_int(value(), "--max-elems");
        else if (a == "--print-interval")     opt.print_interval     = parse_int(value(), "--print-interval");
        else if (a == "--vtk-interval")       opt.vtk_interval       = parse_int(value(), "--vtk-interval");
        else if (a == "--stats-window")       opt.stats_window       = parse_double(value(), "--stats-window");
        else if (a == "--pressure-update") {
            std::string val = value();
            if (val == "rotational")    opt.rotational = true;
            else if (val == "standard") opt.rotational = false;
            else { fprintf(stderr, "Invalid value '%s' for --pressure-update\n", val.c_str()); exit(EXIT_FAILURE); }
        }
        else if (a == "--time-scheme") {
            std::string val = value();
            if (val == "bdf2")       opt.bdf2 = true;
            else if (val == "euler") opt.bdf2 = false;
            else { fprintf(stderr, "Invalid value '%s' for --time-scheme\n", val.c_str()); exit(EXIT_FAILURE); }
        }
        else if (a == "--pressure-stab")      opt.pressure_stab      = parse_double(value(), "--pressure-stab");
        else if (a == "--supg-tau-time")      opt.supg_tau_time      = parse_double(value(), "--supg-tau-time");
        else if (a == "--amg-rebuild")        opt.amg_rebuild        = parse_int(value(), "--amg-rebuild");
        else if (a == "--amr-overlap")        opt.amr_overlap        = parse_int(value(), "--amr-overlap");
        else if (a == "--amg-remap")          opt.amg_remap          = parse_int(value(), "--amg-remap");
        else if (a == "--convection-form") {
            std::string val = value();
            if (val == "skew")            opt.skew_convection = true;
            else if (val == "convective") opt.skew_convection = false;
            else { fprintf(stderr, "Invalid value '%s' for --convection-form\n", val.c_str()); exit(EXIT_FAILURE); }
        }
        else if (a == "--pressure-precond") {
            std::string val = value();
            if (val == "amg")         opt.amg = true;
            else if (val == "jacobi") opt.amg = false;
            else { fprintf(stderr, "Invalid value '%s' for --pressure-precond\n", val.c_str()); exit(EXIT_FAILURE); }
        }
        else if (a == "--momentum-precond") {
            std::string val = value();
            if (val == "ilu")         opt.momentum_ilu = true;
            else if (val == "jacobi") opt.momentum_ilu = false;
            else { fprintf(stderr, "Invalid value '%s' for --momentum-precond\n", val.c_str()); exit(EXIT_FAILURE); }
        }
        else if (a == "--remesh-projection") {
            std::string val = value();
            if (val == "none")         opt.remesh_projection = 0;
            else if (val == "current") opt.remesh_projection = 1;
            else if (val == "both")    opt.remesh_projection = 2;
            else if (val == "scheme")   opt.remesh_projection = 4;
            else { fprintf(stderr, "Invalid value '%s' for --remesh-projection\n", val.c_str()); exit(EXIT_FAILURE); }
        }
        else if (a == "--mms")                opt.mms = true;
        else if (a == "--amr-tol-scale") {
            std::string val = value();
            if (val == "inflow")     opt.relative_tol = true;
            else if (val == "fixed") opt.relative_tol = false;
            else { fprintf(stderr, "Invalid value '%s' for --amr-tol-scale\n", val.c_str()); exit(EXIT_FAILURE); }
        }
        else if (a == "--case") {
            std::string val = value();
            if (val == "2d2")      opt.case_2d3 = false;
            else if (val == "2d3") opt.case_2d3 = true;
            else { fprintf(stderr, "Invalid value '%s' for --case\n", val.c_str()); exit(EXIT_FAILURE); }
        }
        else if (a == "--rotational-div") {
            std::string val = value();
            if (val == "increment")     opt.rotational_from_dp = true;
            else if (val == "velocity") opt.rotational_from_dp = false;
            else { fprintf(stderr, "Invalid value '%s' for --rotational-div\n", val.c_str()); exit(EXIT_FAILURE); }
        }
        else if (a == "--uniform-level")      opt.uniform_level      = parse_int(value(), "--uniform-level");
        else if (a == "--min-level")          opt.amr.min_level      = parse_int(value(), "--min-level");
        else if (a == "--keep-radius")        opt.amr.keep_radius    = parse_double(value(), "--keep-radius");
        else if (a == "--amr-tol-radius")     opt.amr.tol_radius     = parse_double(value(), "--amr-tol-radius");
        else if (a == "--amr-tol-power")      opt.amr.tol_power      = parse_double(value(), "--amr-tol-power");
        else if (a == "--dump-systems")       opt.dump_systems       = parse_int(value(), "--dump-systems");
        else if (a == "--prefix")             opt.prefix             = value();
        else if (a.rfind("--", 0) == 0) {
            fprintf(stderr, "Unknown option %s (see --help)\n", a.c_str());
            exit(EXIT_FAILURE);
        } else {
            switch (positional++) {
                case 0: opt.num_steps       = parse_int(a.c_str(), "num_steps");        break;
                case 1: opt.amr_interval    = parse_int(a.c_str(), "amr_interval");     break;
                case 2: opt.amr.max_level   = parse_int(a.c_str(), "max_amr_level");    break;
                case 3: opt.amr.tol_vort = parse_double(a.c_str(), "amr_tol_vort"); break;
                default:
                    fprintf(stderr, "Too many positional arguments (see --help)\n");
                    exit(EXIT_FAILURE);
            }
        }
    }

    auto require = [](bool cond, const char* msg) {
        if (!cond) { fprintf(stderr, "Invalid options: %s\n", msg); exit(EXIT_FAILURE); }
    };
    require(opt.num_steps > 0, "--steps must be > 0");
    require(opt.dt > 0.0, "--dt must be > 0");
    require(opt.amr_interval >= 0 && opt.init_amr >= 0, "AMR intervals must be >= 0");
    require(opt.amr.max_level >= 0 && opt.amr.max_level <= 12, "--max-level must be in [0, 12]");
    require(opt.amr.tol_vort > 0.0 || opt.amr.tol_interp > 0.0,
            "at least one of --amr-tol-vort / --amr-tol-interp must be > 0");
    require(opt.lookahead >= 0.0, "--lookahead must be >= 0");
    require(opt.pressure_stab >= 0.0, "--pressure-stab must be >= 0");
    require(opt.supg_tau_time >= 0.0, "--supg-tau-time must be >= 0");
    require(opt.amg_rebuild > 0, "--amg-rebuild must be > 0");
    require(opt.amg_remap >= 0, "--amg-remap must be >= 0");
    require(opt.amr_overlap >= 0, "--amr-overlap must be >= 0");
    require(opt.amr_interval == 0 || opt.amr_overlap < opt.amr_interval,
            "--amr-overlap must be smaller than --amr-interval (set --amr-overlap 0 for a synchronous remesh)");
    require(opt.amr.tol_radius >= 0.0 && opt.amr.tol_power >= 0.0, "--amr-tol-radius and --amr-tol-power must be >= 0");
    require(opt.uniform_level >= 0, "--uniform-level must be >= 0");
    require(opt.amr.min_level >= 0 && opt.amr.min_level <= opt.amr.max_level, "--min-level must be in [0, max level]");
    if (opt.uniform_level > 0 || opt.mms) { opt.amr_interval = 0; opt.init_amr = 0; }
    require(opt.amr.coarsen_safety > 0.0 && opt.amr.coarsen_safety < 1.0, "--coarsen-safety must be in (0, 1)");
    require(opt.amr.buffer_layers >= 0, "--buffer must be >= 0");
    require(opt.amr.max_elems > 0, "--max-elems must be > 0");
    require(opt.print_interval > 0 && opt.vtk_interval >= 0, "output intervals must be valid");
    return true;
}
