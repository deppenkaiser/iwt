#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <ocl/ocl.h>
#include <string/string.h>
#include "iwt_kernel.h"
#include "init.h"

int main(int argc, char** argv)
{
    int retval = 1;
    struct iwt_config cfg = {0};
    struct iwt_runtime rt = {0};

    bool load_state = false;
    bool enable_fluctuations = true;
    unsigned int seed = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--load") == 0)
        {
            load_state = true;
        }
        else if (strcmp(argv[i], "--no-fluctuations") == 0)
        {
            enable_fluctuations = false;
        }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
        {
            seed = (unsigned int)atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            printf("Verwendung: %s [Optionen]\n", argv[0]);
            printf("  -l, --load            Lade gespeicherten Zustand\n");
            printf("  --no-fluctuations     Deaktiviere Quantenfluktuationen\n");
            printf("  --seed <n>            Setze Seed für Zufallsgenerator\n");
            printf("  -h, --help            Zeige diese Hilfe\n");
            return 0;
        }
    }

    if (seed == 0)
    {
        seed = (unsigned int)time(NULL);
    }

    cfg.N = 4096;
    cfg.BATCH_SIZE = 512;
    cfg.D = iwt_fractal_dimension();
    cfg.l0 = 1.0;
    cfg.T = 1.0;
    cfg.alpha_IWT = 1.0;
    cfg.beta_IWT = 1.0;
    cfg.MAX_ITER = 1000;
    cfg.I_max = iwt_I_max();
    cfg.I_min = iwt_I_min();
    cfg.DT = 1.0e-2;
    cfg.I_vac = iwt_I_min();
    cfg.phi_0 = iwt_pi() / 2.0;
    cfg.omega_0 = 1.0 / cfg.DT;
    cfg.Z_0 = 1.0;
    cfg.alpha_Z = 0.1;

    cfg.hbar = 1.0;
    cfg.uncertainty_scale = sqrt(cfg.hbar / (2.0 * cfg.DT));
    cfg.enable_fluctuations = enable_fluctuations;
    cfg.seed = seed;

    // OpenCL Initialisierung (Debug-Ausgaben werden später überschrieben)
    if (!ocl_initialize(&rt.ocl))
    {
        return 1;
    }

    if (!ocl_compile(&rt.ocl))
    {
        ocl_deinitialize(&rt.ocl);
        return 1;
    }

    if (!ocl_load_kernels(&rt.ocl))
    {
        ocl_deinitialize(&rt.ocl);
        return 1;
    }

    // Bildschirm löschen und kurze Info
    string_clear_screen();
    string_set_cursor_position(1, 1);

    // Diese 3 Zeilen werden von iwt_print_status() überschrieben
    // Sie sind nur für den Fall sichtbar, dass die Simulation abstürzt
    printf("IWT Simulation (N=%zu, MAX_ITER=%d)                                             \n", cfg.N, cfg.MAX_ITER);
    printf("Fluktuationen: %s | Seed: %u                                                    \n", cfg.enable_fluctuations ? "AKTIVIERT" : "DEAKTIVIERT", cfg.seed);
    printf("                                                                                \n");

    if (initialize_host_data(&rt, &cfg))
    {
        if (initialize_gpu_data(&rt, &cfg))
        {
            if (load_state)
            {
                if (iwt_load_state(&rt, &cfg, "iwt_state.bin"))
                {
                    printf("State loaded from iwt_state.bin\n");
                }
                else
                {
                    printf("Warning: Could not load state, starting fresh\n");
                }
            }
            else
            {
                remove("iwt_state.bin");
            }

            // Cursor auf (1,1) setzen und Simulation starten
            string_set_cursor_position(1, 1);

            if (run_simulation(&rt, &cfg))
            {
                retval = 0;
            }

            deinitialize_gpu_data(&rt);
        }
    }

    deinitialize_host_data(&rt);
    ocl_deinitialize(&rt.ocl);

    return retval;
}
