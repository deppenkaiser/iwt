// ============================================================================
// main.c - GEBEREINIGT
// ============================================================================

#include "init.h"
#include "iwt_kernel.h"
#include <math.h>
#include <ocl/ocl.h>
#include <stdio.h>
#include <string.h>
#include <string/string.h>
#include <time.h>

int main(int argc, char **argv)
{
    int retval = 1;
    struct iwt_config cfg = {0};
    struct iwt_runtime rt = {0};

    bool enable_fluctuations = true;
    unsigned int seed = (unsigned int)time(NULL);

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--no-fluctuations") == 0)
        {
            enable_fluctuations = false;
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            printf("Verwendung: %s [Optionen]\n", argv[0]);
            printf("  --no-fluctuations     Deaktiviere Quantenfluktuationen\n");
            printf("  -h, --help            Zeige diese Hilfe\n");
            return 0;
        }
    }

    cfg.T = 1.0;
    cfg.DT = 1.0e-12;
    cfg.hbar = 1.0;
    cfg.uncertainty_scale = sqrt(cfg.hbar / (2.0 * cfg.T));
    cfg.N = 4096;
    cfg.BATCH_SIZE = 512;
    cfg.D = iwt_fractal_dimension();
    cfg.l0 = 1.0;
    cfg.alpha_IWT = 1.0;
    cfg.beta_IWT = 1.0;
    cfg.MAX_ITER = 500;
    cfg.I_max = iwt_I_max();
    cfg.I_min = iwt_I_min();
    cfg.I_vac = 0.1;
    cfg.phi_0 = iwt_pi() / 2.0;
    cfg.omega_0 = 1.0 / cfg.DT;
    cfg.Z_0 = 1.0;
    cfg.alpha_Z = 0.1;
    cfg.enable_fluctuations = enable_fluctuations;
    cfg.seed = seed;

    printf("=== IWT Parameter (aus Theorie) ===\n");
    printf("D               = %.12f\n", cfg.D);
    printf("l0              = %.12e m\n", cfg.l0);
    printf("T               = %.12e s\n", cfg.T);
    printf("alpha_IWT       = %.12e\n", cfg.alpha_IWT);
    printf("beta_IWT        = %.12e\n", cfg.beta_IWT);
    printf("I_min           = %.12f\n", iwt_I_min());
    printf("I_max           = %.12f\n", iwt_I_max());
    printf("Z_0             = %.12e\n", cfg.Z_0);
    printf("\n=== Quantenfluktuationen (Anhang O & P) ===\n");
    printf("hbar            = %.12e (sim. Einheiten)\n", cfg.hbar);
    printf("uncertainty_scale = %.12e  [√(ℏ/(2·T))]\n", cfg.uncertainty_scale);
    printf("enable_fluctuations = %s\n", cfg.enable_fluctuations ? "JA" : "NEIN");
    printf("seed            = %u\n", cfg.seed);
    printf("\n=== Abgeleitete Simulationsparameter ===\n");
    printf("DT              = %.12e\n", cfg.DT);
    printf("========================================\n\n");

    if (ocl_initialize(&rt.ocl))
    {
        if (ocl_compile(&rt.ocl))
        {
            if (ocl_load_kernels(&rt.ocl))
            {
                if (initialize_host_data(&rt, &cfg))
                {
                    if (initialize_gpu_data(&rt, &cfg))
                    {
                        if (run_simulation(&rt, &cfg))
                        {
                            retval = 0;
                        }
                        deinitialize_gpu_data(&rt);
                    }
                }
                deinitialize_host_data(&rt);
            }
        }
        ocl_deinitialize(&rt.ocl);
    }

    return retval;
}
