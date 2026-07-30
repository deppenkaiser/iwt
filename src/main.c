// ============================================================================
// main.c - GEBEREINIGT
// ============================================================================

#include "init.h"
#include "iwt_kernel.h"
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

    unsigned int seed = (unsigned int)time(NULL);

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
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
    cfg.N = 4096;
    cfg.D = iwt_fractal_dimension();
    cfg.l0 = 1.0;
    cfg.MAX_ITER = 500;
    cfg.seed = seed;

    printf("=== IWT Parameter (aus Theorie) ===\n");
    printf("D               = %.12f\n", cfg.D);
    printf("l0              = %.12e m\n", cfg.l0);
    printf("T               = %.12e s\n", cfg.T);
    printf("\n=== Quantenfluktuationen (Anhang O & P) ===\n");
    printf("hbar            = %.12e (sim. Einheiten)\n", cfg.hbar);
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
