#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <ocl/ocl.h>
#include "iwt_kernel.h"
#include "init.h"

int main(int argc, char** argv)
{
    int retval = 1;
    struct iwt_config cfg = {0};
    struct iwt_runtime rt = {0};

    bool load_state = false;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--load") == 0)
        {
            load_state = true;
        }
    }

    cfg.N = 4096;
    cfg.BATCH_SIZE = 512;
    cfg.D = iwt_fractal_dimension();
    cfg.l0 = 1.0;
    cfg.T = 1.0;
    cfg.alpha_IWT = 1.0;
    cfg.beta_IWT = 1.0;
    cfg.MAX_ITER = 300;
    cfg.I_min = iwt_I_min();
    cfg.I_max = iwt_I_max();
    cfg.DT = 1e-4;

    printf("=== IWT Parameter (aus Theorie) ===\n");
    printf("D               = %.12f\n", cfg.D);
    printf("l0              = %.12e m\n", cfg.l0);
    printf("T               = %.12e s\n", cfg.T);
    printf("alpha_IWT       = %.12e\n", cfg.alpha_IWT);
    printf("beta_IWT        = %.12e\n", cfg.beta_IWT);
    printf("I_min           = %.12f\n", iwt_I_min());
    printf("I_max           = %.12f\n", iwt_I_max());
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
                            if (remove("iwt_state.bin") == 0)
                            {
                                printf("Removed existing state file\n");
                            }
                        }

                        if (run_simulation(&rt, &cfg))
                        {
                            struct iwt_spectrum spec = {0};
                            iwt_compute_spectrum(rt.I, cfg.N, &spec);
                            iwt_print_spectrum(&spec);

                            struct iwt_mds mds = {0};
                            if (iwt_mds_compute(&rt, &cfg, &mds))
                            {
                                iwt_mds_print(&mds, 10);
                                iwt_mds_save_pgm(&mds, rt.I, cfg.N, "iwt_visual.pgm");
                                printf("Visualisierung gespeichert: iwt_visual.pgm\n");
                                iwt_mds_free(&mds);
                            }

                            if (iwt_save_state(&rt, &cfg, "iwt_state.bin"))
                            {
                                printf("State saved to iwt_state.bin\n");
                            }

                            printf("N = %ld, DT = %f\n", cfg.N, cfg.DT);
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