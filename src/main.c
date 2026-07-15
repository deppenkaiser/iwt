#include <stdio.h>
#include <ocl/ocl.h>
#include "iwt_kernel.h"
#include "init.h"

int main(void)
{
    int retval = 1;
    struct iwt_config cfg = {0};
    struct iwt_runtime rt = {0};

	// 1. Fundamentale Parameter setzen
	cfg.N = 4096;
	cfg.BATCH_SIZE = 512;
	cfg.D = iwt_fractal_dimension();
	cfg.l0 = 1.0;
	cfg.T = 1.0;
	cfg.alpha_IWT = 1.0;
	cfg.beta_IWT = 1.0;
	cfg.MAX_ITER = 100;
	cfg.I_min = iwt_I_min();
	cfg.I_max = iwt_I_max();

	// 2. Abgeleitete Parameter berechnen
	cfg.DT = 1e-3;
	cfg.ETA = 1e2;
	cfg.LAMBDA = 1e2;
	cfg.MU = 0.001;
	cfg.ETA_Q = 0.001;
	cfg.LAMBDA_Q = 0.01;
	cfg.GAMMA_Q = 0.001;
	cfg.GAMMA_ENTROPY = 0.01;
	cfg.I0 = 0.01;
	cfg.ETA_Q_POTENTIAL = 0.001;
	cfg.ETA_SOURCE = 1.0e0;
	cfg.PULSE_INTERVAL = 1;

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
	printf("ETA             = %.12e\n", cfg.ETA);
	printf("LAMBDA          = %.12e\n", cfg.LAMBDA);
	printf("ETA_Q           = %.12e\n", cfg.ETA_Q);
	printf("LAMBDA_Q        = %.12e\n", cfg.LAMBDA_Q);
	printf("GAMMA_Q         = %.12e\n", cfg.GAMMA_Q);
	printf("========================================\n\n");

    // 3. OpenCL initialisieren
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
						if (iwt_load_state(&rt, &cfg, "iwt_state.bin"))
        					printf("State loaded from iwt_state.bin\n");

                        if (run_simulation(&rt, &cfg))
                        {
							struct iwt_spectrum spec = {0};
							iwt_compute_spectrum(rt.I, cfg.N, &spec);
							iwt_print_spectrum(&spec);

							if (iwt_save_state(&rt, &cfg, "iwt_state.bin"))
								printf("State saved to iwt_state.bin\n");

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
