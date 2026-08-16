#include "gui.h"
#include "init.h"
#include <api/api.h>
#include <stdio.h>
#include <time.h>

callback bool gui_application(gui_event_type_t event, gui_application_t core)
{
    bool is_ok = false;
    iwt_gui_data_t data = core->user_data;

    switch (event)
    {
        case GE_A_STARTUP:
        {
            data->cfg.T = 1.0;
            data->cfg.DT = 1.0e-12;
            data->cfg.hbar = 1.0;
            data->cfg.N = 4096;
            data->cfg.D = iwt_fractal_dimension();
            data->cfg.l0 = 1.0;
            data->cfg.MAX_ITER = 500;
            data->cfg.seed = (unsigned int)time(NULL);

            printf("=== IWT Parameter (aus Theorie) ===\n");
            printf("D               = %.12f\n", data->cfg.D);
            printf("l0              = %.12e m\n", data->cfg.l0);
            printf("T               = %.12e s\n", data->cfg.T);
            printf("\n=== Quantenfluktuationen (Anhang O & P) ===\n");
            printf("hbar            = %.12e (sim. Einheiten)\n", data->cfg.hbar);
            printf("seed            = %u\n", data->cfg.seed);
            printf("\n=== Abgeleitete Simulationsparameter ===\n");
            printf("DT              = %.12e\n", data->cfg.DT);
            printf("========================================\n\n");

            is_ok = ocl_initialize(&data->rt.ocl)
                && ocl_compile(&data->rt.ocl)
                && ocl_load_kernels(&data->rt.ocl)
                && initialize_host_data(&data->rt, &data->cfg)
                && initialize_gpu_data(&data->rt, &data->cfg);

            if (!is_ok)
            {
                fprintf(stderr, "IWT: OpenCL-/Daten-Initialisierung fehlgeschlagen.\n");
            }
            break;
        }

        case GE_A_ACTIVATE:
        {
            data->gl_area = gui_gl_create(data);
            GtkWidget* gl_frame = gui_frame_create("IWT Live View", data->gl_area);

            data->window = gui_main_window_create(core->app, 800, 800, data, false, true);
            gtk_window_set_child(GTK_WINDOW(data->window), gl_frame);

            is_ok = true;
            break;
        }

        case GE_A_SHUTDOWN:
            deinitialize_gpu_data(&data->rt);
            deinitialize_host_data(&data->rt);
            ocl_deinitialize(&data->rt.ocl);
            is_ok = true;
            break;

        default:
            break;
    }

    return is_ok;
}

callback void gui_gl(gui_gl_t core, gui_event_t e)
{
    switch (e->type)
    {
        case GE_GL_REALIZE:
            break;

        case GE_GL_RENDER:
            glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            break;

        default:
            break;
    }
}
