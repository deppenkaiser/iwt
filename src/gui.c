#include "gui.h"
#include "init.h"
#include "iwt_kernel.h"
#include <api/api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum
{
    IWT_CTRL_MOTION = 1,
    IWT_CTRL_BETA,
    IWT_CTRL_GAMMA
};

private GLuint _compile_shader(GLenum type, const char* source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint status = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        char log[512] = {0};
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
        fprintf(stderr, "IWT: Shader-Fehler: %s\n", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

private GLuint _create_overlay_program(void)
{
    static const char* vertex_source =
        "#version 150 core\n"
        "in vec2 pos;\n"
        "in vec2 uv;\n"
        "out vec2 v_uv;\n"
        "void main()\n"
        "{\n"
        "    v_uv = uv;\n"
        "    gl_Position = vec4(pos, 0.0, 1.0);\n"
        "}\n";

    static const char* fragment_source =
        "#version 150 core\n"
        "in vec2 v_uv;\n"
        "out vec4 frag_color;\n"
        "uniform sampler2D tex;\n"
        "void main()\n"
        "{\n"
        "    frag_color = texture(tex, v_uv);\n"
        "}\n";

    GLuint vs = _compile_shader(GL_VERTEX_SHADER, vertex_source);
    GLuint fs = _compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    if (vs == 0 || fs == 0) return 0;

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glBindAttribLocation(program, 0, "pos");
    glBindAttribLocation(program, 1, "uv");
    glLinkProgram(program);

    GLint status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        char log[512] = {0};
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "IWT: Programm-Link-Fehler: %s\n", log);
        glDeleteProgram(program);
        program = 0;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

callback bool gui_application(gui_event_type_t event, gui_application_t core)
{
    bool is_ok = false;
    iwt_gui_data_t data = core->user_data;

    switch (event)
    {
        case GE_A_STARTUP:
        {
			data->cfg.gamma = 1.0; // Struktur bildend
			data->cfg.beta = 1.0; // Bohm-Kopplungsstärke
            data->cfg.T = 1.0;
            data->cfg.DT = 1.0e-12;
            data->cfg.hbar = 1.0;
            data->cfg.N = 4096;
            data->cfg.D = iwt_fractal_dimension();
            data->cfg.l0 = 1.0;
            data->cfg.MAX_ITER = 500;
            data->cfg.seed = (unsigned int)time(NULL);
            data->cfg.cluster_threshold = 1.5;
			data->cfg.enable_motion = false;

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

            if (is_ok)
            {
                // === Anfangszustand (Vakuum + Basisamplitude) ===
                for (size_t i = 0; i < data->cfg.N; i++)
                {
                    data->rt.I_real[i] = 0.0;
                    data->rt.I_imag[i] = 0.0;
                    data->rt.I_phase[i] = 0.0;
                    data->rt.I_prev_real[i] = 0.0;
                    data->rt.I_prev_imag[i] = 0.0;
                    data->rt.I_phase_prev[i] = 0.0;
                }
            }
            else
            {
                fprintf(stderr, "IWT: OpenCL-/Daten-Initialisierung fehlgeschlagen.\n");
            }
            break;
        }

        case GE_A_ACTIVATE:
        {
            data->gl_area = gui_gl_create(data);
            GtkWidget* gl_frame = gui_frame_create("IWT Live View", data->gl_area);
            gtk_widget_set_vexpand(gl_frame, TRUE);
            gtk_widget_set_hexpand(gl_frame, TRUE);

            struct gui_button_configuration motion_cfg = { .label = "Bewegung aktiv", .toggle = true };
            data->toggle_motion = gui_button_create(IWT_CTRL_MOTION, &motion_cfg, data);
            gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->toggle_motion), data->cfg.enable_motion);

            struct gui_spin_button_configuration beta_cfg = { .alignment = 0.5f, .value = data->cfg.beta, .min = 0.0, .max = 10.0, .increment = 0.01, .digits = 3 };
            data->spin_beta = gui_button_spin_create(IWT_CTRL_BETA, &beta_cfg, data);

            struct gui_spin_button_configuration gamma_cfg = { .alignment = 0.5f, .value = data->cfg.gamma, .min = 0.0, .max = 10.0, .increment = 0.01, .digits = 3 };
            data->spin_gamma = gui_button_spin_create(IWT_CTRL_GAMMA, &gamma_cfg, data);

            GtkWidget* label_beta = gtk_label_new("beta:");
            GtkWidget* label_gamma = gtk_label_new("gamma:");

            GtkWidget* control_box = gui_box_horizontal_create(8);
            gui_box_append_widget(control_box, data->toggle_motion);
            gui_box_append_widget(control_box, label_beta);
            gui_box_append_widget(control_box, data->spin_beta);
            gui_box_append_widget(control_box, label_gamma);
            gui_box_append_widget(control_box, data->spin_gamma);

            GtkWidget* main_box = gui_box_vertical_create(4);
            gui_box_append_widget(main_box, control_box);
            gui_box_append_widget(main_box, gl_frame);

            data->window = gui_main_window_create(core->app, 800, 800, data, false, true);
            gtk_window_set_child(GTK_WINDOW(data->window), main_box);

            is_ok = true;
            break;
        }

        case GE_A_SHUTDOWN:
            deinitialize_gpu_data(&data->rt);
            deinitialize_host_data(&data->rt);
            ocl_deinitialize(&data->rt.ocl);
            free(data->overlay_rgb);
            data->overlay_rgb = NULL;
            is_ok = true;
            break;

        default:
            break;
    }

    return is_ok;
}

callback void gui_button(gui_button_t core, gui_event_t e)
{
    iwt_gui_data_t data = core->user_data;

    switch (e->type)
    {
        case GE_B_TOGGLED:
            if (core->id == IWT_CTRL_MOTION)
            {
                data->cfg.enable_motion = e->data.b_toggled.active;
            }
            break;

        case GE_B_SELECTED:
            if (core->id == IWT_CTRL_BETA)
            {
                data->cfg.beta = gui_button_spin_get_double(core->button);
            }
            else if (core->id == IWT_CTRL_GAMMA)
            {
                data->cfg.gamma = gui_button_spin_get_double(core->button);
            }
            break;

        default:
            break;
    }
}

callback void gui_gl(gui_gl_t core, gui_event_t e)
{
    iwt_gui_data_t data = core->user_data;

    switch (e->type)
    {
        case GE_GL_REALIZE:
        {
            data->overlay_width = (int)sqrt((double)data->cfg.N);
            data->overlay_height = data->overlay_width;
            data->overlay_rgb = calloc((size_t)data->overlay_width * data->overlay_height * 3, sizeof(unsigned char));

            glGenTextures(1, &data->gl_texture);
            glBindTexture(GL_TEXTURE_2D, data->gl_texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, data->overlay_width, data->overlay_height, 0,
                GL_RGB, GL_UNSIGNED_BYTE, data->overlay_rgb);

            data->gl_program = _create_overlay_program();

            static const float quad_vertices[] = {
                // pos          // uv
                -1.0f, -1.0f,   0.0f, 1.0f,
                 1.0f, -1.0f,   1.0f, 1.0f,
                 1.0f,  1.0f,   1.0f, 0.0f,
                -1.0f, -1.0f,   0.0f, 1.0f,
                 1.0f,  1.0f,   1.0f, 0.0f,
                -1.0f,  1.0f,   0.0f, 0.0f,
            };

            glGenVertexArrays(1, &data->gl_vao);
            glGenBuffers(1, &data->gl_vbo);
            glBindVertexArray(data->gl_vao);
            glBindBuffer(GL_ARRAY_BUFFER, data->gl_vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
            glBindVertexArray(0);
            break;
        }

        case GE_GL_RENDER:
        {
            run_simulation_step(&data->rt, &data->cfg);
            data->iter++;

            iwt_build_overlay_rgb(data->rt.mass, data->rt.charge, data->cfg.N,
                data->overlay_rgb, data->overlay_width, data->overlay_height);

            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            glBindTexture(GL_TEXTURE_2D, data->gl_texture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data->overlay_width, data->overlay_height,
                GL_RGB, GL_UNSIGNED_BYTE, data->overlay_rgb);

            glUseProgram(data->gl_program);
            glUniform1i(glGetUniformLocation(data->gl_program, "tex"), 0);
            glBindVertexArray(data->gl_vao);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindVertexArray(0);
            break;
        }

        default:
            break;
    }
}
