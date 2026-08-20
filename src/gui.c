#include "gui.h"
#include "init.h"
#include "iwt_kernel.h"
#include <api/api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum
{
    IWT_CTRL_MOTION = 1,
    IWT_CTRL_BETA,
    IWT_CTRL_GAMMA,
    IWT_CTRL_CLUSTER_THRESHOLD
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

private void _vec3_sub(float* out, const float* a, const float* b)
{
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

private void _vec3_cross(float* out, const float* a, const float* b)
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

private float _vec3_dot(const float* a, const float* b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

private void _vec3_normalize(float* v)
{
    float len = sqrtf(_vec3_dot(v, v));
    if (len > 1e-8f)
    {
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }
}

// Perspektivische Projektionsmatrix (spaltenweise, wie von GLSL erwartet)
private void _mat4_perspective(float* m, float fovy_rad, float aspect, float znear, float zfar)
{
    float f = 1.0f / tanf(fovy_rad * 0.5f);
    for (int i = 0; i < 16; i++) m[i] = 0.0f;
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (zfar + znear) / (znear - zfar);
    m[11] = -1.0f;
    m[14] = (2.0f * zfar * znear) / (znear - zfar);
}

// Kamera-View-Matrix (LookAt), spaltenweise
private void _mat4_look_at(float* m, const float* eye, const float* center, const float* up)
{
    float f[3], s[3], u[3];
    _vec3_sub(f, center, eye);
    _vec3_normalize(f);
    _vec3_cross(s, f, up);
    _vec3_normalize(s);
    _vec3_cross(u, s, f);

    m[0] = s[0]; m[4] = s[1]; m[8]  = s[2];  m[12] = -_vec3_dot(s, eye);
    m[1] = u[0]; m[5] = u[1]; m[9]  = u[2];  m[13] = -_vec3_dot(u, eye);
    m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2]; m[14] = _vec3_dot(f, eye);
    m[3] = 0.0f; m[7] = 0.0f; m[11] = 0.0f; m[15] = 1.0f;
}

private void _mat4_mul(float* out, const float* a, const float* b)
{
    float tmp[16];
    for (int col = 0; col < 4; col++)
    {
        for (int row = 0; row < 4; row++)
        {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++)
            {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            tmp[col * 4 + row] = sum;
        }
    }
    memcpy(out, tmp, sizeof(tmp));
}

// Mausrad-Zoom: dy>0 = nach unten scrollen = rauszoomen (Kamera weiter weg)
private gboolean _on_scroll(GtkEventControllerScroll* controller, double dx, double dy, gpointer user_data)
{
    (void)controller;
    (void)dx;
    iwt_gui_data_t data = user_data;

    data->zoom *= (dy > 0.0) ? 1.1f : 0.9f;
    if (data->zoom < 0.1f) data->zoom = 0.1f;
    if (data->zoom > 10.0f) data->zoom = 10.0f;

    return TRUE;
}

// Maus-Drag: Kamera per Klick+Ziehen um die Szene rotieren (Orbit)
private void _on_drag_begin(GtkGestureDrag* gesture, double start_x, double start_y, gpointer user_data)
{
    (void)gesture;
    (void)start_x;
    (void)start_y;
    iwt_gui_data_t data = user_data;
    data->drag_last_x = 0.0;
    data->drag_last_y = 0.0;
}

private void _on_drag_update(GtkGestureDrag* gesture, double offset_x, double offset_y, gpointer user_data)
{
    (void)gesture;
    iwt_gui_data_t data = user_data;

    double delta_x = offset_x - data->drag_last_x;
    double delta_y = offset_y - data->drag_last_y;

    data->cam_yaw   += (float)delta_x * 0.005f;
    data->cam_pitch += (float)delta_y * 0.005f;
    if (data->cam_pitch > 1.5f) data->cam_pitch = 1.5f;
    if (data->cam_pitch < -1.5f) data->cam_pitch = -1.5f;

    data->drag_last_x = offset_x;
    data->drag_last_y = offset_y;
}

private GLuint _create_points_program(void)
{
    static const char* vertex_source =
        "#version 150 core\n"
        "in vec3 pos;\n"
        "in vec3 color;\n"
        "uniform mat4 u_mvp;\n"
        "uniform float u_size_scale;\n"
        "out vec3 v_color;\n"
        "out float v_depth;\n"
        "void main()\n"
        "{\n"
        "    v_color = color;\n"
        "    gl_Position = u_mvp * vec4(pos, 1.0);\n"
        "    v_depth = gl_Position.w;\n"
        "    gl_PointSize = clamp((300.0 / v_depth) * u_size_scale, 1.0, 40.0);\n"
        "}\n";

    static const char* fragment_source =
        "#version 150 core\n"
        "in vec3 v_color;\n"
        "in float v_depth;\n"
        "out vec4 frag_color;\n"
        "void main()\n"
        "{\n"
        "    float fade = clamp(3.0 / v_depth, 0.3, 1.0);\n"
        "    frag_color = vec4(v_color * fade, 1.0);\n"
        "}\n";

    GLuint vs = _compile_shader(GL_VERTEX_SHADER, vertex_source);
    GLuint fs = _compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    if (vs == 0 || fs == 0) return 0;

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glBindAttribLocation(program, 0, "pos");
    glBindAttribLocation(program, 1, "color");
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
            data->cfg.N = 8192;
            data->cfg.D = iwt_fractal_dimension();
            data->cfg.l0 = 1.0;
            data->cfg.seed = (unsigned int)time(NULL);
            data->cfg.cluster_threshold = 1.2; // [1.0, 1.5]
			data->cfg.enable_motion = false;
            data->zoom = 1.0f;
            data->cam_yaw = 0.785398f;
            data->cam_pitch = 0.5236f;

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

            GtkEventController* scroll_controller = gtk_event_controller_scroll_new(GTK_EVENT_CONTROLLER_SCROLL_VERTICAL);
            g_signal_connect(scroll_controller, "scroll", G_CALLBACK(_on_scroll), data);
            gtk_widget_add_controller(data->gl_area, scroll_controller);

            GtkGesture* drag_gesture = gtk_gesture_drag_new();
            g_signal_connect(drag_gesture, "drag-begin", G_CALLBACK(_on_drag_begin), data);
            g_signal_connect(drag_gesture, "drag-update", G_CALLBACK(_on_drag_update), data);
            gtk_widget_add_controller(data->gl_area, GTK_EVENT_CONTROLLER(drag_gesture));

            struct gui_button_configuration motion_cfg = { .label = "Bewegung aktiv", .toggle = true };
            data->toggle_motion = gui_button_create(IWT_CTRL_MOTION, &motion_cfg, data);
            gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->toggle_motion), data->cfg.enable_motion);

            struct gui_spin_button_configuration beta_cfg = { .alignment = 0.5f, .value = data->cfg.beta, .min = 0.0, .max = 10.0, .increment = 0.01, .digits = 3 };
            data->spin_beta = gui_button_spin_create(IWT_CTRL_BETA, &beta_cfg, data);

            struct gui_spin_button_configuration gamma_cfg = { .alignment = 0.5f, .value = data->cfg.gamma, .min = 0.0, .max = 10.0, .increment = 0.01, .digits = 3 };
            data->spin_gamma = gui_button_spin_create(IWT_CTRL_GAMMA, &gamma_cfg, data);

            struct gui_spin_button_configuration threshold_cfg = { .alignment = 0.5f, .value = data->cfg.cluster_threshold, .min = 0.0, .max = 5.0, .increment = 0.01, .digits = 3 };
            data->spin_cluster_threshold = gui_button_spin_create(IWT_CTRL_CLUSTER_THRESHOLD, &threshold_cfg, data);

            GtkWidget* label_beta = gtk_label_new("beta:");
            GtkWidget* label_gamma = gtk_label_new("gamma:");
            GtkWidget* label_threshold = gtk_label_new("cluster_threshold:");

            GtkWidget* control_box = gui_box_horizontal_create(8);
            gui_box_append_widget(control_box, data->toggle_motion);
            gui_box_append_widget(control_box, label_beta);
            gui_box_append_widget(control_box, data->spin_beta);
            gui_box_append_widget(control_box, label_gamma);
            gui_box_append_widget(control_box, data->spin_gamma);
            gui_box_append_widget(control_box, label_threshold);
            gui_box_append_widget(control_box, data->spin_cluster_threshold);

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
            free(data->node_colors);
            data->node_colors = NULL;
            free(data->points_buffer);
            data->points_buffer = NULL;
            free(data->cluster_points_buffer);
            data->cluster_points_buffer = NULL;
            is_ok = true;
            break;

        default:
            break;
    }

    return is_ok;
}

callback void gui_main_window(gui_main_window_t core, gui_event_t e)
{
    iwt_gui_data_t data = core->user_data;

    if (e->type == GE_KEY_PRESSED)
    {
        const float step = 0.05f;
        switch (e->data.key_pressed.keyval)
        {
            case GDK_KEY_Left:
                data->cam_yaw -= step;
                e->data.key_pressed.handled = true;
                break;
            case GDK_KEY_Right:
                data->cam_yaw += step;
                e->data.key_pressed.handled = true;
                break;
            case GDK_KEY_Up:
                data->cam_pitch += step;
                if (data->cam_pitch > 1.5f) data->cam_pitch = 1.5f;
                e->data.key_pressed.handled = true;
                break;
            case GDK_KEY_Down:
                data->cam_pitch -= step;
                if (data->cam_pitch < -1.5f) data->cam_pitch = -1.5f;
                e->data.key_pressed.handled = true;
                break;
            default:
                break;
        }
    }
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
            else if (core->id == IWT_CTRL_CLUSTER_THRESHOLD)
            {
                data->cfg.cluster_threshold = gui_button_spin_get_double(core->button);
                iwt_recompute_adjacency(&data->rt, &data->cfg);
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
            data->node_colors = malloc((size_t)data->cfg.N * 3 * sizeof(float));
            data->points_buffer = malloc((size_t)data->cfg.N * 6 * sizeof(float));
            data->cluster_points_buffer = malloc((size_t)data->rt.cluster_capacity * 6 * sizeof(float));

            data->gl_program = _create_points_program();

            glGenVertexArrays(1, &data->gl_vao);
            glGenBuffers(1, &data->gl_vbo);
            glBindVertexArray(data->gl_vao);
            glBindBuffer(GL_ARRAY_BUFFER, data->gl_vbo);
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)((size_t)data->cfg.N * 6 * sizeof(float)), NULL, GL_DYNAMIC_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
            glBindVertexArray(0);

            glGenVertexArrays(1, &data->gl_vao_clusters);
            glGenBuffers(1, &data->gl_vbo_clusters);
            glBindVertexArray(data->gl_vao_clusters);
            glBindBuffer(GL_ARRAY_BUFFER, data->gl_vbo_clusters);
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)((size_t)data->rt.cluster_capacity * 6 * sizeof(float)), NULL, GL_DYNAMIC_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
            glBindVertexArray(0);

            glEnable(GL_PROGRAM_POINT_SIZE);

            data->gl_u_mvp = glGetUniformLocation(data->gl_program, "u_mvp");
            data->gl_u_size_scale = glGetUniformLocation(data->gl_program, "u_size_scale");
            break;
        }

		case GE_GL_RENDER:
		{
			run_simulation_step(&data->rt, &data->cfg);
			data->iter++;

			// --- Knotenwolke (alle N Knoten) ---
			iwt_compute_node_colors(data->rt.mass, data->rt.charge, data->cfg.N, data->node_colors);
			for (size_t i = 0; i < data->cfg.N; i++)
			{
				data->points_buffer[i * 6 + 0] = (float)data->rt.pos_x[i];
				data->points_buffer[i * 6 + 1] = (float)data->rt.pos_y[i];
				data->points_buffer[i * 6 + 2] = (float)data->rt.pos_z[i];
				data->points_buffer[i * 6 + 3] = data->node_colors[i * 3 + 0];
				data->points_buffer[i * 6 + 4] = data->node_colors[i * 3 + 1];
				data->points_buffer[i * 6 + 5] = data->node_colors[i * 3 + 2];
			}

			// --- Cluster-Schwerpunkte ---
			size_t cluster_draw_count = 0;
			for (size_t c = 0; c < data->rt.cluster_count; c++)
			{
				iwt_cluster_t cl = &data->rt.clusters[c];
				if (!cl->is_active) continue;

				data->cluster_points_buffer[cluster_draw_count * 6 + 0] = (float)cl->x;
				data->cluster_points_buffer[cluster_draw_count * 6 + 1] = (float)cl->y;
				data->cluster_points_buffer[cluster_draw_count * 6 + 2] = (float)cl->z;

				// Farbe: Rot = positive Ladung, Blau = negative Ladung
				float charge_norm = (float)(cl->charge / (fabs(cl->charge) + 1.0));
				float brightness = (float)(cl->mass / (cl->mass + 1.0));

				if (charge_norm > 0.0f) {
					data->cluster_points_buffer[cluster_draw_count * 6 + 3] = brightness;      // R
					data->cluster_points_buffer[cluster_draw_count * 6 + 4] = 0.0f;            // G
					data->cluster_points_buffer[cluster_draw_count * 6 + 5] = 0.0f;            // B
				} else {
					data->cluster_points_buffer[cluster_draw_count * 6 + 3] = 0.0f;            // R
					data->cluster_points_buffer[cluster_draw_count * 6 + 4] = 0.0f;            // G
					data->cluster_points_buffer[cluster_draw_count * 6 + 5] = brightness;      // B
				}

				cluster_draw_count++;
			}

			glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);

			glUseProgram(data->gl_program);

			int width = gtk_widget_get_width(data->gl_area);
			int height = gtk_widget_get_height(data->gl_area);
			float aspect = (height > 0) ? (float)width / (float)height : 1.0f;

			float radius = 3.6f * data->zoom;
			float eye[3] = {
				radius * cosf(data->cam_pitch) * cosf(data->cam_yaw),
				radius * sinf(data->cam_pitch),
				radius * cosf(data->cam_pitch) * sinf(data->cam_yaw)
			};
			float center[3] = { 0.0f, 0.0f, 0.0f };
			float up[3]     = { 0.0f, 1.0f, 0.0f };

			float view[16], proj[16], mvp[16];
			_mat4_look_at(view, eye, center, up);
			_mat4_perspective(proj, 0.785398f, aspect, 0.1f, 200.0f);
			_mat4_mul(mvp, proj, view);

			glUniformMatrix4fv(data->gl_u_mvp, 1, GL_FALSE, mvp);

			// 1. Knotenwolke zeichnen (klein)
			glUniform1f(data->gl_u_size_scale, 1.0f);
			glBindBuffer(GL_ARRAY_BUFFER, data->gl_vbo);
			glBufferSubData(GL_ARRAY_BUFFER, 0,
				(GLsizeiptr)((size_t)data->cfg.N * 6 * sizeof(float)), data->points_buffer);
			glBindVertexArray(data->gl_vao);
			glDrawArrays(GL_POINTS, 0, (GLsizei)data->cfg.N);

			// 2. Cluster-Schwerpunkte darueber zeichnen (gross/hell)
			glUniform1f(data->gl_u_size_scale, 3.0f);
			glBindBuffer(GL_ARRAY_BUFFER, data->gl_vbo_clusters);
			glBufferSubData(GL_ARRAY_BUFFER, 0,
				(GLsizeiptr)(cluster_draw_count * 6 * sizeof(float)), data->cluster_points_buffer);
			glBindVertexArray(data->gl_vao_clusters);
			glDrawArrays(GL_POINTS, 0, (GLsizei)cluster_draw_count);

			glBindVertexArray(0);
			break;
		}

        default:
            break;
    }
}
