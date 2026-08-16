#include "gui.h"
#include <api/api.h>

callback bool gui_application(gui_event_type_t event, gui_application_t core)
{
    bool is_ok = false;
    iwt_gui_data_t data = core->user_data;

    switch (event)
    {
        case GE_A_STARTUP:
            is_ok = true;
            break;

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
