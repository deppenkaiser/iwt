#include "gui_input.h"
#include "gui.h"

gboolean gui_input_on_scroll(GtkEventControllerScroll* controller, double dx, double dy, gpointer user_data)
{
    (void)controller; (void)dx;
    iwt_gui_data_t data = user_data;
    data->zoom *= (dy > 0.0) ? 1.1f : 0.9f;
    if (data->zoom < 0.1f) data->zoom = 0.1f;
    if (data->zoom > 10.0f) data->zoom = 10.0f;
    return TRUE;
}

void gui_input_on_drag_begin(GtkGestureDrag* gesture, double start_x, double start_y, gpointer user_data)
{
    (void)gesture; (void)start_x; (void)start_y;
    iwt_gui_data_t data = user_data;
    data->drag_last_x = 0.0;
    data->drag_last_y = 0.0;
}

void gui_input_on_drag_update(GtkGestureDrag* gesture, double offset_x, double offset_y, gpointer user_data)
{
    (void)gesture;
    iwt_gui_data_t data = user_data;
    double delta_x = offset_x - data->drag_last_x;
    double delta_y = offset_y - data->drag_last_y;
    data->cam_yaw += (float)delta_x * 0.005f;
    data->cam_pitch += (float)delta_y * 0.005f;
    if (data->cam_pitch > 1.5f) data->cam_pitch = 1.5f;
    if (data->cam_pitch < -1.5f) data->cam_pitch = -1.5f;
    data->drag_last_x = offset_x;
    data->drag_last_y = offset_y;
}
