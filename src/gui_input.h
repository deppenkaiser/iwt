#pragma once
#include <gtk/gtk.h>

gboolean gui_input_on_scroll(GtkEventControllerScroll* controller, double dx, double dy, gpointer user_data);
void gui_input_on_drag_begin(GtkGestureDrag* gesture, double start_x, double start_y, gpointer user_data);
void gui_input_on_drag_update(GtkGestureDrag* gesture, double offset_x, double offset_y, gpointer user_data);
