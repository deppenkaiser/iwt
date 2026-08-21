#pragma once
#include <gui/gui.h>

GLuint gui_shader_compile(GLenum type, const char* source);
GLuint gui_shader_create_points_program(void);
