#pragma once

#include <GL/gl.h>
#include <stdbool.h>

void gui_math_vec3_sub(float* out, const float* a, const float* b);
void gui_math_vec3_cross(float* out, const float* a, const float* b);
float gui_math_vec3_dot(const float* a, const float* b);
void gui_math_vec3_normalize(float* v);
void gui_math_mat4_perspective(float* m, float fovy_rad, float aspect, float znear, float zfar);
void gui_math_mat4_ortho(float* m, float left, float right, float bottom, float top, float znear, float zfar);
void gui_math_mat4_look_at(float* m, const float* eye, const float* center, const float* up);
void gui_math_mat4_mul(float* out, const float* a, const float* b);
