#include "gui_math.h"
#include <string.h>
#include <math.h>
#include <vector/vector.h>

void gui_math_vec3_sub(float* out, const float* a, const float* b)
{
    struct vector_3d va = {(ld) a[0], (ld) a[1], (ld) a[2]};
    struct vector_3d vb = {(ld) b[0], (ld) b[1], (ld) b[2]};
    struct vector_3d v = vector_sub(&va, &vb);
    out[0] = (float) v.x;
    out[1] = (float) v.y;
    out[2] = (float) v.z;
}

void gui_math_vec3_cross(float* out, const float* a, const float* b)
{
    struct vector_3d va = {(ld) a[0], (ld) a[1], (ld) a[2]};
    struct vector_3d vb = {(ld) b[0], (ld) b[1], (ld) b[2]};
    struct vector_3d v = vector_cross(&va, &vb);
    out[0] = (float) v.x;
    out[1] = (float) v.y;
    out[2] = (float) v.z;
}

float gui_math_vec3_dot(const float* a, const float* b)
{
    struct vector_3d va = {(ld) a[0], (ld) a[1], (ld) a[2]};
    struct vector_3d vb = {(ld) b[0], (ld) b[1], (ld) b[2]};
    ld d = vector_dot(&va, &vb);
    return (float) d;
}

void gui_math_vec3_normalize(float* v)
{
    struct vector_3d vv = {(ld) v[0], (ld) v[1], (ld) v[2]};
    struct vector_3d n = vector_normalize(&vv);
    v[0] = (float) n.x;
    v[1] = (float) n.y;
    v[2] = (float) n.z;
}

void gui_math_mat4_perspective(float* m, float fovy_rad, float aspect, float znear, float zfar)
{
    float f = 1.0f / tanf(fovy_rad * 0.5f);
    for (int i = 0; i < 16; i++) m[i] = 0.0f;
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (zfar + znear) / (znear - zfar);
    m[11] = -1.0f;
    m[14] = (2.0f * zfar * znear) / (znear - zfar);
}

void gui_math_mat4_look_at(float* m, const float* eye, const float* center, const float* up)
{
    float f[3], s[3], u[3];
    gui_math_vec3_sub(f, center, eye);
    gui_math_vec3_normalize(f);
    gui_math_vec3_cross(s, f, up);
    gui_math_vec3_normalize(s);
    gui_math_vec3_cross(u, s, f);

    m[0] = s[0]; m[4] = s[1]; m[8] = s[2]; m[12] = -gui_math_vec3_dot(s, eye);
    m[1] = u[0]; m[5] = u[1]; m[9] = u[2]; m[13] = -gui_math_vec3_dot(u, eye);
    m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2]; m[14] = gui_math_vec3_dot(f, eye);
    m[3] = 0.0f; m[7] = 0.0f; m[11] = 0.0f; m[15] = 1.0f;
}

void gui_math_mat4_mul(float* out, const float* a, const float* b)
{
    float tmp[16];
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) sum += a[k * 4 + row] * b[col * 4 + k];
            tmp[col * 4 + row] = sum;
        }
    }
    memcpy(out, tmp, sizeof(tmp));
}
