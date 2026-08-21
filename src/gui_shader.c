#include "gui_shader.h"
#include <stdio.h>

GLuint gui_shader_compile(GLenum type, const char* source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint status = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        char log[512] = {0};
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
        fprintf(stderr, "IWT: Shader-Fehler: %s\n", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

GLuint gui_shader_create_points_program(void)
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

    GLuint vs = gui_shader_compile(GL_VERTEX_SHADER, vertex_source);
    GLuint fs = gui_shader_compile(GL_FRAGMENT_SHADER, fragment_source);
    if (vs == 0 || fs == 0) return 0;

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glBindAttribLocation(program, 0, "pos");
    glBindAttribLocation(program, 1, "color");
    glLinkProgram(program);

    GLint status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
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
