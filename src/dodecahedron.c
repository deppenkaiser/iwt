#include "dodecahedron.h"
#include <api/api.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
	double center[3];
	double scale;
	double rot[3][3];
} _dodeca_cell_t;

private
void _mat3_identity(double m[3][3])
{
	m[0][0] = 1.0;
	m[0][1] = 0.0;
	m[0][2] = 0.0;
	m[1][0] = 0.0;
	m[1][1] = 1.0;
	m[1][2] = 0.0;
	m[2][0] = 0.0;
	m[2][1] = 0.0;
	m[2][2] = 1.0;
}

private
void _mat3_mul(double out[3][3], const double a[3][3], const double b[3][3])
{
	double tmp[3][3];
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			double sum = 0.0;
			for (int k = 0; k < 3; k++)
			{
				sum += a[i][k] * b[k][j];
			}
			tmp[i][j] = sum;
		}
	}
	memcpy(out, tmp, sizeof(tmp));
}

private
void _mat3_vec_mul(double out[3], const double m[3][3], const double v[3])
{
	for (int i = 0; i < 3; i++)
	{
		out[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
	}
}

// Rodrigues-Rotationsformel: Drehung um `angle` (rad) um die Einheitsachse `axis`
private
void _mat3_from_axis_angle(double m[3][3], const double axis[3], double angle)
{
	double x = axis[0], y = axis[1], z = axis[2];
	double c = cos(angle);
	double s = sin(angle);
	double t = 1.0 - c;

	m[0][0] = t * x * x + c;
	m[0][1] = t * x * y - s * z;
	m[0][2] = t * x * z + s * y;
	m[1][0] = t * x * y + s * z;
	m[1][1] = t * y * y + c;
	m[1][2] = t * y * z - s * x;
	m[2][0] = t * x * z - s * y;
	m[2][1] = t * y * z + s * x;
	m[2][2] = t * z * z + c;
}

// 20 Dodekaeder-Ecken (roh, unnormiert) - exakt wie in dodekaeder.scad
private
void _base_vertices_raw(double v[20][3])
{
	double p = (1.0 + sqrt(5.0)) / 2.0;
	double raw[20][3] = {
		{1, 1, 1}, {1, 1, -1}, {1, -1, 1}, {1, -1, -1}, {-1, 1, 1}, {-1, 1, -1}, {-1, -1, 1}, {-1, -1, -1}, {p, 1.0 / p, 0}, {p, -1.0 / p, 0}, {-p, 1.0 / p, 0}, {-p, -1.0 / p, 0}, {0, p, 1.0 / p}, {0, p, -1.0 / p}, {0, -p, 1.0 / p}, {0, -p, -1.0 / p}, {1.0 / p, 0, p}, {1.0 / p, 0, -p}, {-1.0 / p, 0, p}, {-1.0 / p, 0, -p}};
	memcpy(v, raw, sizeof(raw));
}

// 12 Fuenfeck-Flaechen (Eckindizes) - exakt wie in dodekaeder.scad
static const int _base_faces[12][5] = {
	{0, 12, 4, 18, 16},
	{0, 12, 13, 1, 8},
	{0, 8, 9, 2, 16},
	{2, 14, 6, 18, 16},
	{12, 4, 10, 5, 13},
	{11, 10, 5, 19, 7},
	{13, 5, 19, 17, 1},
	{1, 8, 9, 3, 17},
	{11, 6, 18, 4, 10},
	{15, 14, 6, 11, 7},
	{15, 3, 17, 19, 7},
	{14, 2, 9, 3, 15}};

// 12 Flaechenzentren (normiert) = Ecken des dualen (eingepassten) Ikosaeders
private
void _base_face_centers(const double v[20][3], double c[12][3])
{
	for (int f = 0; f < 12; f++)
	{
		double sum[3] = {0.0, 0.0, 0.0};
		for (int k = 0; k < 5; k++)
		{
			int idx = _base_faces[f][k];
			sum[0] += v[idx][0];
			sum[1] += v[idx][1];
			sum[2] += v[idx][2];
		}
		double norm = sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
		c[f][0] = sum[0] / norm;
		c[f][1] = sum[1] / norm;
		c[f][2] = sum[2] / norm;
	}
}

// Fraktale Punkterzeugung (Breadth-First) ausgehend von einem beliebigen
// Wurzel-Zentrum `center` statt fest (0,0,0) - Kern von
// dodecahedron_generate_points() und dodecahedron_generate_multi_root_points()
private
bool _dodecahedron_generate_from_center(double* pos_x, double* pos_y, double* pos_z, size_t N, double R0, const double center[3])
{
	const double phi = (1.0 + sqrt(5.0)) / 2.0;
	const double s = 2.0 + phi;									  // Skalierungsfaktor
	const double psi_rad = 2.0 * (4.0 * atan(1.0)) * (2.0 - phi); // Goldener Winkel (rad)

	double base_v_raw[20][3];
	_base_vertices_raw(base_v_raw);

	// Alle Rohecken haben identische Norm sqrt(3) -> einheitliche Normierung
	double inv_norm = 1.0 / sqrt(3.0);
	double base_v[20][3];
	for (int i = 0; i < 20; i++)
	{
		base_v[i][0] = base_v_raw[i][0] * inv_norm;
		base_v[i][1] = base_v_raw[i][1] * inv_norm;
		base_v[i][2] = base_v_raw[i][2] * inv_norm;
	}

	double base_face_dir[12][3];
	_base_face_centers(base_v, base_face_dir);

	// Breadth-First-Warteschlange (dynamisch wachsend)
	size_t queue_cap = 64;
	_dodeca_cell_t* queue = malloc(queue_cap * sizeof(_dodeca_cell_t));
	if (!queue)
		return false;

	size_t queue_head = 0;
	size_t queue_tail = 0;

	queue[queue_tail].center[0] = center[0];
	queue[queue_tail].center[1] = center[1];
	queue[queue_tail].center[2] = center[2];
	queue[queue_tail].scale = R0;
	_mat3_identity(queue[queue_tail].rot);
	queue_tail++;

	size_t generated = 0;

	while (queue_head < queue_tail && generated < N)
	{
		_dodeca_cell_t cell = queue[queue_head];
		queue_head++;

		int vertices_written = 0;
		for (int i = 0; i < 20 && generated < N; i++)
		{
			double scaled[3] = {base_v[i][0] * cell.scale, base_v[i][1] * cell.scale, base_v[i][2] * cell.scale};
			double rotated[3];
			_mat3_vec_mul(rotated, cell.rot, scaled);

			pos_x[generated] = cell.center[0] + rotated[0];
			pos_y[generated] = cell.center[1] + rotated[1];
			pos_z[generated] = cell.center[2] + rotated[2];
			generated++;
			vertices_written++;
		}

		if (vertices_written == 20)
		{
			// Schale vollstaendig gefuellt -> 12 Kind-Dodekaeder an den
			// Flaechenzentren (= Ikosaeder-Ecken) erzeugen
			for (int f = 0; f < 12; f++)
			{
				if (queue_tail >= queue_cap)
				{
					queue_cap *= 2;
					_dodeca_cell_t* new_queue = realloc(queue, queue_cap * sizeof(_dodeca_cell_t));
					if (!new_queue)
					{
						free(queue);
						return false;
					}
					queue = new_queue;
				}

				double scaled_dir[3] = {
					base_face_dir[f][0] * cell.scale,
					base_face_dir[f][1] * cell.scale,
					base_face_dir[f][2] * cell.scale};
				double world_offset[3];
				_mat3_vec_mul(world_offset, cell.rot, scaled_dir);

				_dodeca_cell_t* child = &queue[queue_tail];
				child->center[0] = cell.center[0] + world_offset[0];
				child->center[1] = cell.center[1] + world_offset[1];
				child->center[2] = cell.center[2] + world_offset[2];
				child->scale = cell.scale / s;

				double r_local[3][3];
				_mat3_from_axis_angle(r_local, base_face_dir[f], psi_rad);
				_mat3_mul(child->rot, cell.rot, r_local);

				queue_tail++;
			}
		}
	}

	free(queue);
	return generated == N;
}

bool dodecahedron_generate_points(double* pos_x, double* pos_y, double* pos_z, size_t N, double R0)
{
	const double center[3] = {0.0, 0.0, 0.0};
	return _dodecahedron_generate_from_center(pos_x, pos_y, pos_z, N, R0, center);
}

bool dodecahedron_generate_multi_root_points(double* pos_x, double* pos_y, double* pos_z, size_t N, double R0, int nx, int ny, int nz, double spacing)
{
	int root_count = nx * ny * nz;
	if (root_count <= 0)
		return false;

	size_t base_n = N / (size_t) root_count;
	size_t remainder = N % (size_t) root_count;

	size_t offset = 0;
	int root_index = 0;
	bool all_ok = true;

	for (int ix = 0; ix < nx; ix++)
	{
		for (int iy = 0; iy < ny; iy++)
		{
			for (int iz = 0; iz < nz; iz++)
			{
				size_t n_this_root = base_n + ((size_t) root_index < remainder ? 1 : 0);

				double center[3] = {
					((double) ix - (nx - 1) / 2.0) * spacing,
					((double) iy - (ny - 1) / 2.0) * spacing,
					((double) iz - (nz - 1) / 2.0) * spacing};

				if (!_dodecahedron_generate_from_center(
						pos_x + offset, pos_y + offset, pos_z + offset,
						n_this_root, R0, center))
				{
					all_ok = false;
				}

				offset += n_this_root;
				root_index++;
			}
		}
	}

	return all_ok && (offset == N);
}
