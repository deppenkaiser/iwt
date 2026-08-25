#include "gui.h"
#include <ocl/ocl.h>
#include <stdint.h>
#include <string/string.h>

int main(int argc, char** argv)
{
	/* Cache-Verzeichnis im Binär-Verzeichnis anlegen, falls nicht vorhanden */
	{
		char exe_path[STRING_MAXLEN];
		string_get_exe_path(exe_path, sizeof(exe_path));

		char exe_path_copy[STRING_MAXLEN];
		string_copy(exe_path_copy, sizeof(exe_path_copy), exe_path);

		const char* base_dir = string_dirname_from_filepath(exe_path_copy);
		if (base_dir)
		{
			string_append_directory_to_path_and_create(base_dir, ".cache");
		}
	}

	struct iwt_gui_data data = {0};
	return gui_application_run("iwt.app", argc, argv, &data);
}
