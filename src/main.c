#include "gui.h"
#include <ocl/ocl.h>
#include <stdint.h>
#include <string/string.h>

int main(int argc, char** argv)
{
	struct iwt_gui_data data = {0};
	return gui_application_run("iwt.app", argc, argv, &data);
}
