#ifndef IWT_INIT_H
#define IWT_INIT_H

#include "iwt.h"

// Alloziert Speicher für I, K, Q, sumJ
bool iwt_allocate(iwt_system_t *sys);

// Initialisiert I und K mit Werten aus der Konfiguration
void iwt_initialize_system(iwt_system_t *sys);

// Gibt Speicher frei
void iwt_free_system(iwt_system_t *sys);

#endif // IWT_INIT_H