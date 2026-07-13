#ifndef IWT_BATCH_H
#define IWT_BATCH_H

#include "iwt.h"

// Verarbeitet einen einzelnen Batch (alle Stufen)
// Gibt den max|Q| für diesen Batch zurück
double iwt_process_batch(
    iwt_system_t *sys,
    size_t batch_id
);

#endif // IWT_BATCH_H