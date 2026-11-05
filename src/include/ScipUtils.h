#ifndef SCIP_UTILS_H
#define SCIP_UTILS_H

#include <stdexcept>
#include <string>
#include <scip/scip.h>
#include <scip/scipdefplugins.h>

// turns a SCIP error code into an exception.
#define SCIP_CHECK(call)                                                              \
    do {                                                                              \
        SCIP_RETCODE scip_retcode_ = (call);                                          \
        if (scip_retcode_ != SCIP_OKAY)                                               \
            throw std::runtime_error("SCIP error " + std::to_string((int)scip_retcode_) \
                                     + " in " #call);                                 \
    } while (0)

// SCIP environment with the default plugins, silent and with a time limit (if positive).
inline SCIP* create_scip(double time_limit) {
    SCIP* scip = nullptr;
    SCIP_CHECK(SCIPcreate(&scip));
    SCIP_CHECK(SCIPincludeDefaultPlugins(scip));
    SCIP_CHECK(SCIPsetIntParam(scip, "display/verblevel", 0));
    if (time_limit > 0) SCIP_CHECK(SCIPsetRealParam(scip, "limits/time", time_limit));
    return scip;
}

#endif
