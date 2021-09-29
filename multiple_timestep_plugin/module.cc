// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "MultipleTimestepIntegrator.h"
#include "RespaForceCompute.h"
#include "RespaPotentialPair.h"
#include "AllRespaPairPotentials.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

// specify the python module. Note that the name must explicitly match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
PYBIND11_MODULE(_multiple_timestep_plugin, m)
{
    export_MultipleTimestepIntegrator(m);
    export_RespaForceCompute(m);
    //Could it be that I need to call the export_NeighborList function here?

    export_RespaPotentialPair<RespaPotentialPairLJ>(m,"RespaPotentialPairLJ");/*
    export_RespaPotentialPair<RespaPotentialPairGauss>(m,"RespaPotentialPairGauss");
    export_RespaPotentialPair<RespaPotentialPairSLJ>(m,"RespaPotentialPairSLJ");
    export_RespaPotentialPair<RespaPotentialPairYukawa>(m,"RespaPotentialPairYukawa");
    export_RespaPotentialPair<RespaPotentialPairEwald>(m,"RespaPotentialPairEwald");
    export_RespaPotentialPair<RespaPotentialPairMorse>(m,"RespaPotentialPairMorse");
    export_RespaPotentialPair<RespaPotentialPairDPD>(m,"RespaPotentialPairDPD");
    export_RespaPotentialPair<RespaPotentialPairMoliere>(m,"RespaPotentialPairMoliere");
    export_RespaPotentialPair<RepsaPotentialPairZBL>(m,"RepsaPotentialPairZBL");
    export_RespaPotentialPairDPDThermo<RespaPotentialPairDPDThermoDPD, RespaPotentialPairDPD>(m, "RespaPotentialPairDPDThermoDPD");
    export_RespaPotentialPair<RespaPotentialPairDPDLJ>(m,"RespaPotentialPairDPDLJ");
    export_RespaPotentialPairDPDThermo<RespaPotentialPairDPDLJThermoDPD, RespaPotentialPairDPDLJ>(m, "RespaPotentialPairDPDLJThermoDPD");
    export_RespaPotentialPair<RespaPotentialPairForceShiftedLJ>(m,"RespaPotentialPairForceShiftedLJ");
    export_RespaPotentialPair<RespaPotentialPairMie>(m,"RespaPotentialPairMie");
    export_RespaPotentialPair<RespaPotentialPairReactionField>(m,"RespaPotentialPairReactionField");
    export_RespaPotentialPair<RespaPotentialPairBuckingham>(m,"RespaPotentialPairBuckingham");
    export_RespaPotentialPair<RespaPotentialPairLJ1208>(m,"RespaPotentialPairLJ1208");
    export_RespaPotentialPair<RespaPotentialPairDLVO>(m,"RespaPotentialPairDLVO");
    export_RespaPotentialPair<RespaPotentialPairFourier>(m,"RespaPotentialPairFourier");*/



#ifdef ENABLE_HIP
    export_MultipleTimestepGPU(m);
#endif
}
