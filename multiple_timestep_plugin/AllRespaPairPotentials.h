//
// Created by nathan on 9/18/2021.
//

#ifndef MULTIPLE_TIMESTEP_PLUGIN_ALLRESPAPAIRPOTENTIALS_H
#define MULTIPLE_TIMESTEP_PLUGIN_ALLRESPAPAIRPOTENTIALS_H

#include "RespaPotentialPair.h"
#include "RespaPotentialPairDPDThermo.h"

#include <hoomd/md/EvaluatorPairLJ.h>
#include <hoomd/md/EvaluatorPairGauss.h>
#include <hoomd/md/EvaluatorPairYukawa.h>
#include <hoomd/md/EvaluatorPairEwald.h>
#include <hoomd/md/EvaluatorPairSLJ.h>
#include <hoomd/md/EvaluatorPairMorse.h>
#include <hoomd/md/EvaluatorPairDPDThermo.h>
#include <hoomd/md/PotentialPairDPDThermo.h>
#include <hoomd/md/EvaluatorPairMoliere.h>
#include <hoomd/md/EvaluatorPairZBL.h>
#include <hoomd/md/EvaluatorPairDPDLJThermo.h>
#include <hoomd/md/EvaluatorPairForceShiftedLJ.h>
#include <hoomd/md/EvaluatorPairMie.h>
#include <hoomd/md/EvaluatorPairReactionField.h>
#include <hoomd/md/EvaluatorPairBuckingham.h>
#include <hoomd/md/EvaluatorPairLJ1208.h>
#include <hoomd/md/EvaluatorPairDLVO.h>
#include <hoomd/md/EvaluatorPairFourier.h>

#ifdef ENABLE_CUDA
//#include "PotentialPairGPU.h"
//#include "PotentialPairDPDThermoGPU.h"
//#include "PotentialPairDPDThermoGPU.cuh"
//#include "AllDriverPotentialPairGPU.cuh"
#endif

/*! \file AllPairPotentials.h
    \brief Handy list of typedefs for all of the templated pair potentials in hoomd
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Pair potential force compute for lj forces
typedef RespaPotentialPair<EvaluatorPairLJ> RespaPotentialPairLJ;
//! Pair potential force compute for gaussian forces
typedef RespaPotentialPair<EvaluatorPairGauss> RespaPotentialPairGauss;
//! Pair potential force compute for slj forces
typedef RespaPotentialPair<EvaluatorPairSLJ> RespaPotentialPairSLJ;
//! Pair potential force compute for yukawa forces
typedef RespaPotentialPair<EvaluatorPairYukawa> RespaPotentialPairYukawa;
//! Pair potential force compute for ewald forces
typedef RespaPotentialPair<EvaluatorPairEwald> RespaPotentialPairEwald;
//! Pair potential force compute for morse forces
typedef RespaPotentialPair<EvaluatorPairMorse> RespaPotentialPairMorse;
//! Pair potential force compute for dpd conservative forces
typedef RespaPotentialPair<EvaluatorPairDPDThermo> RespaPotentialPairDPD;
//! Pair potential force compute for Moliere forces
typedef RespaPotentialPair<EvaluatorPairMoliere> RespaPotentialPairMoliere;
//! Pair potential force compute for ZBL forces
typedef RespaPotentialPair<EvaluatorPairZBL> RepsaPotentialPairZBL;
//! Pair potential force compute for dpd thermostat and conservative forces
//typedef RespaPotentialPairDPDThermo<EvaluatorPairDPDThermo> RespaPotentialPairDPDThermoDPD;
//! Pair potential force compute for dpdlj conservative forces (not intended to be used)
typedef RespaPotentialPair<EvaluatorPairDPDLJThermo> RespaPotentialPairDPDLJ;
//! Pair potential force compute for dpd thermostat and LJ conservative forces
//typedef RespaPotentialPairDPDThermo<EvaluatorPairDPDLJThermo> RespaPotentialPairDPDLJThermoDPD;
//! Pair potential force compute for force shifted LJ on the GPU
typedef RespaPotentialPair<EvaluatorPairForceShiftedLJ> RespaPotentialPairForceShiftedLJ;
//! Pair potential force compute for Mie potential
typedef RespaPotentialPair<EvaluatorPairMie> RespaPotentialPairMie;
//! Pair potential force compute for ReactionField potential
typedef RespaPotentialPair<EvaluatorPairReactionField> RespaPotentialPairReactionField;
//! Pair potential force compute for Buckingham forces
typedef RespaPotentialPair<EvaluatorPairBuckingham> RespaPotentialPairBuckingham;
//! Pair potential force compute for lj1208 forces
typedef RespaPotentialPair<EvaluatorPairLJ1208> RespaPotentialPairLJ1208;
//! Pair potential force compute for DLVO potential
typedef RespaPotentialPair<EvaluatorPairDLVO> RespaPotentialPairDLVO;
//! Pair potential force compute for Fourier potential
typedef RespaPotentialPair<EvaluatorPairFourier> RespaPotentialPairFourier;

#ifdef ENABLE_CUDA
//! Pair potential force compute for lj forces on the GPU
typedef RespaPotentialPairGPU< EvaluatorPairLJ, gpu_compute_ljtemp_forces > RespaPotentialPairLJGPU;
//! Pair potential force compute for gaussian forces on the GPU
typedef RespaPotentialPairGPU< EvaluatorPairGauss, gpu_compute_gauss_forces > RespaPotentialPairGaussGPU;
//! Pair potential force compute for slj forces on the GPU
typedef RespaPotentialPairGPU< EvaluatorPairSLJ, gpu_compute_slj_forces > RespaPotentialPairSLJGPU;
//! Pair potential force compute for yukawa forces on the GPU
typedef RespaPotentialPairGPU< EvaluatorPairYukawa, gpu_compute_yukawa_forces > RespaPotentialPairYukawaGPU;
//! Pair potential force compute for ewald forces on the GPU
typedef RespaPotentialPairGPU< EvaluatorPairEwald, gpu_compute_ewald_forces > RespaPotentialPairEwaldGPU;
//! Pair potential force compute for morse forces on the GPU
typedef RespaPotentialPairGPU< EvaluatorPairMorse, gpu_compute_morse_forces > RespaPotentialPairMorseGPU;
//! Pair potential force compute for dpd conservative forces on the GPU
typedef RespaPotentialPairGPU<EvaluatorPairDPDThermo, gpu_compute_dpdthermo_forces > RespaPotentialPairDPDGPU;
//! Pair potential force compute for Moliere forces on the GPU
typedef RespaPotentialPairGPU<EvaluatorPairMoliere, gpu_compute_moliere_forces > RespaPotentialPairMoliereGPU;
//! Pair potential force compute for ZBL forces on the GPU
typedef RespaPotentialPairGPU<EvaluatorPairZBL, gpu_compute_zbl_forces > RespaPotentialPairZBLGPU;
//! Pair potential force compute for dpd thermostat and conservative forces on the GPU
typedef RespaPotentialPairDPDThermoGPU<EvaluatorPairDPDThermo, gpu_compute_dpdthermodpd_forces > RespaPotentialPairDPDThermoDPDGPU;
//! Pair potential force compute for dpdlj conservative forces on the GPU (not intended to be used)
typedef RespaPotentialPairGPU<EvaluatorPairDPDLJThermo, gpu_compute_dpdljthermo_forces > RespaPotentialPairDPDLJGPU;
//! Pair potential force compute for dpd thermostat and LJ conservative forces on the GPU
typedef RespaPotentialPairDPDThermoGPU<EvaluatorPairDPDLJThermo, gpu_compute_dpdljthermodpd_forces > RespaPotentialPairDPDLJThermoDPDGPU;
//! Pair potential force compute for force shifted LJ on the GPU
typedef RespaPotentialPairGPU<EvaluatorPairForceShiftedLJ, gpu_compute_force_shifted_lj_forces> RespaPotentialPairForceShiftedLJGPU;
//! Pair potential force compute for Mie potential
typedef RespaPotentialPairGPU<EvaluatorPairMie, gpu_compute_mie_forces> RespaPotentialPairMieGPU;
//! Pair potential force compute for reaction field forces on the GPU
typedef RespaPotentialPairGPU< EvaluatorPairReactionField, gpu_compute_reaction_field_forces > RespaPotentialPairReactionFieldGPU;
//! Pair potential force compute for Buckingham forces on the GPU
typedef RespaPotentialPairGPU< EvaluatorPairBuckingham, gpu_compute_buckingham_forces > RespaPotentialPairBuckinghamGPU;
//! Pair potential force compute for lj1208 forces on the GPU
typedef RespaPotentialPairGPU< EvaluatorPairLJ1208, gpu_compute_lj1208_forces > RespaPotentialPairLJ1208GPU;
//! Pair potential force compute for DLVO forces on the GPU
typedef RespaPotentialPairGPU< EvaluatorPairDLVO, gpu_compute_dlvo_forces > RespaPotentialPairDLVOGPU;
//! Pair potential force compute for Fourier forces on the gpu
typedef RespaPotentialPairGPU<EvaluatorPairFourier, gpu_compute_fourier_forces> RespaPotentialPairFourierGPU;
#endif


#endif //MULTIPLE_TIMESTEP_PLUGIN_ALLRESPAPAIRPOTENTIALS_H
