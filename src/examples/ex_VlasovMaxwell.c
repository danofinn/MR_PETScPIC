static char help[] = "Landau Damping test using Vlasov-Poisson equations\n";

/*
  To run the code with particles sinusoidally perturbed in x space use the test "pp_poisson_bsi_1d_4" or "pp_poisson_bsi_2d_4"
  According to Lukas, good damping results come at ~16k particles per cell

  To visualize the efield use

    -monitor_efield

  To visualize the swarm distribution use

    -ts_monitor_hg_swarm

  To visualize the particles, we can use

    -ts_monitor_sp_swarm -ts_monitor_sp_swarm_retain 0 -ts_monitor_sp_swarm_phase 1 -draw_size 500,500

*/
#include <petscts.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscfe.h>
#include <petscds.h>
#include <petscbag.h>
#include <petscdraw.h>
#include <petsc/private/dmpleximpl.h>  /* For norm and dot */
#include <petsc/private/petscfeimpl.h> /* For interpolation */
#include "petscdm.h"
#include "petscdmlabel.h"

const char *EMTypes[] = {"maxwell", "none", "EMType", "EM_", NULL};
typedef enum {
  EM_NONE,
  EM_MAXWELL
} EMType;

typedef enum {
  V0,
  X0,
  T0,
  M0,
  Q0,
  PHI0,
  POISSON,
  VLASOV,
  SIGMA,
  NUM_CONSTANTS
} ConstantType;
typedef struct {
  PetscScalar v0; /* Velocity scale, often the thermal velocity */
  PetscScalar t0; /* Time scale */
  PetscScalar x0; /* Space scale */
  PetscScalar m0; /* Mass scale */
  PetscScalar q0; /* Charge scale */
  PetscScalar kb;
  PetscScalar epsi0;
  PetscScalar phi0;          /* Potential scale */
  PetscScalar poissonNumber; /* Non-Dimensional Poisson Number */
  PetscScalar vlasovNumber;  /* Non-Dimensional Vlasov Number */
  PetscReal   sigma;         /* Nondimensional charge per length in x */
} Parameter;

typedef struct {
  PetscBag    bag;            /* Problem parameters */
  PetscBool   error;          /* Flag for printing the error */
  PetscBool   periodic;          /* Use periodic boundaries */
  PetscBool   fake_1D;           /* Run simulation in 2D but zeroing second dimension */
  PetscBool   perturbed_weights; /* Uniformly sample x,v space with gaussian weights */
  PetscInt    ostep; /* print the energy at each ostep time steps */
  PetscInt    numParticles;
  PetscReal   timeScale;              /* Nondimensionalizing time scale */
  PetscReal   charges[2];             /* The charges of each species */
  PetscReal   masses[2];              /* The masses of each species */
  PetscReal   thermal_energy[2];      /* Thermal Energy (used to get other constants)*/
  PetscReal   cosine_coefficients[2]; /*(alpha, k)*/
  PetscReal   totalWeight;
  PetscReal   stepSize;
  PetscInt    steps;
  PetscReal   initVel;
  EMType      em; /* Type of electrostatic model */
  SNES        snes;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  PetscInt d                      = 2;
  options->error                  = PETSC_FALSE;
  options->periodic               = PETSC_FALSE;
  options->fake_1D                = PETSC_FALSE;
  options->perturbed_weights      = PETSC_FALSE;
  options->ostep                  = 100;
  options->timeScale              = 2.0e-14;
  options->charges[0]             = -1.0;
  options->charges[1]             = 1.0;
  options->masses[0]              = 1.0;
  options->masses[1]              = 1000.0;
  options->thermal_energy[0]      = 1.0;
  options->thermal_energy[1]      = 1.0;
  options->cosine_coefficients[0] = 0.01;
  options->cosine_coefficients[1] = 0.5;
  options->initVel                = 1;
  options->totalWeight            = 1.0;
  options->em                     = EM_MAXWELL;
  options->numParticles           = 32768;

  PetscOptionsBegin(comm, "", "Central Orbit Options", "DMSWARM");
  PetscCall(PetscOptionsBool("-error", "Flag to print the error", "ex9.c", options->error, &options->error, NULL));
  PetscCall(PetscOptionsBool("-periodic", "Flag to use periodic particle boundaries", "ex9.c", options->periodic, &options->periodic, NULL));
  PetscCall(PetscOptionsBool("-fake_1D", "Flag to run a 1D simulation (but really in 2D)", "ex9.c", options->fake_1D, &options->fake_1D, NULL));
  PetscCall(PetscOptionsBool("-perturbed_weights", "Flag to run uniform sampling with perturbed weights", "ex9.c", options->perturbed_weights, &options->perturbed_weights, NULL));
  PetscCall(PetscOptionsInt("-output_step", "Number of time steps between output", "ex9.c", options->ostep, &options->ostep, NULL));
  PetscCall(PetscOptionsReal("-timeScale", "Nondimensionalizing time scale", "ex9.c", options->timeScale, &options->timeScale, NULL));
  PetscCall(PetscOptionsReal("-initial_velocity", "Initial velocity of perturbed particle", "ex9.c", options->initVel, &options->initVel, NULL));
  PetscCall(PetscOptionsReal("-total_weight", "Total weight of all particles", "ex9.c", options->totalWeight, &options->totalWeight, NULL));
  PetscCall(PetscOptionsRealArray("-cosine_coefficients", "Amplitude and frequency of cosine equation used in initialization", "ex9.c", options->cosine_coefficients, &d, NULL));
  PetscCall(PetscOptionsRealArray("-charges", "Species charges", "ex9.c", options->charges, &d, NULL));
  PetscCall(PetscOptionsEnum("-em_type", "Type of electrostatic solver", "ex9.c", EMTypes, (PetscEnum)options->em, (PetscEnum *)&options->em, NULL));
  PetscOptionsEnd();

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupContext(DM dm, DM sw, AppCtx *user)
{
  PetscFunctionBeginUser;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyContext(AppCtx *user)
{
  PetscFunctionBeginUser;
  
  PetscCall(PetscBagDestroy(&user->bag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupParameters(MPI_Comm comm, AppCtx *ctx)
{
  PetscBag   bag;
  Parameter *p;

  PetscFunctionBeginUser;
  /* setup PETSc parameter bag */
  PetscCall(PetscBagGetData(ctx->bag, (void **)&p));
  PetscCall(PetscBagSetName(ctx->bag, "par", "Vlasov-Poisson Parameters"));
  bag = ctx->bag;
  PetscCall(PetscBagRegisterScalar(bag, &p->v0, 1.0, "v0", "Velocity scale, m/s"));
  PetscCall(PetscBagRegisterScalar(bag, &p->t0, 1.0, "t0", "Time scale, s"));
  PetscCall(PetscBagRegisterScalar(bag, &p->x0, 1.0, "x0", "Space scale, m"));
  PetscCall(PetscBagRegisterScalar(bag, &p->v0, 1.0, "phi0", "Potential scale, kg*m^2/A*s^3"));
  PetscCall(PetscBagRegisterScalar(bag, &p->q0, 1.0, "q0", "Charge Scale, A*s"));
  PetscCall(PetscBagRegisterScalar(bag, &p->m0, 1.0, "m0", "Mass Scale, kg"));
  PetscCall(PetscBagRegisterScalar(bag, &p->epsi0, 1.0, "epsi0", "Permittivity of Free Space, kg"));
  PetscCall(PetscBagRegisterScalar(bag, &p->kb, 1.0, "kb", "Boltzmann Constant, m^2 kg/s^2 K^1"));

  PetscCall(PetscBagRegisterScalar(bag, &p->sigma, 1.0, "sigma", "Charge per unit area, C/m^3"));
  PetscCall(PetscBagRegisterScalar(bag, &p->poissonNumber, 1.0, "poissonNumber", "Non-Dimensional Poisson Number"));
  PetscCall(PetscBagRegisterScalar(bag, &p->vlasovNumber, 1.0, "vlasovNumber", "Non-Dimensional Vlasov Number"));
  PetscCall(PetscBagSetFromOptions(bag));
  {
    PetscViewer       viewer;
    PetscViewerFormat format;
    PetscBool         flg;

    PetscCall(PetscOptionsGetViewer(comm, NULL, NULL, "-param_view", &viewer, &format, &flg));
    if (flg) {
      PetscCall(PetscViewerPushFormat(viewer, format));
      PetscCall(PetscBagView(bag, viewer));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerPopFormat(viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = 0.0;
  return PETSC_SUCCESS;
}

/* Faraday Equation
  f_0 = \frac{1}{c^2} \frac{\partial E}{\partiakl t} - \nabla \times B
*/
static void f0_E(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  PetscReal curl[3] = {0,0,0}, c = 1;
  
  // curl[0] = u_x[2 * dim + 1] - u_x[1 * dim + 2];
  // curl[1] = u_x[0 * dim + 2] - u_x[2 * dim + 0];
  // curl[2] = u_x[1 * dim + 0] - u_x[0 * dim + 1];
  curl[0] = u_x[uOff_x[1] + 2 * dim + 1] - u_x[uOff_x[1] + 1 * dim + 2];
  curl[1] = u_x[uOff_x[1] + 0 * dim + 2] - u_x[uOff_x[1] + 2 * dim + 0];
  curl[2] = u_x[uOff_x[1] + 1 * dim + 0] - u_x[uOff_x[1] + 0 * dim + 1];
  for (d = 0; d < dim; ++d) f0[d] += 1/(c*c) * u_t[d * dim + d] - curl[d];
}

/* Ampere Equation
  f_0 = \frac{\partial B}{\partiakl t} + \nabla \times E
*/
static void f0_B(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  PetscReal curl[3] = {0,0,0};
  
  // curl[0] = u_x[2 * dim + 1] - u_x[1 * dim + 2];
  // curl[1] = u_x[0 * dim + 2] - u_x[2 * dim + 0];
  // curl[2] = u_x[1 * dim + 0] - u_x[0 * dim + 1];
  curl[0] = u_x[uOff_x[0] + 2 * dim + 1] - u_x[uOff_x[0] + 1 * dim + 2];
  curl[1] = u_x[uOff_x[0] + 0 * dim + 2] - u_x[uOff_x[0] + 2 * dim + 0];
  curl[2] = u_x[uOff_x[0] + 1 * dim + 0] - u_x[uOff_x[0] + 0 * dim + 1];
  for (d = 0; d < dim; ++d) f0[d] += u_t[uOff[0] + d * dim + d] + curl[d];
}

static PetscErrorCode CreateFEM(DM dm, AppCtx *user)
{
  PetscFE   feE, feB;
  PetscDS   ds;
  PetscBool simplex;
  PetscInt  dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  if (user->em == EM_MAXWELL) {
    DMLabel        label;
    const PetscInt id = 1;

    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, dim, simplex, "efield_", PETSC_DETERMINE, &feE));
    PetscCall(PetscObjectSetName((PetscObject)feE, "efield"));
    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, dim, simplex, "bfield_", PETSC_DETERMINE, &feB));
    PetscCall(PetscObjectSetName((PetscObject)feB, "bfield"));
    PetscCall(PetscFECopyQuadrature(feE, feB));
    PetscCall(DMSetField(dm, 0, NULL, (PetscObject)feE));
    PetscCall(DMSetField(dm, 1, NULL, (PetscObject)feB));
    PetscCall(DMCreateDS(dm));
    PetscCall(PetscFEDestroy(&feE));
    PetscCall(PetscFEDestroy(&feB));

    PetscCall(DMGetLabel(dm, "marker", &label));
    PetscCall(DMGetDS(dm, &ds));

    PetscCall(PetscDSSetResidual(ds, 0, f0_E, NULL));
    PetscCall(PetscDSSetResidual(ds, 1, f0_B, NULL));
    // PetscCall(PetscDSSetJacobian(ds, 0, 0, g0_qq, NULL, NULL, NULL));
    // PetscCall(PetscDSSetJacobian(ds, 0, 1, NULL, NULL, g2_qphi, NULL));
    // PetscCall(PetscDSSetJacobian(ds, 1, 0, NULL, g1_phiq, NULL, NULL));

    // PetscCall(DMClone(dm, &dmAux));

    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))zero, NULL, NULL, NULL));

  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscPDFPertubedConstant2D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = (1 + 0.01 * PetscCosReal(0.5 * x[0])) / (2 * PETSC_PI);
  p[1] = (1 + 0.01 * PetscCosReal(0.5 * x[1])) / (2 * PETSC_PI);
  return PETSC_SUCCESS;
}
PetscErrorCode PetscPDFPertubedConstant1D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = (1. + 0.01 * PetscCosReal(0.5 * x[0])) / (2 * PETSC_PI);
  return PETSC_SUCCESS;
}

PetscErrorCode PetscPDFCosine1D(const PetscReal x[], const PetscReal scale[], PetscReal p[])
{
  const PetscReal alpha = scale ? scale[0] : 0.0;
  const PetscReal k     = scale ? scale[1] : 1.;
  p[0]                  = (1 + alpha * PetscCosReal(k * x[0]));
  return PETSC_SUCCESS;
}

PetscErrorCode PetscPDFCosine2D(const PetscReal x[], const PetscReal scale[], PetscReal p[])
{
  const PetscReal alpha = scale ? scale[0] : 0.;
  const PetscReal k     = scale ? scale[0] : 1.;
  p[0]                  = (1 + alpha * PetscCosReal(k * (x[0] + x[1])));
  return PETSC_SUCCESS;
}

static PetscErrorCode InitializeParticles_PerturbedWeights(DM sw, AppCtx *user)
{
  DM           vdm, dm;
  PetscScalar *weight;
  PetscReal   *x, *v, vmin[3], vmax[3], gmin[3], gmax[3], xi0[3];
  PetscInt    *N, Ns, dim, *cellid, *species, Np, cStart, cEnd, Npc, n;
  PetscInt     p, q, s, c, d, cv;
  PetscBool    flg;
  PetscMPIInt  size, rank;
  Parameter   *param;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sw), &size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sw), &rank));
  PetscOptionsBegin(PetscObjectComm((PetscObject)sw), "", "DMSwarm Options", "DMSWARM");
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(PetscOptionsInt("-dm_swarm_num_species", "The number of species", "DMSwarmSetNumSpecies", Ns, &Ns, &flg));
  if (flg) PetscCall(DMSwarmSetNumSpecies(sw, Ns));
  PetscCall(PetscCalloc1(Ns, &N));
  n = Ns;
  PetscCall(PetscOptionsIntArray("-dm_swarm_num_particles", "The target number of particles", "", N, &n, NULL));
  PetscOptionsEnd();

  Np = N[0];
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Np = %" PetscInt_FMT "\n", Np));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

  PetscCall(DMCreate(PETSC_COMM_WORLD, &vdm));
  PetscCall(DMSetType(vdm, DMPLEX));
  PetscCall(DMPlexSetOptionsPrefix(vdm, "v"));
  PetscCall(DMSetFromOptions(vdm));
  PetscCall(DMViewFromOptions(vdm, NULL, "-vdm_view"));

  PetscCall(DMGetBoundingBox(dm, gmin, gmax));
  PetscCall(PetscBagGetData(user->bag, (void **)&param));
  PetscCall(DMSwarmSetLocalSizes(sw, Np, 0));
  Npc = Np / (cEnd - cStart);
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
  for (c = 0, p = 0; c < cEnd - cStart; ++c) {
    for (s = 0; s < Ns; ++s) {
      for (q = 0; q < Npc; ++q, ++p) cellid[p] = c;
    }
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
  PetscCall(PetscFree(N));

  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));

  PetscCall(DMSwarmSortGetAccess(sw));
  PetscInt vStart, vEnd;
  PetscCall(DMPlexGetHeightStratum(vdm, 0, &vStart, &vEnd));
  PetscCall(DMGetBoundingBox(vdm, vmin, vmax));
  for (c = 0; c < cEnd - cStart; ++c) {
    const PetscInt cell = c + cStart;
    PetscInt      *pidx, Npc;
    PetscReal      centroid[3], volume;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
    PetscCall(DMPlexComputeCellGeometryFVM(dm, cell, &volume, centroid, NULL));
    for (q = 0; q < Npc; ++q) {
      const PetscInt p = pidx[q];

      for (d = 0; d < dim; ++d) {
        x[p * dim + d] = centroid[d];
        v[p * dim + d] = vmin[0] + (q + 0.5) * (vmax[0] - vmin[0]) / Npc;
        if (user->fake_1D && d > 0) v[p * dim + d] = 0;
      }
    }
    PetscCall(PetscFree(pidx));
  }
  PetscCall(DMGetCoordinatesLocalSetUp(vdm));

  /* Setup Quadrature for spatial and velocity weight calculations*/
  PetscQuadrature  quad_x;
  PetscInt         Nq_x;
  const PetscReal *wq_x, *xq_x;
  PetscReal       *xq_x_extended;
  PetscReal        weightsum = 0., totalcellweight = 0., *weight_x, *weight_v;
  PetscReal        scale[2] = {user->cosine_coefficients[0], user->cosine_coefficients[1]};

  PetscCall(PetscCalloc2(cEnd - cStart, &weight_x, Np, &weight_v));
  if (user->fake_1D) PetscCall(PetscDTGaussTensorQuadrature(1, 1, 5, -1.0, 1.0, &quad_x));
  else PetscCall(PetscDTGaussTensorQuadrature(dim, 1, 5, -1.0, 1.0, &quad_x));
  PetscCall(PetscQuadratureGetData(quad_x, NULL, NULL, &Nq_x, &xq_x, &wq_x));
  if (user->fake_1D) {
    PetscCall(PetscCalloc1(Nq_x * dim, &xq_x_extended));
    for (PetscInt i = 0; i < Nq_x; ++i) xq_x_extended[i * dim] = xq_x[i];
  }
  /* Integrate the density function to get the weights of particles in each cell */
  for (d = 0; d < dim; ++d) xi0[d] = -1.0;
  for (c = cStart; c < cEnd; ++c) {
    PetscReal          v0_x[3], J_x[9], invJ_x[9], detJ_x, xr_x[3], den_x;
    PetscInt          *pidx, Npc, q;
    PetscInt           Ncx;
    const PetscScalar *array_x;
    PetscScalar       *coords_x = NULL;
    PetscBool          isDGx;
    weight_x[c] = 0.;

    PetscCall(DMPlexGetCellCoordinates(dm, c, &isDGx, &Ncx, &array_x, &coords_x));
    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0_x, J_x, invJ_x, &detJ_x));
    for (q = 0; q < Nq_x; ++q) {
      /*Transform quadrature points from ref space to real space (0,12.5664)*/
      if (user->fake_1D) CoordinatesRefToReal(dim, dim, xi0, v0_x, J_x, &xq_x_extended[q * dim], xr_x);
      else CoordinatesRefToReal(dim, dim, xi0, v0_x, J_x, &xq_x[q * dim], xr_x);

      /*Transform quadrature points from real space to ideal real space (0, 2PI/k)*/
      if (user->fake_1D) {
        PetscCall(PetscPDFCosine1D(xr_x, scale, &den_x));
        detJ_x = J_x[0];
      } else PetscCall(PetscPDFCosine2D(xr_x, scale, &den_x));
      /*We have to transform the quadrature weights as well*/
      weight_x[c] += den_x * (wq_x[q] * detJ_x);
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "c:%" PetscInt_FMT " [x_a,x_b] = %1.15f,%1.15f -> cell weight = %1.15f\n", c, (double)PetscRealPart(coords_x[0]), (double)PetscRealPart(coords_x[2]), (double)weight_x[c]));
    totalcellweight += weight_x[c];
    PetscCheck(Npc / size == vEnd - vStart, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of particles %" PetscInt_FMT " in cell (rank %d/%d) != %" PetscInt_FMT " number of velocity vertices", Npc, rank, size, vEnd - vStart);

    /* Set weights to be gaussian in velocity cells (using exact solution) */
    for (cv = 0; cv < vEnd - vStart; ++cv) {
      PetscInt           Nc;
      const PetscScalar *array_v;
      PetscScalar       *coords_v = NULL;
      PetscBool          isDG;
      PetscCall(DMPlexGetCellCoordinates(vdm, cv, &isDG, &Nc, &array_v, &coords_v));

      const PetscInt p = pidx[cv];

      weight_v[p] = 0.5 * (PetscErfReal(coords_v[1] / PetscSqrtReal(2.)) - PetscErfReal(coords_v[0] / PetscSqrtReal(2.)));

      weight[p] = user->totalWeight * weight_v[p] * weight_x[c];
      weightsum += weight[p];

      PetscCall(DMPlexRestoreCellCoordinates(vdm, cv, &isDG, &Nc, &array_v, &coords_v));
    }
    PetscCall(DMPlexRestoreCellCoordinates(dm, c, &isDGx, &Ncx, &array_x, &coords_x));
    PetscCall(PetscFree(pidx));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "particle weight sum = %1.10f cell weight sum = %1.10f\n", (double)totalcellweight, (double)weightsum));
  if (user->fake_1D) PetscCall(PetscFree(xq_x_extended));
  PetscCall(PetscFree2(weight_x, weight_v));
  PetscCall(PetscQuadratureDestroy(&quad_x));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMDestroy(&vdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeConstants(DM sw, AppCtx *user)
{
  DM         dm;
  PetscInt  *species;
  PetscReal *weight, totalCharge = 0., totalWeight = 0., gmin[3], gmax[3];
  PetscInt   Np, p, dim;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMGetBoundingBox(dm, gmin, gmax));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
  for (p = 0; p < Np; ++p) {
    totalWeight += weight[p];
    totalCharge += user->charges[species[p]] * weight[p];
  }
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
  {
    Parameter *param;
    PetscReal  Area;

    PetscCall(PetscBagGetData(user->bag, (void **)&param));
    switch (dim) {
    case 1:
      Area = (gmax[0] - gmin[0]);
      break;
    case 2:
      if (user->fake_1D) {
        Area = (gmax[0] - gmin[0]);
      } else {
        Area = (gmax[0] - gmin[0]) * (gmax[1] - gmin[1]);
      }
      break;
    case 3:
      Area = (gmax[0] - gmin[0]) * (gmax[1] - gmin[1]) * (gmax[2] - gmin[2]);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Dimension %" PetscInt_FMT " not supported", dim);
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dim = %" PetscInt_FMT "\ttotalWeight = %f, user->charges[species[p]] = %f\ttotalCharge = %f, Total Area = %f\n", dim, (double)totalWeight, (double)user->charges[0], (double)totalCharge, (double)Area));
    param->sigma = PetscAbsReal(totalCharge / (Area));

    PetscCall(PetscPrintf(PETSC_COMM_SELF, "sigma: %g\n", (double)param->sigma));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "(x0,v0,t0,m0,q0,phi0): (%e, %e, %e, %e, %e, %e) - (P, V) = (%e, %e)\n", (double)param->x0, (double)param->v0, (double)param->t0, (double)param->m0, (double)param->q0, (double)param->phi0, (double)param->poissonNumber,
                          (double)param->vlasovNumber));
  }
  /* Setup Constants */
  {
    PetscDS    ds;
    Parameter *param;
    PetscCall(PetscBagGetData(user->bag, (void **)&param));
    PetscScalar constants[NUM_CONSTANTS];
    constants[SIGMA]   = param->sigma;
    constants[V0]      = param->v0;
    constants[T0]      = param->t0;
    constants[X0]      = param->x0;
    constants[M0]      = param->m0;
    constants[Q0]      = param->q0;
    constants[PHI0]    = param->phi0;
    constants[POISSON] = param->poissonNumber;
    constants[VLASOV]  = param->vlasovNumber;
    PetscCall(DMGetDS(dm, &ds));
    PetscCall(PetscDSSetConstants(ds, NUM_CONSTANTS, constants));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeVelocites_Fake1D(DM sw, AppCtx *user)
{
  DM         dm;
  PetscReal *v;
  PetscInt  *species, cStart, cEnd;
  PetscInt   dim, Np, p;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscRandom rnd;
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)sw), &rnd));
  PetscCall(PetscRandomSetInterval(rnd, 0, 1.));
  PetscCall(PetscRandomSetFromOptions(rnd));

  for (p = 0; p < Np; ++p) {
    PetscReal a[3] = {0., 0., 0.}, vel[3] = {0., 0., 0.};

    PetscCall(PetscRandomGetValueReal(rnd, &a[0]));
    if (user->perturbed_weights) {
      PetscCall(PetscPDFSampleConstant1D(a, NULL, vel));
    } else {
      PetscCall(PetscPDFSampleGaussian1D(a, NULL, vel));
    }
    v[p * dim] = vel[0];
  }
  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSwarm(DM dm, AppCtx *user, DM *sw)
{
  PetscReal v0[2] = {1., 0.};
  PetscInt  dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "velocity", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "species", 1, PETSC_INT));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "initCoordinates", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "initVelocity", dim, PETSC_REAL));
  // PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "E_field", dim, PETSC_REAL));
  // PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "currentDensity", 1, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));

  if (user->perturbed_weights) {
    PetscCall(InitializeParticles_PerturbedWeights(*sw, user));
  } else {
    PetscCall(DMSwarmComputeLocalSizeFromOptions(*sw));
    PetscCall(DMSwarmInitializeCoordinates(*sw));
    if (user->fake_1D) {
      PetscCall(InitializeVelocites_Fake1D(*sw, user));
    } else {
      PetscCall(DMSwarmInitializeVelocitiesFromOptions(*sw, v0));
    }
  }
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSetApplicationContext(*sw, user));
  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  {
    Vec gc, gc0, gv, gv0;

    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, "initCoordinates", &gc0));
    PetscCall(VecCopy(gc, gc0));
    PetscCall(VecViewFromOptions(gc, NULL, "-ic_x_view"));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, "initCoordinates", &gc0));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, "velocity", &gv));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, "initVelocity", &gv0));
    PetscCall(VecCopy(gv, gv0));
    PetscCall(VecViewFromOptions(gv, NULL, "-ic_v_view"));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, "velocity", &gv));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, "initVelocity", &gv0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles_Maxwell(SNES snes, DM sw, PetscReal E[], PetscReal B[])
{
  AppCtx         *user;
  DM              dm, bfield_dm;
  KSP             ksp;
  IS              bfield_IS;
  PetscFE         fe;
  PetscFEGeom     feGeometry;
  PetscQuadrature q;
  Mat             M_p, M;
  Vec             wq, v, Mq_wq, Mf_Jz, Jz0, locEB;
  PetscReal      *coords, *Jz;
  PetscInt        dim, d, cStart, cEnd, c, Np, fields = 1;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMGetApplicationContext(sw, &user));

  /*ORDER:
  1. Create Mass matrices M_p and M_f
    - M_f => DMPlex
    - M_p => Swarm
  
  2. Solve M_f f = M_p w_p for f (finite element weights)
    - M_p^T * (w_p)
    - KSPSolve -> M_f f = M_p w_p
      - f = M_f^-1 M_p w_p
  
  3. Create current density vector J_z = \int f(x,v,t) v dv = f v 

  4. Setup TS object 

  5. TSSolve 

  6. Parse E and B fields and output to RHSFunction

  */
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMCreateSubDM(dm, 1, &fields, &bfield_IS, &bfield_dm));

  /* Create Mass Matrices (M_p & M_f) */
  PetscCall(DMCreateMassMatrix(sw, bfield_dm, &M_p));
  PetscCall(DMCreateMassMatrix(bfield_dm, bfield_dm, &M));
  PetscCall(MatViewFromOptions(M_p, NULL, "-mp_view"));
  PetscCall(MatViewFromOptions(M, NULL, "-m_view"));

  /* Create weights vector wq */
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &wq));
  PetscCall(PetscObjectSetName((PetscObject)wq, "particle weight"));
  PetscCall(VecViewFromOptions(wq, NULL, "-weights_view"));

  /* Create velocity vector v */
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &v));
  PetscCall(PetscObjectSetName((PetscObject)v, "particle velocity"));
  PetscCall(VecViewFromOptions(v, NULL, "-v_view"));

  /* Create vector to store Mq_wq = M_p*w_q */
  PetscCall(DMGetGlobalVector(bfield_dm, &Mq_wq));
  PetscCall(PetscObjectSetName((PetscObject)Mq_wq, "Mpwq"));

  /* Set Mq_wq = M_p*w_q */
  PetscCall(MatMultTranspose(M_p, wq, Mq_wq));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &wq));

  /* Get projection LHS: M_f Jz = M_p*w_q*v_p */
  PetscCall(DMGetGlobalVector(bfield_dm, &Mf_Jz));
  PetscCall(PetscObjectSetName((PetscObject)Mf_Jz, "Mf_Jz"));
  PetscCall(VecPointwiseMult(Mf_Jz,Mq_wq,v));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &v));

  /* Create local current density vector Jz0 */
  PetscCall(DMGetGlobalVector(bfield_dm, &Jz0));
  PetscCall(PetscObjectSetName((PetscObject)Jz0, "Current density (Jz)")); 

  /* KSPSolve for FEM weights (f) */
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "em_proj"));
  PetscCall(KSPSetOperators(ksp, M, M));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, Mf_Jz, Jz0));

  // /* Integral over reference element is size 1.  Reference element area is 4.  Scale rho0 by 1/4 because the basis function is 1/4 */
  PetscCall(VecScale(Jz0, 0.25));
  PetscCall(VecViewFromOptions(Jz0, NULL, "-Jz0_view"));
  PetscCall(VecViewFromOptions(Mf_Jz, NULL, "-MfJz_view"));

  PetscCall(DMRestoreGlobalVector(bfield_dm, &Mq_wq));
  PetscCall(DMRestoreGlobalVector(bfield_dm, &Mf_Jz));
  PetscCall(DMRestoreGlobalVector(bfield_dm, &Jz0));
  PetscCall(MatDestroy(&M_p));
  PetscCall(MatDestroy(&M));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&bfield_dm));
  PetscCall(ISDestroy(&bfield_IS));

  /* Create Auxiliary field for current density */
  DM        coordDM, dmAux;
  PetscBool simplex;
  PetscFE   feAux;
  PetscDS   ds;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, "aux_", PETSC_DETERMINE, &feAux));
  PetscCall(PetscObjectSetName((PetscObject)feAux, "aux_"));
  PetscCall(PetscFECopyQuadrature(fe, feAux));
  PetscCall(DMGetCoordinateDM(dm, &coordDM));
  PetscCall(DMClone(dm, &dmAux));
  PetscCall(DMSetCoordinateDM(dmAux, coordDM));
  PetscCall(DMSetField(dmAux, 0, NULL, (PetscObject)feAux));
  PetscCall(DMCreateDS(dmAux));
  /* Put auxiliary vector (Jz) in DM for f0 function - current density*/
  PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, Jz0));
  PetscCall(DMDestroy(&dmAux));

  /* TSSolve for E and B field */
  TS  ts_fem;
  Vec EB_sol;
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts_fem));
  PetscCall(TSSetProblemType(ts_fem, TS_NONLINEAR));
  PetscCall(TSSetDM(ts_fem, dm));
  PetscCall(TSSetApplicationContext(ts_fem, &user));
  PetscCall(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &user));
  PetscCall(DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &user));
  PetscCall(DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &user));
  PetscCall(TSSetMaxTime(ts_fem, 0.1));
  PetscCall(TSSetTimeStep(ts_fem, 0.00001));
  PetscCall(TSSetMaxSteps(ts_fem, 100));
  PetscCall(TSSetExactFinalTime(ts_fem, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetOptionsPrefix(ts_fem,"emfem_"));
  PetscCall(TSSetFromOptions(ts_fem));
  // PetscCall(TSComputeInitialCondition(ts_fem,InitializeFields));
  // PetscCall(CreateSolution(ts_fem));
  // PetscCall(TSGetSolution(ts_fem, &EB_sol));
  // PetscCall(TSComputeInitialCondition(ts_fem, EB_sol));
  PetscCall(TSSolve(ts_fem, EB_sol));
  PetscCall(VecViewFromOptions(EB_sol, NULL, "-EM_sol_view"));
  PetscCall(TSDestroy(&ts_fem));

  PetscCall(DMGetLocalVector(dm, &locEB));
  PetscCall(DMGlobalToLocalBegin(dm, EB_sol, INSERT_VALUES, locEB));
  PetscCall(DMGlobalToLocalEnd(dm, EB_sol, INSERT_VALUES, locEB));
  PetscCall(DMRestoreGlobalVector(dm, &EB_sol));

  /*  */
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(PetscFEGetQuadrature(fe, &q));
  PetscCall(PetscFECreateCellGeometry(fe, q, &feGeometry));
  for (c = cStart; c < cEnd; ++c) {
    PetscTabulation tab;
    PetscScalar    *clEB = NULL;
    PetscReal      *pcoord, *refcoord;
    PetscInt       *points;
    PetscInt        Ncp, cp;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Ncp, &points));
    // PetscCall(DMGetWorkArray(dm, Ncp * dim, MPIU_REAL, &pcoord));
    // PetscCall(DMGetWorkArray(dm, Ncp * dim, MPIU_REAL, &refcoord));
    for (cp = 0; cp < Ncp; ++cp)
      // for (d = 0; d < dim; ++d) pcoord[cp * dim + d] = coords[points[cp] * dim + d];
    // PetscCall(DMPlexCoordinatesToReference(dm, c, Ncp, pcoord, refcoord));
    // PetscCall(PetscFECreateTabulation(fe, 1, Ncp, refcoord, 1, &tab));
    // PetscCall(DMPlexComputeCellGeometryFEM(dm, c, q, feGeometry.v, feGeometry.J, feGeometry.invJ, feGeometry.detJ));
    // PetscCall(DMPlexVecGetClosure(dm, NULL, locEB, c, NULL, &clEB));

    for (cp = 0; cp < Ncp; ++cp) {
      const PetscInt p = points[cp];

      for (d = 0; d < dim; ++d) {
        E[p * dim + d] = 0.;
        B[p * dim + d] = 0.;
      }
      // PetscCall(PetscFEInterpolateAtPoints_Static(fe, tab, clEB, &feGeometry, cp, &E[p * dim]));
      // PetscCall(PetscFEPushforward(fe, &feGeometry, 1, &E[p * dim]));
      // for (d = 0; d < dim; ++d) {
      //   // E[p * dim + d] *= -2.0;
      //   if (user->fake_1D && d > 0) {
      //     E[p * dim + d] = 0;
      //     B[p * dim + d] = 0;
      //   }
      // }
    }
    // PetscCall(DMPlexVecRestoreClosure(dm, NULL, locEB, c, NULL, &clPhi));
    // PetscCall(DMRestoreWorkArray(dm, Ncp * dim, MPIU_REAL, &pcoord));
    // PetscCall(DMRestoreWorkArray(dm, Ncp * dim, MPIU_REAL, &refcoord));
    // PetscCall(PetscTabulationDestroy(&tab));
    PetscCall(PetscFree(points));
  }
  PetscCall(PetscFEDestroyCellGeometry(fe, &feGeometry));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMRestoreLocalVector(dm, &locEB));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles(SNES snes, DM sw, PetscReal E[], PetscReal B[])
{
  AppCtx  *ctx;
  PetscInt dim, Np;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(sw, DM_CLASSID, 2);
  PetscAssertPointer(E, 3);
  PetscAssertPointer(B, 4);
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMGetApplicationContext(sw, &ctx));
  PetscCall(PetscArrayzero(E, Np * dim));
  PetscCall(PetscArrayzero(B, Np * dim));

  switch (ctx->em) {
  case EM_MAXWELL:
    PetscCall(ComputeFieldAtParticles_Maxwell(snes, sw, E, B));
    break;
  case EM_NONE:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No solver for electrostatic model %s", EMTypes[ctx->em]);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec G, void *ctx)
{
  DM                 sw;
  SNES               snes = ((AppCtx *)ctx)->snes;
  const PetscReal   *coords, *vel;
  const PetscScalar *u;
  PetscScalar       *g;
  PetscReal         *E, *B, m_p = 1., q_p = -1., vxB[3] = {0., 0., 0.};
  PetscInt           dim, d, Np, p;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  // PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  // PetscCall(DMSwarmGetField(sw, "B_field", NULL, NULL, (void **)&B));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(G, &g));

  PetscCall(ComputeFieldAtParticles_Maxwell(snes, sw, E, B));
  /* v x B = |  i   j   k  | = | v_1 v_2 | i - | v_0 v_2 | j + | v_0 v_1 | k
             | v_0 v_1 v_2 |   | B_1 B_2 |     | B_0 B_2 |     | B_0 B_1 |
             | B_0 B_1 B_2 |

           = (v_1 B_2 - v_2 B_1) i - (v_0 B_2 - v_2 B_0) j + (v_0 B_1 - v_1 B_0) k */
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    vxB[0] = vel[p*dim + 1] * B[p*dim + 2] - vel[p*dim + 1] * B[p*dim + 2];
    vxB[1] = - vel[p*dim + 0] * B[p*dim + 2] + vel[p*dim + 2] * B[p*dim + 0];
    vxB[2] = vel[p*dim + 0] * B[p*dim + 1] - vel[p*dim + 1] * B[p*dim + 0];
    for (d = 0; d < dim; ++d) {
      g[(p * 2 + 0) * dim + d] = u[(p * 2 + 1) * dim + d];
      g[(p * 2 + 1) * dim + d] = q_p * (E[p * dim + d] - vxB[d]) / m_p;
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  // PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  // PetscCall(DMSwarmRestoreField(sw, "B_field", NULL, NULL, (void **)&B));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(G, &g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* J_{ij} = dF_i/dx_j
   J_p = (  0   1)
         (-w^2  0)
   TODO Now there is another term with w^2 from the electric field. I think we will need to invert the operator.
        Perhaps we can approximate the Jacobian using only the cellwise P-P gradient from Coulomb
*/
static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec U, Mat J, Mat P, void *ctx)
{
  DM               sw;
  const PetscReal *coords, *vel;
  PetscInt         dim, d, Np, p, rStart;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(MatGetOwnershipRange(P, &rStart, NULL));
  PetscCall(DMSwarmGetField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  for (p = 0; p < Np; ++p) {
    PetscScalar vals[4] = {0., 1., -1., 0.};

    for (d = 0; d < dim; ++d) {
      const PetscInt rows[2] = {(p * 2 + 0) * dim + d + rStart, (p * 2 + 1) * dim + d + rStart};
      PetscCall(MatSetValues(P, 2, rows, 2, rows, vals, INSERT_VALUES));
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunctionX(TS ts, PetscReal t, Vec V, Vec Xres, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  DM                 sw;
  const PetscScalar *v;
  PetscScalar       *xres;
  PetscInt           Np, p, d, dim;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetLocalSize(Xres, &Np));
  PetscCall(VecGetArrayRead(V, &v));
  PetscCall(VecGetArray(Xres, &xres));
  Np /= dim;
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      xres[p * dim + d] = v[p * dim + d];
      if (user->fake_1D && d > 0) xres[p * dim + d] = 0;
    }
  }
  PetscCall(VecRestoreArrayRead(V, &v));
  PetscCall(VecRestoreArray(Xres, &xres));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunctionV(TS ts, PetscReal t, Vec X, Vec Vres, void *ctx)
{
  DM                 sw;
  AppCtx            *user = (AppCtx *)ctx;
  SNES               snes = ((AppCtx *)ctx)->snes;
  const PetscScalar *x;
  const PetscReal   *coords, *vel;
  PetscReal         *E, *B, m_p, q_p, vxB[3] = {0., 0., 0.};
  PetscScalar       *vres;
  PetscInt           Np, p, dim, d;
  Parameter         *param;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  // PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(PetscBagGetData(user->bag, (void **)&param));
  m_p = user->masses[0] * param->m0;
  q_p = user->charges[0] * param->q0;
  PetscCall(VecGetLocalSize(Vres, &Np));
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(Vres, &vres));
  PetscCheck(dim == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Dimension must be 2");

  PetscCall(ComputeFieldAtParticles(snes, sw, E, B));

  /* v x B = |  i   j   k  | = | v_1 v_2 | i - | v_0 v_2 | j + | v_0 v_1 | k
             | v_0 v_1 v_2 |   | B_1 B_2 |     | B_0 B_2 |     | B_0 B_1 |
             | B_0 B_1 B_2 |

           = (v_1 B_2 - v_2 B_1) i - (v_0 B_2 - v_2 B_0) j + (v_0 B_1 - v_1 B_0) k */
  Np /= dim;
  for (p = 0; p < Np; ++p) {
    vxB[0] = vel[p*dim + 1] * B[p*dim + 2] - vel[p*dim + 1] * B[p*dim + 2];
    vxB[1] = - vel[p*dim + 0] * B[p*dim + 2] + vel[p*dim + 2] * B[p*dim + 0];
    vxB[2] = vel[p*dim + 0] * B[p*dim + 1] - vel[p*dim + 1] * B[p*dim + 0];
    for (d = 0; d < dim; ++d) {
      vres[p * dim + d] = q_p * (E[p * dim + d] - vxB[d]) / m_p;
      if (user->fake_1D && d > 0) vres[p * dim + d] = 0.;
    }
  }
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(Vres, &vres));
  PetscCall(DMSwarmRestoreField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  // PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSolution(TS ts)
{
  DM       sw;
  Vec      u;
  PetscInt dim, Np;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetBlockSize(u, dim));
  PetscCall(VecSetSizes(u, 2 * Np * dim, PETSC_DECIDE));
  PetscCall(VecSetUp(u));
  PetscCall(TSSetSolution(ts, u));
  PetscCall(VecDestroy(&u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetProblem(TS ts)
{
  AppCtx *user;
  DM      sw;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, (void **)&user));
  // Define unified system for (X, V)
  {
    Mat      J;
    PetscInt dim, Np;

    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(DMSwarmGetLocalSize(sw, &Np));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
    PetscCall(MatSetSizes(J, 2 * Np * dim, 2 * Np * dim, PETSC_DECIDE, PETSC_DECIDE));
    PetscCall(MatSetBlockSize(J, 2 * dim));
    PetscCall(MatSetFromOptions(J));
    PetscCall(MatSetUp(J));
    PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, user));
    PetscCall(TSSetRHSJacobian(ts, J, J, RHSJacobian, user));
    PetscCall(MatDestroy(&J));
  }
  /* Define split system for X and V */
  {
    Vec             u;
    IS              isx, isv, istmp;
    const PetscInt *idx;
    PetscInt        dim, Np, rstart;

    PetscCall(TSGetSolution(ts, &u));
    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(DMSwarmGetLocalSize(sw, &Np));
    PetscCall(VecGetOwnershipRange(u, &rstart, NULL));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, Np, (rstart / dim) + 0, 2, &istmp));
    PetscCall(ISGetIndices(istmp, &idx));
    PetscCall(ISCreateBlock(PETSC_COMM_WORLD, dim, Np, idx, PETSC_COPY_VALUES, &isx));
    PetscCall(ISRestoreIndices(istmp, &idx));
    PetscCall(ISDestroy(&istmp));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, Np, (rstart / dim) + 1, 2, &istmp));
    PetscCall(ISGetIndices(istmp, &idx));
    PetscCall(ISCreateBlock(PETSC_COMM_WORLD, dim, Np, idx, PETSC_COPY_VALUES, &isv));
    PetscCall(ISRestoreIndices(istmp, &idx));
    PetscCall(ISDestroy(&istmp));
    PetscCall(TSRHSSplitSetIS(ts, "position", isx));
    PetscCall(TSRHSSplitSetIS(ts, "momentum", isv));
    PetscCall(ISDestroy(&isx));
    PetscCall(ISDestroy(&isv));
    PetscCall(TSRHSSplitSetRHSFunction(ts, "position", NULL, RHSFunctionX, user));
    PetscCall(TSRHSSplitSetRHSFunction(ts, "momentum", NULL, RHSFunctionV, user));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmTSRedistribute(TS ts)
{
  DM        sw;
  Vec       u;
  PetscReal t, maxt, dt;
  PetscInt  n, maxn;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(TSGetTime(ts, &t));
  PetscCall(TSGetMaxTime(ts, &maxt));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetStepNumber(ts, &n));
  PetscCall(TSGetMaxSteps(ts, &maxn));

  PetscCall(TSReset(ts));
  PetscCall(TSSetDM(ts, sw));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetTime(ts, t));
  PetscCall(TSSetMaxTime(ts, maxt));
  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSSetStepNumber(ts, n));
  PetscCall(TSSetMaxSteps(ts, maxn));

  PetscCall(CreateSolution(ts));
  PetscCall(SetProblem(ts));
  PetscCall(TSGetSolution(ts, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  InitializeSolveAndSwarm - Set the solution values to the swarm coordinates and velocities, and also possibly set the initial values.

  Input Parameters:
+ ts         - The TS
- useInitial - Flag to also set the initial conditions to the current coordinates and velocities and setup the problem

  Output Parameter:
. u - The initialized solution vector

  Level: advanced

.seealso: InitializeSolve()
*/
static PetscErrorCode InitializeSolveAndSwarm(TS ts, PetscBool useInitial)
{
  DM       sw;
  Vec      u, gc, gv, gc0, gv0;
  IS       isx, isv;
  PetscInt dim;
  AppCtx  *user;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, &user));
  PetscCall(DMGetDimension(sw, &dim));
  if (useInitial) {
    PetscReal v0[2] = {1., 0.};
    if (user->perturbed_weights) {
      PetscCall(InitializeParticles_PerturbedWeights(sw, user));
    } else {
      PetscCall(DMSwarmComputeLocalSizeFromOptions(sw));
      PetscCall(DMSwarmInitializeCoordinates(sw));
      if (user->fake_1D) {
        PetscCall(InitializeVelocites_Fake1D(sw, user));
      } else {
        PetscCall(DMSwarmInitializeVelocitiesFromOptions(sw, v0));
      }
    }
    PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
    PetscCall(DMSwarmTSRedistribute(ts));
  }
  PetscCall(TSGetSolution(ts, &u));
  PetscCall(TSRHSSplitGetIS(ts, "position", &isx));
  PetscCall(TSRHSSplitGetIS(ts, "momentum", &isv));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "initCoordinates", &gc0));
  if (useInitial) PetscCall(VecCopy(gc, gc0));
  PetscCall(VecISCopy(u, isx, SCATTER_FORWARD, gc));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "initCoordinates", &gc0));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &gv));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "initVelocity", &gv0));
  if (useInitial) PetscCall(VecCopy(gv, gv0));
  PetscCall(VecISCopy(u, isv, SCATTER_FORWARD, gv));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &gv));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "initVelocity", &gv0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeSolve(TS ts, Vec u)
{
  PetscFunctionBegin;
  PetscCall(TSSetSolution(ts, u));
  PetscCall(InitializeSolveAndSwarm(ts, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MigrateParticles(TS ts)
{
  DM sw;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMViewFromOptions(sw, NULL, "-migrate_view_pre"));
  {
    Vec u, gc, gv;
    IS  isx, isv;

    PetscCall(TSGetSolution(ts, &u));
    PetscCall(TSRHSSplitGetIS(ts, "position", &isx));
    PetscCall(TSRHSSplitGetIS(ts, "momentum", &isv));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
    PetscCall(VecISCopy(u, isx, SCATTER_REVERSE, gc));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &gv));
    PetscCall(VecISCopy(u, isv, SCATTER_REVERSE, gv));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &gv));
  }
  PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
  PetscCall(DMSwarmTSRedistribute(ts));
  PetscCall(InitializeSolveAndSwarm(ts, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MigrateParticles_Periodic(TS ts)
{
  DM       sw, dm;
  PetscInt dim;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMViewFromOptions(sw, NULL, "-migrate_view_pre"));
  {
    Vec        u, position, momentum, gc, gv;
    IS         isx, isv;
    PetscReal *pos, *mom, *x, *v;
    PetscReal  lower_bound[3], upper_bound[3];
    PetscInt   p, d, Np;

    PetscCall(TSGetSolution(ts, &u));
    PetscCall(DMSwarmGetLocalSize(sw, &Np));
    PetscCall(DMSwarmGetCellDM(sw, &dm));
    PetscCall(DMGetBoundingBox(dm, lower_bound, upper_bound));
    PetscCall(TSRHSSplitGetIS(ts, "position", &isx));
    PetscCall(TSRHSSplitGetIS(ts, "momentum", &isv));
    PetscCall(VecGetSubVector(u, isx, &position));
    PetscCall(VecGetSubVector(u, isv, &momentum));
    PetscCall(VecGetArray(position, &pos));
    PetscCall(VecGetArray(momentum, &mom));

    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
    PetscCall(VecISCopy(u, isx, SCATTER_REVERSE, gc));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &gv));
    PetscCall(VecISCopy(u, isv, SCATTER_REVERSE, gv));

    PetscCall(VecGetArray(gc, &x));
    PetscCall(VecGetArray(gv, &v));
    for (p = 0; p < Np; ++p) {
      for (d = 0; d < dim; ++d) {
        if (pos[p * dim + d] < lower_bound[d]) {
          x[p * dim + d] = pos[p * dim + d] + (upper_bound[d] - lower_bound[d]);
        } else if (pos[p * dim + d] > upper_bound[d]) {
          x[p * dim + d] = pos[p * dim + d] - (upper_bound[d] - lower_bound[d]);
        } else {
          x[p * dim + d] = pos[p * dim + d];
        }
        PetscCheck(x[p * dim + d] >= lower_bound[d] && x[p * dim + d] <= upper_bound[d], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "p: %" PetscInt_FMT "x[%" PetscInt_FMT "] %g", p, d, (double)x[p * dim + d]);
        v[p * dim + d] = mom[p * dim + d];
      }
    }
    PetscCall(VecRestoreArray(gc, &x));
    PetscCall(VecRestoreArray(gv, &v));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &gv));

    PetscCall(VecRestoreArray(position, &pos));
    PetscCall(VecRestoreArray(momentum, &mom));
    PetscCall(VecRestoreSubVector(u, isx, &position));
    PetscCall(VecRestoreSubVector(u, isv, &momentum));
  }
  PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
  PetscCall(DMSwarmTSRedistribute(ts));
  PetscCall(InitializeSolveAndSwarm(ts, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM     dm, sw;
  TS     ts;
  Vec    u;
  AppCtx user;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(PetscBagCreate(PETSC_COMM_SELF, sizeof(Parameter), &user.bag));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  // PetscCall(CreatePoisson(dm, &user));
  PetscCall(CreateSwarm(dm, &user, &sw));
  PetscCall(SetupParameters(PETSC_COMM_WORLD, &user));
  PetscCall(InitializeConstants(sw, &user));
  PetscCall(DMSetApplicationContext(sw, &user));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetDM(ts, sw));
  PetscCall(TSSetMaxTime(ts, 0.1));
  PetscCall(TSSetTimeStep(ts, 0.00001));
  PetscCall(TSSetMaxSteps(ts, 100));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));

  PetscCall(TSSetFromOptions(ts));
  PetscReal dt;
  PetscInt  maxn;
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetMaxSteps(ts, &maxn));
  user.steps    = maxn;
  user.stepSize = dt;
  PetscCall(SetupContext(dm, sw, &user));

  const char *fieldNames[] = {DMSwarmPICField_coor, "velocity"};
  PetscCall(DMSwarmVectorDefineFields(sw, sizeof(fieldNames)/sizeof(char*), fieldNames));
  PetscCall(TSSetComputeInitialCondition(ts, InitializeSolve));
  // PetscCall(TSSetComputeExactError(ts, ComputeError));
  if (user.periodic) {
    PetscCall(TSSetPostStep(ts, MigrateParticles_Periodic));
  } else {
    PetscCall(TSSetPostStep(ts, MigrateParticles));
  }
  PetscCall(CreateSolution(ts));
  PetscCall(TSGetSolution(ts, &u));
  PetscCall(TSComputeInitialCondition(ts, u));

  PetscCall(TSSolve(ts, NULL));

  PetscCall(SNESDestroy(&user.snes));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(DestroyContext(&user));
  PetscCall(PetscFinalize());
  return 0;
}

/* OPTIONS:

  build:
    requires: double !complex
  # Recommend -draw_size 500,500

  test_1: (2D No EM)
  -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 5,5 -dm_plex_box_lower 0,0 -dm_plex_box_upper 12.5664,12.5664 \
  -perturbed_weights -dm_swarm_velocity_function constant -dm_swarm_num_particles 625 \
  -dm_swarm_num_species 1 -dm_plex_box_bd periodic,periodic -periodic \
  -vdm_plex_dim 2 -vdm_plex_box_lower -10,-10 -vdm_plex_box_upper 10,10 -vdm_plex_simplex 0 -vdm_plex_box_faces 5,5 \
  -em_type none -ts_type basicsymplectic -ts_basicsymplectic_type 1 -ts_max_time 1.0 -ts_max_steps 10 -ts_dt 0.01\
  -ts_monitor_sp_swarm -ts_monitor_sp_swarm_retain 0 -ts_monitor_sp_swarm_phase 0 \
  -periodic -cosine_coefficients 0.01,0.5 -charges -1.0,1.0 -total_weight 1.0 \
  -dm_view -output_step 1


-ts_type basicsymplectic -ts_basicsymplectic_type 1 -em_type mixed\
             -ksp_rtol 1e-10\
             -em_ksp_type preonly\
             -em_ksp_error_if_not_converged\
             -em_snes_error_if_not_converged\
             -em_pc_type fieldsplit\
             -em_fieldsplit_field_pc_type lu \
             -em_fieldsplit_potential_pc_type svd\
             -em_pc_fieldsplit_type schur\
             -em_pc_fieldsplit_schur_fact_type full\
             -em_pc_fieldsplit_schur_precondition full\
             -potential_petscspace_degree 0 \
             -potential_petscdualspace_lagrange_use_moments \
             -potential_petscdualspace_lagrange_moment_order 2 \
             -field_petscspace_degree 2\
             -field_petscfe_default_quadrature_order 1\
             -field_petscspace_type sum \
             -field_petscspace_variables 2 \
             -field_petscspace_components 2 \
             -field_petscspace_sum_spaces 2 \
             -field_petscspace_sum_concatenate true \
             -field_sumcomp_0_petscspace_variables 2 \
             -field_sumcomp_0_petscspace_type tensor \
             -field_sumcomp_0_petscspace_tensor_spaces 2 \
             -field_sumcomp_0_petscspace_tensor_uniform false \
             -field_sumcomp_0_tensorcomp_0_petscspace_degree 1 \
             -field_sumcomp_0_tensorcomp_1_petscspace_degree 0 \
             -field_sumcomp_1_petscspace_variables 2 \
             -field_sumcomp_1_petscspace_type tensor \
             -field_sumcomp_1_petscspace_tensor_spaces 2 \
             -field_sumcomp_1_petscspace_tensor_uniform false \
             -field_sumcomp_1_tensorcomp_0_petscspace_degree 0 \
             -field_sumcomp_1_tensorcomp_1_petscspace_degree 1 \
             -field_petscdualspace_form_degree -1 \
             -field_petscdualspace_order 1 \
             -field_petscdualspace_lagrange_trimmed true \
             -ksp_gmres_restart 500

TEST*/
