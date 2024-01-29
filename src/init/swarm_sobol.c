#include <petscdm.h>
#include <petscviewer.h>
#include <petsc/private/dmswarmimpl.h>
/*
 Frances Y. Kuo

 Email: <f.kuo@unsw.edu.au>
 School of Mathematics and Statistics
 University of New South Wales
 Sydney NSW 2052, Australia
 
 Last updated: 21 October 2008

   You may incorporate this source code into your own program 
   provided that you
   1) acknowledge the copyright owner in your program and publication
   2) notify the copyright owner by email
   3) offer feedback regarding your experience with different direction numbers


 -----------------------------------------------------------------------------
 Licence pertaining to sobol.cc and the accompanying sets of direction numbers
 -----------------------------------------------------------------------------
 Copyright (c) 2008, Frances Y. Kuo and Stephen Joe
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
     * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
 
     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
 
     * Neither the names of the copyright holders nor the names of the
       University of New South Wales and the University of Waikato
       and its contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 -----------------------------------------------------------------------------
*/

PetscErrorCode DMSwarm_GenerateSobolPoints(PetscInt N, PetscInt D, PetscReal **points)
{
  unsigned       *C, *V, *X, L, i, j, k;
  unsigned        d_two[4] = {2,1,0,1};
  unsigned        d_three[5] = {3,2,1,1,3};
  unsigned        d_four[6] = {4,3,1,1,3,1};
  unsigned        d_five[6] = {5,3,2,1,1,1};
  unsigned        d_six[7] = {6,4,1,1,1,3,3};
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  if (D > 6) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"No support for dimension > 6.\n");
  L = (PetscInt)PetscCeilReal(PetscLogReal((PetscReal)N)/PetscLogReal(2.0)); 
  ierr = PetscCalloc1(N, &C);CHKERRQ(ierr);
  C[0] = 1;
  for (i=1;i<=N-1;i++) {
    C[i] = 1;
    PetscInt value = i;
    while (value & 1) {
      value >>= 1;
      C[i]++;
    }
  }
  ierr = PetscCalloc1(N*D, points);CHKERRQ(ierr);
  ierr = PetscCalloc1(L+1, &V);CHKERRQ(ierr);
  for (i=1;i<=L;i++) V[i] = 1 << (32-i);
  PetscCalloc1(N, &X);
  for (i=1;i<=N-1;i++) {
    X[i] = X[i-1] ^ V[C[i-1]];
    (*points)[i*D+0] = PetscAbsReal(((PetscReal)X[i]))/PetscPowReal(2.0,32);
  }
  PetscFree(V);
  PetscFree(X);
  for (j=1;j<=D-1;j++) {
    unsigned *d_number;
    unsigned *m, d, s, a;
    
    switch (j){
      case 1:
        d_number = d_two;
        break;
      case 2:
        d_number = d_three;
        break;
      case 3:
        d_number = d_four;
        break;
      case 4:
        d_number = d_five;
        break;
      case 5:
        d_number = d_six;
        break;
    }
    d = d_number[0];
    s = d_number[1];
    a = d_number[2];
    ierr = PetscCalloc1(s+1, &m);CHKERRQ(ierr);
    for (i=1;i<=s;i++) {m[i] = d_number[2+i];PetscPrintf(PETSC_COMM_WORLD, "%i m_i\n", m[i]);}
    ierr = PetscCalloc1(L+1, &V);CHKERRQ(ierr);
    if (L <= s) {
      for (i=1;i<=L;i++) V[i] = m[i] << (32-i); 
    }
    else {
      for (i=1;i<=s;i++) V[i] = m[i] << (32-i); 
      for (i=s+1;i<=L;i++) {
	      V[i] = V[i-s] ^ (V[i-s] >> s); 
	      for (k=1;k<=s-1;k++) V[i] ^= (((a >> (s-1-k)) & 1) * V[i-k]);
      }
    }
    ierr = PetscCalloc1(N, &X);CHKERRQ(ierr);
    for (i=1;i<=N-1;i++) {
      X[i] = X[i-1] ^ V[C[i-1]];
      (*points)[(i*D)+j] = PetscAbsReal((PetscReal)X[i])/PetscPowReal(2.0,32);
    }
    ierr = PetscFree(m);CHKERRQ(ierr);
    ierr = PetscFree(V);CHKERRQ(ierr);
    ierr = PetscFree(X);CHKERRQ(ierr);
  }
  ierr = PetscFree(C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMSwarmInitializePointCoordinatesQuietStart - Initialize the particles in DMSwarm to a quiet maxwellian using low discrepency sobol sequences
   in the spatial dimensions (2X, 3X) and uniform sampling in 1X. Speed is sampled from the inverse error function and spatial coordinates are used to
   convert it into a 1V, 2V, or 3V vector. The maxwellian will be homogenous only local to its cell, particularly in cases where a density function
   is used to determine the number of particles per cell in a uniform weighting scheme. 

   Note: The user must define the field "velocity" on swarm creation and field register and assign it the size of the dimension. 

   Non-collective

   Input parameters:
+  dm  - The DMSwarm
.  Np  - The number of particles in each cell, or NULL if supplying x
.  x   - Function pointer for determining N_p, may be NULL in which case spatial density will be homogeneous
-  ctx - application context

   Level: beginner

.seealso: DMSwarmInitializePointCoordinatesRandom()
@*/

PetscErrorCode DMSwarmInitializePointCoordinatesQuietStart(DM sw, PetscInt Np, void(*x)(), void* ctx){
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if(!x & !Np) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must provide particle per cell (Np) or particle density function(x).\n")
  ierr = DMGetDimension(sw, &dim);
  switch(dim){
    case 1:
      ierr = DMSwarmInitializePointCoordinatesQuietStart_1D_Private(sw, Np, x, ctx);CHKERRQ(ierr);
      break;
    case 2:
      ierr = DMSwarmInitializePointCoordinatesQuietStart_2D_Private(sw, Np, x, ctx);CHKERRQ(ierr);
      break;
    case 3:
      ierr = DMSwarmInitializePointCoordinatesQuietStart_3D_Private(sw, Np, x, ctx);CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}
