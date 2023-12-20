static char help[] = "Structure of a basic PETSc example\n\n";

#include <petsc.h>

int main(int argc,char **args) {

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));  // <-- always call


  PetscCall(PetscFinalize());
  return 0;
}
