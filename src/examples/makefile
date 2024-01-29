include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex_VlasovPoisson: ex_VlasovPoisson.o
	-${CLINKER} -o ex_VlasovPoisson ex_VlasovPoisson.o  ${PETSC_LIB}
	${RM} ex_VlasovPoisson.o