CC          = nvcc
CLINKER     = nvcc

CFLAGS      = -O2 -g 
LIBS        = -lm
DEPEND= makedepend

SRC     = Collider.cu Particle.cu sampleMSD.cu sampleRDF.cu sampleVAF.cu langevin_thermostat.cu system.cpp main.cu readdat.cpp
OBJS    = Collider.o Particle.o sampleMSD.o sampleRDF.o sampleVAF.o langevin_thermostat.o system.o main.o readdat.o
EXECS   = simul.exe

default: md_nve

all: md_nve

md_nve: $(OBJS)
	$(CLINKER) $(OPTFLAGS) -o $(EXECS) $(OBJS) $(LIBS)

clean:
	/bin/rm -f *.o *~ $(EXECS)

# Rule to compile CUDA files
%.o: %.cu
	$(CC) $(CFLAGS) -c $<

Collider.o: system.h cuda_opt_constants.h
Particle.o: system.h
sampleMSD.o: Particle.h write_settings.h
sampleRDF.o: Particle.h
sampleVAF.o: system.h Particle.h write_settings.h
langevin_thermostat.o: system.h cuda_opt_constants.h Particle.h
system.o: system.h
readdat.o: system.h
