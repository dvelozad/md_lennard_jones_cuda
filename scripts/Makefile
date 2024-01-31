CC          = nvcc
CLINKER     = nvcc

CFLAGS      = #-O3 -g 
LIBS        = -lm
DEPEND= makedepend

SRC     = neighbor_list.cu Collider.cu Particle.cu sampleMSD.cu sampleRDF.cu sampleVAF.cu langevin_thermostat.cu system.cu main.cu readdat.cu write_output.cu
OBJS    = neighbor_list.o Collider.o Particle.o sampleMSD.o sampleRDF.o sampleVAF.o langevin_thermostat.o system.o main.o readdat.o write_output.o
EXECS   = simul.exe

default: md_nve

all: md_nve

md_nve: $(OBJS)
	$(CLINKER) $(OPTFLAGS) -o $(EXECS) $(OBJS) $(LIBS)

clean:
	/bin/rm -f *.o *~ $(EXECS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $<

neighbor_list.o: Particle.h system.h
Collider.o: system.h cuda_opt_constants.h
Particle.o: system.h
sampleMSD.o: Particle.h write_settings.h
sampleRDF.o: Particle.h
sampleVAF.o: system.h Particle.h write_settings.h
langevin_thermostat.o: system.h cuda_opt_constants.h Particle.h
system.o: system.h
readdat.o: system.h
