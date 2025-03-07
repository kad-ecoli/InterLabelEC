CC=g++
CFLAGS=-O3
LDFLAGS=-static

PROGRAM=parse_isa ec2exp fasta2miss obo2Fcsv uniprot2ec uniprot2nonec

all: ${PROGRAM}

parse_isa: parse_isa.cpp
	${CC} ${CFLAGS} $@.cpp -o $@ ${LDFLAGS}

ec2exp: ec2exp.cpp StringTools.hpp
	${CC} ${CFLAGS} $@.cpp -o $@ ${LDFLAGS}

fasta2miss: fasta2miss.cpp
	${CC} ${CFLAGS} $@.cpp -o $@ ${LDFLAGS}

obo2Fcsv: obo2Fcsv.cpp StringTools.hpp
	${CC} ${CFLAGS} $@.cpp -o $@ ${LDFLAGS}

uniprot2ec: uniprot2ec.cpp StringTools.hpp
	${CC} ${CFLAGS} $@.cpp -o $@ ${LDFLAGS}

uniprot2nonec: uniprot2nonec.cpp StringTools.hpp
	${CC} ${CFLAGS} $@.cpp -o $@ ${LDFLAGS}

clean:
	rm ${PROGRAM}
