CC=g++
CFLAGS=-O3
LDFLAGS=-static

PROGRAM=parse_isa

all: ${PROGRAM}

parse_isa: parse_isa.cpp
	${CC} ${CFLAGS} $@.cpp -o $@ ${LDFLAGS}

clean:
	rm ${PROGRAM}
