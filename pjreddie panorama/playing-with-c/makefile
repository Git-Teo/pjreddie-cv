# MAKEFILE 1
# CC=gcc
# CFLAGS=-I.
#
# hellomake: hellomake.o hellofunc.o
# 	$(CC) -o hellomake hellomake.o hellofunc.o
# MAKEFILE 2
# CC=gcc
# CFLAGS=-I.
# DEPS = hellomake.h
#
# %.o: %.c $(DEPS)
# 	$(CC) -c -o $@ $< $(CFLAGS)
#
# hellomake: hellomake.o hellofunc.o
# 	$(CC) -o hellomake hellomake.o hellofunc.o
# MAKEFILE 3
# CC=gcc
# CFLAGS=-I.
# DEPS = hellomake.h
# OBJ = hellomake.o hellofunc.o
#
# %.o: %.c $(DEPS)
# 	$(CC) -c -o $@ $< $(CFLAGS)
#
# hellomake: $(OBJ)
# 	$(CC) -o $@ $^ $(CFLAGS)
# MAKEFILE 4
# CC=gcc
# CFLAGS=-I.
# DEPS = hellomake.h
# OBJ = hellomake.o hellofunc.o
#
# %.o: %.c $(DEPS)
# 	$(CC) -c -o $@ $< $(CFLAGS)
#
# hellomake: $(OBJ)
# 	$(CC) -o $@ $^ $(CFLAGS)
# MAKEFILE 5
IDIR= ../include
CC= gcc
CFLAGS= -I$(IDIR)

ODIR= obj
LDIR= ../lib

LIBS= -lm

_DEPS= hellomake.h
DEPS= $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ= hellomake.o hellofunc.o
OBJ= $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

hellomake: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~