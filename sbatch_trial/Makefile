main:
	mpic++ -o main main.cpp
clean:
	rm ./main
run:
	make clean
	make
	mpirun -np 4 ./main

