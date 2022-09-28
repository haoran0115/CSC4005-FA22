#include <stdio.h>
#include <mpi.h>

int main(){
    // mpi initialize
    MPI_Init(NULL, NULL);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // main program
    printf("Hello Wrold from rank %d\n", rank);

    // mpi finalize
    MPI_Finalize();
    return 0;
}
