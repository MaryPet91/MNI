/*
*
* Studente: Petraglia Mariangela 0522500473
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NANOSECONDS_PER_SECOND 1E9


void prod_mat_vett(double w[], double *a, int rows, int cols, double v[])
{
    int i, j;
    
    for(i=0;i<rows;i++)
    {
        w[i]=0;
        for(j=0;j<cols;j++)
        { 
            w[i] += a[i*cols+j]* v[j];
        } 
    }    
}

void print_matrix_array(double *array, int rows, int columns) {
    int i;

    for (i = 0; i < rows * columns; ++i) {
        printf("%5.3f \t", array[i]);
        if (i % columns == columns - 1) {
            printf("\n\n");
        }
    }
}

int main(int argc, char **argv) {
    int i, j;
    int n;
    double *A, *v, *w;
    struct timespec fine,inizio;
    double tempoTot;

    if (argc < 2) {
        printf("Numero di parametri insufficiente!");
        exit(0);;
    }
    else{
        n = atoi(argv[1]);
    }
 
    //printf("Inserisci n: ");
    //scanf("%d", &n);
    printf("\n");

    // Alloco spazio di memoria
    A = malloc(n * n * sizeof(double));
    v = malloc(n * sizeof(double));

    for (i = 0; i < n; i++) {
        v[i] = i;
        for (j = 0; j < n; j++) {
            if (j == 0) {
                A[i*n+j] = 1.0/(i + 1) - 1;
            }
            else {
                A[i*n+j] = 1.0/(i+1) - pow(1.0/2.0, j); 
            }
        }
    }

   /* printf("Matrice A:\n");
    print_matrix_array(A, n, n);
    printf("Vettore v:\n");
    print_matrix_array(v, n, 1);
    printf("\n");*/

    w = malloc(n * sizeof(double)); 

    clock_gettime(CLOCK_REALTIME, &inizio);
    prod_mat_vett(w, A, n, n, v);
    clock_gettime(CLOCK_REALTIME, &fine);

    //printf("Risultato vettore w:\n");
    //print_matrix_array(w, n, 1);
    tempoTot = (fine.tv_sec - inizio.tv_sec) + (fine.tv_nsec - inizio.tv_nsec)/NANOSECONDS_PER_SECOND;
    printf("Tempo tot = %lf \n",tempoTot);

    return 0;
}