/*
*
* Studente: Petraglia Mariangela 0522500473
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h> 


/**
* Funzione che esegue il prodotto matrice vettore
*/
void prod_mat_vett(double w[], double *a, int ROWS, int COLS, double v[])
{
    int i, j;
    
    for(i=0;i<ROWS;i++)
    {
        w[i]=0;
        for(j=0;j<COLS;j++)
        { 
            w[i] += a[i*COLS+j]* v[j];
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

int nproc;              // Numero di processi totale
int me;                 // Il mio id
int n;                  // Dimensione della matrice
int local_n;            // Dimensione dei dati locali
int i,j;                    // Iteratori vari 

// Variabili di lavoro
double *A, *v, *localA,*local_w, *w;
//Variabili per il tempo
double T_inizio,T_fine,T_max;

/*Attiva MPI*/
MPI_Init(&argc, &argv);
/*Trova il numero totale dei processi*/
MPI_Comm_size (MPI_COMM_WORLD, &nproc);
/*Da ad ogni processo il proprio numero identificativo*/
MPI_Comm_rank (MPI_COMM_WORLD, &me);

// Se sono a radice
if(me == 0)
{   
    printf("*** MAT - VET I STRATEGIA *** \n"); 
    if (argc < 2) {
      		printf("Numero di parametri insufficiente!");
      		exit(0);;
    }
   	else{
     	n = atoi(argv[1]);
    }
    //printf("Inserire n: \n"); 
    //scanf("%d",&n); 
    // Porzione di dati da processare
    printf("n = %d \t. La matrice è %d*%d \n", n, n, n);
    local_n = n/nproc;  
    
    // Alloco spazio di memoria
    A = malloc(n * n * sizeof(double));
    v = malloc(sizeof(double)*n);
    w = malloc(sizeof(double)*n); 
    
    for (i=0;i<n;i++)
    {
        v[i]=i;  
        for(j=0;j<n;j++)
        {
            if (j==0)
           		A[i*n+j]= 1.0/(i+1)-1;
            else
                A[i*n+j]= 1.0/(i+1)-pow(1.0/2.0,j);   
        }  
    }
    //stampa la matrice solo nel caso sia al massimo 10*10 e il vettore che sia di 10 elem
    if(n<10){
        printf("Matrice A[] \n");
    	print_matrix_array(A,n,n);
        printf("\n"); 

    printf("Vettore v[] \n"); 
    for (i=0;i<n;i++){   
        printf("%5.3f\n", v[i]);
    }
    printf("\n");
    } 
    else{
        printf("Matrice A e vettore v troppo grandi per essere stampati a video \n");
    }

}   


// Spedisco n e local_n
MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);            
MPI_Bcast(&local_n,1,MPI_INT,0,MPI_COMM_WORLD);            

// Se sono un figlio alloco v 
if(me != 0)
    v = malloc(sizeof(double)*n);
    
MPI_Bcast(&v[0],n,MPI_DOUBLE,0,MPI_COMM_WORLD);            

// tutti allocano A locale e il vettore dei risultati
localA = malloc(local_n * n * sizeof(double));
local_w = malloc(local_n * sizeof(double));

// Adesso 0 invia a tutti un pezzo della matrice
int num = local_n*n;
MPI_Scatter(
    &A[0], num, MPI_DOUBLE,
    &localA[0], num, MPI_DOUBLE,
    0, MPI_COMM_WORLD);

// Scriviamo la matrice locale ricevuta se sono al massimo 30 elementi
if(num<30){
	printf("\n Local_A %d: \n", me); 
	print_matrix_array(localA,local_n,n);
    printf("\n");
}
printf("\n");

/* sincronizzazione dei processori del contesto MPI_COMM_WORLD*/
    
MPI_Barrier(MPI_COMM_WORLD);
 
T_inizio=MPI_Wtime(); //inizio del cronometro per il calcolo del tempo di inizio

// Effettuiamo i calcoli
prod_mat_vett(local_w,localA,local_n,n,v);

MPI_Barrier(MPI_COMM_WORLD); // sincronizzazione
T_fine=MPI_Wtime()-T_inizio; // calcolo del tempo di fine
    
// 0 raccoglie i risultati parziali
MPI_Gather(&local_w[0],local_n,MPI_DOUBLE,&w[0],local_n,MPI_DOUBLE,0,MPI_COMM_WORLD);

/* calcolo del tempo totale di esecuzione*/
MPI_Reduce(&T_fine,&T_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
/*0 stampa a video i risultati finali*/
if(me==0){  
    //    printf("\n Il risultato è w [] \n"); 
    //    for(i = 0; i < n; i++)
    //        printf("%5.3f ", w[i]);
    printf("\n");
    printf("\nProcessori impegnati: %d\n", nproc);
    printf("\nTempo calcolo locale: %lf\n", T_fine);
    printf("\nMPI_Reduce max time: %f\n",T_max);
}// end if


MPI_Finalize (); /* Disattiva MPI */
return 0;  
}