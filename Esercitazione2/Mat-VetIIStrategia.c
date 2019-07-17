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
void prod_mat_vett(double w[], double *a, int rows, int cols, double v[]){
    int i, j;
    
    for(i=0;i<rows;i++){
        w[i]=0;
        for(j=0;j<cols;j++){ 
            w[i] += a[i*cols+j] * v[j]; 
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
int i,j;                // Iteratori vari 

// Variabili di lavoro
double *A, *v, *w, *local_A, *local_v, *local_w;
//Variabili per il tempo
double T_inizio,T_fine,T_max;

/*Attiva MPI*/
MPI_Init(&argc, &argv);
/*Trova il numero totale dei processi*/
MPI_Comm_size (MPI_COMM_WORLD, &nproc);
/*Da ad ogni processo il proprio numero identificativo*/
MPI_Comm_rank (MPI_COMM_WORLD, &me);

printf("Processo %d \n",me);

// Master
if(me == 0){
    printf("*** MAT - VET II STRATEGIA *** \n"); 
    //printf("Inserire n: \n"); 
    //scanf("%d",&n); 
    if (argc < 2) {
      		printf("Numero di parametri insufficiente!");
      		exit(0);;
    }
   	else{
     	n = atoi(argv[1]);
    }

    // Porzione di dati da processare
    local_n = n/nproc;  
    
    // Allocazione spazio di memoria
    A = malloc(sizeof *A * n * n);
    v = malloc(sizeof *v * n);
    w = malloc(sizeof *w * n); 
    
    //INIZIALIZIAMO LA MATRICE DIRETTAMENTE TRASPOSTA
    for (i=0;i<n;i++)
    {
        v[i]=i;  
        for(j=0;j<n;j++)
        {
            if (j==0)
                A[j*n+i]= 1.0/(i+1)-1;
            else
                A[j*n+i]= 1.0/(i+1)-pow(1.0/2.0,j); 
        }
    }

    //stampa la matrice solo nel caso sia al massimo 10*10
    if(n<10){
        printf("\n Matrice A[] \n");
        print_matrix_array(A,n,n);
    	printf("\n"); 

    	printf("\n Vettore v[] \n"); 
    	for (i=0;i<n;i++){   
        	printf("%5.3f\n", v[i]);
    	}
    	printf("\n");
    } 
    else
        printf("Matrice A e vettore v troppo grandi per essere stampati a video \n");
    
}//fine master


// Spedisco n e local_n
MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);            
MPI_Bcast(&local_n,1,MPI_INT,0,MPI_COMM_WORLD);            

//tutti i processi allocano local_v ù
//local_n è il numero di elementi del vettore v Che è uguale al numero di colonne che inviamo
local_v = malloc(sizeof *local_v * local_n); 
    
//scatter vettore per righe
MPI_Scatter(&v[0],local_n, MPI_DOUBLE, &local_v[0], local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

MPI_Barrier(MPI_COMM_WORLD);

if(n < 10){
    printf(" local v di %d :\n", me);
    for (i=0;i<local_n;i++)
        {   
           printf("%5.3f\n", local_v[i]);
        }
        printf("\n");
}

// tutti allocano A locale, A trasposta locale e il vettore w dei risultati
local_A = malloc(sizeof *local_A * local_n * n);
local_w = malloc(sizeof *local_w * n);

// Adesso 0 invia a tutti un pezzo della matrice
int num = local_n*n;

MPI_Scatter(
    &A[0], num, MPI_DOUBLE,
    &local_A[0], num, MPI_DOUBLE,
    0, MPI_COMM_WORLD);

MPI_Barrier(MPI_COMM_WORLD);

// Scriviamo la matrice locale ricevuta se sono al massimo 30 elementi
if(num<30){
	printf("\n Local_A %d: \n", me); 
    print_matrix_array(local_A,local_n,n);
}
printf("\n");

/* sincronizzazione dei processori del contesto MPI_COMM_WORLD*/
	
MPI_Barrier(MPI_COMM_WORLD);
 
T_inizio=MPI_Wtime(); //inizio del cronometro per il calcolo del tempo di inizio

//PRODOTTO MATRICE-VETTORE
for(i=0;i<n;i++){
    local_w[i] = 0;
        for(j=0;j<local_n;j++){ //A[i*n+j]
            local_w[i] += local_A[j*n+i] * local_v[j];          
        } 
    }  

MPI_Reduce(&local_w[0],&w[0],n,
    MPI_DOUBLE, MPI_SUM,
    0, MPI_COMM_WORLD);

MPI_Barrier(MPI_COMM_WORLD); // sincronizzazione
T_fine=MPI_Wtime()-T_inizio; // calcolo del tempo di fine

MPI_Reduce(&T_fine,&T_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

/*stampa a video dei risultati finali*/
if(me==0){
    if(n<=10){
    printf("\n Il risultato è w [] \n"); 
        for(i = 0; i < n; i++)
            printf("%5.3f ", w[i]);
    }    
    printf("\n");
	printf("\nProcessori impegnati: %d\n", nproc);
	printf("\nTempo calcolo locale: %lf\n", T_fine);
	printf("\nMPI_Reduce max time: %f\n",T_max);
}// end if

MPI_Finalize (); /* Disattiva MPI */
return 0;  
}
