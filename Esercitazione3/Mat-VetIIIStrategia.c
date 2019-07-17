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

/**
 * Funzione che esegue la trasposizione di una matrice di interi
 */
void transpose_matrix_array(double *array, double *transpose, int rows, int columns) { 
    int i, j;
    int source_position, destination_position;
    float source;

    for (i = 0; i < rows; ++i) {
        for (j = 0; j < columns; ++j) {
            source_position = columns * i + j;
            destination_position = rows * j + i;
            source = *(array + source_position);
            *(transpose + destination_position) = source;
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
double *A, *v, *localA, *localAt, *local_w, *local_v, *w, *subA, *subAt, *col_w;
int reorder, me_griglia, me_riga, me_colonna, dim=2, coord_riga, coord_col, num;
int period[2], coords[2], ndim[2], belongs[2];
MPI_Comm griglia, righe, colonne; //nuovi communicator

//Variabili per il tempo
double T_inizio,T_fine,T_max;

/*Attiva MPI*/
MPI_Init(&argc, &argv);
/*Trova il numero totale dei processi*/
MPI_Comm_size (MPI_COMM_WORLD, &nproc);
/*Da ad ogni processo il proprio numero identificativo*/
MPI_Comm_rank (MPI_COMM_WORLD, &me);

printf("P[%d]\n", me);
//MASTER
if(me == 0){
	printf("*** MAT - VET III STRATEGIA *** \n"); 
    printf("Inserire n: \n"); 
    scanf("%d",&n); 
    // Porzione di dati da processare
    local_n = n/sqrt(nproc);
    //local_n = n/nproc;  
    //printf("local_n = %d \n",local_n); 
    
    // Alloco spazio di memoria
    A = malloc(n * n * sizeof(double));
    v = malloc(n * sizeof(double));
    w = malloc(n * sizeof(double)); 
    
    //P00 inizializza i dati
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

    if(n<10){
        printf("\n Matrice A: \n"); 
        print_matrix_array(A,n,n);
        printf("\n"); 

        printf("\n Vettore v: \n"); 
        for (i=0;i<n;i++)   
            printf("v[%d] %5.3f \t",i,v[i]);
        printf("\n");
    }
 	
}     

ndim[0]=ndim[1]=dim; //righe e colonne della topologia
period[0]=period[1]=0; //nessuna periodicità
reorder=0; //nessuna ottimizzazione

MPI_Cart_create(MPI_COMM_WORLD,dim,ndim,period,reorder,&griglia); //crea griglia

MPI_Comm_rank(griglia,&me_griglia); //ogni processo acquisisce il proprio rank nella griglia
MPI_Cart_coords(griglia, me_griglia, dim, coords); //coordinate del processo

// Spedisco n e local_n
MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);            
MPI_Bcast(&local_n,1,MPI_INT,0,MPI_COMM_WORLD);    

//creo sottogruppi per righe, belongs[1]=1 fissa l'indice di riga
belongs[0] = 0; 	belongs[1] = 1;

MPI_Cart_sub(griglia, belongs, &righe);
MPI_Comm_rank(righe, &me_riga);
MPI_Cart_coords(righe, me_riga, 1, &coord_riga);

//creo sottogruppi per colonne, belongs[0]=1 fissa l'indice di colonna
belongs[0] = 1; 	belongs[1] = 0;

MPI_Cart_sub(griglia, belongs, &colonne);
MPI_Comm_rank(colonne, &me_colonna);
MPI_Cart_coords(colonne, me_colonna, 1, &coord_col);

MPI_Barrier(MPI_COMM_WORLD); //facciamo sì che tutti i processi acquiscano la propria coordinata

//GESTIONE MATRICE
//scatter matrice lungo la prima colonna di processi
num = local_n * n;
subA = malloc (num * sizeof(double));
subAt = malloc (num * sizeof(double));

if(coords[1] == 0){ //processi prima colonna    
    MPI_Scatter(&A[0], num, MPI_DOUBLE, &subA[0], num, MPI_DOUBLE, 0, colonne);
    transpose_matrix_array(subA, subAt, local_n, n); //n, local_n
}

localAt = malloc (local_n * local_n * sizeof(double));
//0 è il primo processo di ogni riga
MPI_Scatter(&subAt[0], local_n*local_n, MPI_DOUBLE, &localAt[0], local_n*local_n, MPI_DOUBLE, 0, righe); 

localA = malloc (local_n * local_n * sizeof(double));
transpose_matrix_array(localAt, localA, local_n, local_n);

if(n<10){
    printf("\n local_A: \n"); 
    print_matrix_array(localA,local_n,local_n);
    printf("\n");  
}


//GESTIONE VETTORE
local_v = malloc (local_n * sizeof(double));
if(coords[0]==0){ //Scatter lungo la prima riga
    MPI_Scatter(&v[0], local_n, MPI_DOUBLE, &local_v[0], local_n, MPI_DOUBLE, 0, righe);
}

MPI_Bcast(&local_v[0], local_n, MPI_DOUBLE, 0, colonne);

//CALCOLI PARZIALI
local_w = malloc (local_n * sizeof(double));

MPI_Barrier(MPI_COMM_WORLD);
T_inizio = MPI_Wtime(); //inizio del cronometro per il calcolo del tempo di inizio

prod_mat_vett(local_w, localA, local_n, local_n, local_v);

MPI_Barrier(MPI_COMM_WORLD); // sincronizzazione
T_fine = MPI_Wtime() - T_inizio; // calcolo del tempo di fine

//CALCOLI TOTALI
if(coords[1]==0)
    col_w = malloc (local_n * sizeof(double));

MPI_Reduce(&local_w[0], &col_w[0], local_n, MPI_DOUBLE, MPI_SUM, 0, righe); //ogni processo Pi0 raccoglie w[i]

MPI_Gather(&col_w[0], local_n, MPI_DOUBLE, &w[0], local_n, MPI_DOUBLE, 0, colonne); //P00 raccoglie tutti i w[i]

MPI_Reduce(&T_fine, &T_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

if(me==0){  
    printf("\n Il risultato è w [] \n"); 
    for(i = 0; i < n; i++)
        printf("%5.3f \t", w[i]);
    printf("\n");
    printf("\nProcessori impegnati: %d\n", nproc);
    printf("\nTempo calcolo locale: %lf\n", T_fine);
    printf("\nMPI_Reduce max time: %f\n",T_max);
}// end if

MPI_Finalize (); /* Disattiva MPI */
return 0;  
}