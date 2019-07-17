#include<cuda.h>
#include<stdio.h>

void initializeArray(int*,int);
void stampaMatriceArray(int*, int, int);
void equalArray(int*, int*, int);
void sommaMatriciCompPerCompCPU(int *, int *, int *, int);

//specifica il tipo di funzione kernel
__global__ void sommaMatriciCompPerCompGPU(int*, int*, int*, int);

int main(int argn, char * argv[]){

	//numero di blocchi e numero di thread per blocco
	dim3 gridDim, blockDim(8,4); //blocco 8*4 = 32 thread totali
	int N; //numero totale di elementi dell'array (matrice)

	//array memorizzati sull'host
	int *A_host, *B_host, *C_host;
	//array memorizzati sul device
	int *A_device, *B_device, *C_device;
	int *copy; //array in cui copieremo i risultati di C_device
	int size; //size in byte di ciascun array

	printf("***\t SOMMA COMPONENTE PER COMPONENTE DI DUE MATRICI \t***\n");
	printf("Inserisci il numero di elementi della matrice\n");
	scanf("%d",&N); 

	//determinazione esatta del numero di blocchi
	gridDim.x = N / blockDim.x + ((N % blockDim.x) == 0 ? 0:1); //se la divisione ha resto dobbiamo aggiungere un blocco in più alle righe 
	gridDim.y = N / blockDim.y + ((N % blockDim.y) == 0 ? 0:1); //se la divisione ha resto dobbiamo aggiungere un blocco in più alle colonne

	//size in byte di ogni array
	size = N*N*sizeof(int);
	
	//stampa delle info sull'esecuzione del kernel
	printf("Taglia della matrice N*N = %d * %d\n", N,N);
	printf("Numero di thread per blocco = %d\n", blockDim.x*blockDim.y);
	printf("Numero di blocchi = %d\n", gridDim.x*gridDim.y);
	
	//allocazione dati sull'host
	A_host=(int*)malloc(size);
	B_host=(int*)malloc(size);
	C_host=(int*)malloc(size);
	copy=(int*)malloc(size); //array in cui copieremo i risultati di C_device

	//allocazione dati sul device
	cudaMalloc((void**)&A_device,size);
	cudaMalloc((void**)&B_device,size);
	cudaMalloc((void**)&C_device,size);

	//inizializzazione dati sull'host
	initializeArray(A_host, N*N);
	initializeArray(B_host, N*N);
	
	//copia dei dati dall'host al device
	cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_device, B_host, size, cudaMemcpyHostToDevice);
	
	//azzeriamo il contenuto della vettore C
	memset(C_host, 0, size); //setta a 0 l'array C_host
	cudaMemset(C_device, 0, size); //setta a 0 l'array C_device
	
	//invocazione del kernel
	sommaMatriciCompPerCompGPU<<<gridDim, blockDim>>>(A_device, B_device, C_device, N*N);
	
	//copia dei risultati dal device all'host
	cudaMemcpy(copy, C_device, size, cudaMemcpyDeviceToHost);
	
	//chiamata alla funzione seriale per il prodotto di due array
	sommaMatriciCompPerCompCPU(A_host, B_host, C_host, N*N);

	//test di correttezza: verifichiamo che le due somme di matrici corrispondano
	equalArray(C_host, copy, N*N);

	//de-allocazione host
	free(A_host);
	free(B_host);
	free(C_host);
	free(copy);
	//de-allocazione device
	cudaFree(A_device);
	cudaFree(B_device);
	cudaFree(C_device);
	exit(0);
	}

	void initializeArray(int *array, int n){
		int i;
		for(i=0;i<n;i++)
			array[i] = i;
	}

	void stampaMatriceArray(int* matrice, int righe, int colonne){
		int i;
		for(i=0;i<righe*colonne;i++){
			printf("%d \t", matrice[i]);
			if(i%righe==colonne-1)
				printf("\n");
		}
		printf("\n");
	}

	void equalArray(int* a, int*b, int n){
		int i=0;
		while(a[i]==b[i])
			i++;
		if(i<n)
			printf("I risultati dell'host e del device sono diversi\n");
		else
			printf("I risultati dell'host e del device coincidono\n");
	}

	//Seriale
	void sommaMatriciCompPerCompCPU(int *a, int *b, int *c, int n){
		int i;
		for(i=0;i<n;i++)
			c[i]=a[i]+b[i];
	}

	//Parallelo
	__global__ void sommaMatriciCompPerCompGPU(int *a, int *b, int *c, int n){
		int i, j, index;
		i = blockIdx.x * blockDim.x + threadIdx.x;
		j = blockIdx.y * blockDim.y + threadIdx.y;
		index = j * gridDim.x * blockDim.x + i;

		if(index < n)
			c[index] = a[index]+b[index];
	}
