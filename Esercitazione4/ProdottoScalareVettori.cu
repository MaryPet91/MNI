#include<cuda.h>
#include<stdio.h>

void initializeArray(int*,int);
void stampaArray(int*, int);
void equalArray(int*, int*, int);
void prodottoArrayCompPerCompCPU(int *, int *, int *, int);

//specifica il tipo di funzione kernel
__global__ void prodottoArrayCompPerCompGPU(int*, int*, int*, int );
int main(int argn, char * argv[]){

	//numero di blocchi e numero di thread per blocco
	dim3 gridDim, blockDim;
	int N; //numero totale di elementi dell'array
	//array memorizzati sull'host
	int *A_host, *B_host, *C_host;
	//array memorizzati sul device
	int *A_device, *B_device, *C_device;
	int *copy; //array in cui copieremo i risultati di C_device
	int size; //size in byte di ciascun array
	int sumC_host, sumC_device, i; 

	printf("***\t PRODOTTO COMPONENTE PER COMPONENTE DI DUE ARRAY \t***\n");
	printf("Inserisci il numero elementi dei vettori\n");
	scanf("%d",&N); 
	//printf("Inserisci il numero di thread per blocco\n");
	//scanf("%d",&blockDim); 
	blockDim = 32;

	//determinazione esatta del numero di blocchi
	gridDim = N / blockDim.x + ((N % blockDim.x) == 0 ? 0:1); //se la divisione ha resto dobbiamo aggiungere un blocco in più -> load balancing
	
	//size in byte di ogni array
	size = N*sizeof(int);
	
	//stampa delle info sull'esecuzione del kernel
	printf("Numero di elementi = %d\n", N);
	printf("Numero di thread per blocco = %d\n", blockDim.x);
	printf("Numero di blocchi = %d\n", gridDim.x);
	
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
	initializeArray(A_host, N);
	initializeArray(B_host, N);
	
	//copia dei dati dall'host al device
	cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_device, B_host, size, cudaMemcpyHostToDevice);
	
	//azzeriamo il contenuto della vettore C
	memset(C_host, 0, size); //setta a 0 l'array C_host
	cudaMemset(C_device, 0, size); //setta a 0 l'array C_device
	
	//invocazione del kernel
	prodottoArrayCompPerCompGPU<<<gridDim, blockDim>>>(A_device, B_device, C_device, N);
	
	//copia dei risultati dal device all'host
	cudaMemcpy(copy, C_device, size, cudaMemcpyDeviceToHost);
	
	//chiamata alla funzione seriale per il prodotto di due array
	prodottoArrayCompPerCompCPU(A_host, B_host, C_host, N);
	
	//stampa degli array e dei risultati

	//test di correttezza: verifichiamo che le due somme corrispondano
	
	sumC_host = sumC_device = 0;

	for(i=0; i<N; i++){
		sumC_host += C_host[i]; 
		sumC_device += copy[i];
	}

	if(sumC_host==sumC_device)
		printf("Le somme coincidono \n");
	else
		printf("Le somme NON coincidono \n");

	/*sumC_host = (int*) malloc (sizeof(int)*1);
	sumC_device = (int*) malloc (sizeof(int)*1);
	sumArray(C_host, N , sumC_host);
	sumArray(C_device, N , sumC_device);*/

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

	void stampaArray(int* array, int n){
		int i;
		for(i=0;i<n;i++)
			printf("%d ", array[i]);
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
	void prodottoArrayCompPerCompCPU(int *a, int *b, int *c, int n){
		int i;
		for(i=0;i<n;i++)
			c[i]=a[i]*b[i];
	}

	//Seriale
	void sumArray(int *c, int n, int *sum){
		int i;
		*sum = 0; 
		for(i=0;i<n;i++)
			*sum+=c[i];
	}

	//Parallelo
	__global__ void prodottoArrayCompPerCompGPU(int* a, int* b, int* c, int n){
		int index=threadIdx.x + blockIdx.x * blockDim.x;
		if(index < n)
			c[index] = a[index]*b[index];
	}
