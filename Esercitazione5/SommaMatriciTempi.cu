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
	dim3 gridDim, blockDim;
	int N; //numero totale di elementi dell'array (matrice)
	int flag;

	//array memorizzati sull'host
	int *A_host, *B_host, *C_host;
	//array memorizzati sul device
	int *A_device, *B_device, *C_device;
	int *copy; //array in cui copieremo i risultati di C_device
	int size; //size in byte di ciascun array

	int SM = 1536; //max num blocchi 8
	int threadEffettiviSM = 0;
	int blocResidentiSM = 0;
	int num = 8;

	if(argn<4){
		printf("Numero di parametri insufficiente!!!\n");
		printf("Uso corretto: %s <NumElementi> <NumThreadPerBlocco> <flag per la Stampa>\n",argv[0]);
		printf("Uso dei valori di default\n");
		blockDim.x = blockDim.y = num;
		N=100;
		flag=0;
	}
	else{
		num= atoi(argv[2]);
		N=atoi(argv[1]);
		blockDim.x = blockDim.y = num;
		flag=atoi(argv[3]);
	}

	printf("***\t SOMMA COMPONENTE PER COMPONENTE DI DUE MATRICI \t***\n");

	//determinazione esatta del numero di blocchi
	gridDim.x = N / blockDim.x + ((N % blockDim.x) == 0 ? 0:1); //se la divisione ha resto dobbiamo aggiungere un blocco in più alle righe 
	gridDim.y = N / blockDim.y + ((N % blockDim.y) == 0 ? 0:1); //se la divisione ha resto dobbiamo aggiungere un blocco in più alle colonne

	//size in byte di ogni array
	size = N*N*sizeof(int);
	blocResidentiSM = SM / (blockDim.x*blockDim.y);

	//stampa delle info sull'esecuzione del kernel
	printf("Taglia della matrice N*N = %d * %d\n", N,N);
	printf("Numero di thread per blocco = %d\n", blockDim.x*blockDim.y);
	printf("Numero di blocchi = %d\n", gridDim.x*gridDim.y);
	printf("Numero di blocchi residenti per SM in totale= %d \n", blocResidentiSM);
	printf("Numero di SM usati in totale= %d \n", blocResidentiSM/8);

	threadEffettiviSM = blockDim.x*blockDim.y*8;
	if(threadEffettiviSM == SM)
		printf("Uso ottimale degli SM \n");
	else
		printf("Usati solo %d thread di %d per ogni SM \n",threadEffettiviSM,SM);

	
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

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	
	//invocazione del kernel
	sommaMatriciCompPerCompGPU<<<gridDim, blockDim>>>(A_device, B_device, C_device, N*N);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop); // assicura che tutti siano arrivati all'evento stop prima di registrare il tempo
	float elapsed;
	// tempo tra i due eventi in millisecondi
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	
	//copia dei risultati dal device all'host
	cudaMemcpy(copy, C_device, size, cudaMemcpyDeviceToHost);
	
	printf("tempo GPU=%f\n", elapsed);

	// calcolo su CPU
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	//chiamata alla funzione seriale per il prodotto di due array
	sommaMatriciCompPerCompCPU(A_host, B_host, C_host, N*N);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop); // assicura che tutti siano arrivati all'evento stop prima di registrare il tempo
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("tempo CPU=%f\n", elapsed);

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
		for(i=0;i<n;i++){
			array[i] = 1/((i+1)*10);
			if (i % 2 == 0)
        		array[i] = array[i]*(-1);		
        }
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
