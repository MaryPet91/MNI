/*
* Studente: Petraglia Mariangela 0522500473
*/
#include<cuda.h>
#include<stdio.h>

//funzioni host
void initializeArray(int*,int);
void stampaArray(int*, int);
void equalArray(int*, int*, int);
void prodottoArrayCompPerCompCPU(int *, int *, int *, int);

//funzioni kernel
__global__ void dotProdGPU(int *, int *, int *, int);
__global__ void reduce2(int *, int *, int *, int);
__global__ void reduce3(int *, int *, int *, int);

int main(int argn, char * argv[]){

	//numero di blocchi e numero di thread per blocco
	dim3 gridDim, blockDim;
	int N; //numero totale di elementi dell'array
	//array memorizzati sull'host
	int *A_host, *B_host, *C_host;
	//array memorizzati sul device
	int *A_device, *B_device, *C_device;
	int *copy, *shared; //array in cui copieremo i risultati di C_device
	int size; //size in byte di ciascun array
	int sumC_host, sumC_device, i, sumReduce = 0; 

	int SM = 1536; //max num blocchi 8
	int threadEffettiviSM = 0;
	int blocResidentiSM = 0;
	int num = 8;
	int flag;

	cudaEvent_t start, stop;
	float elapsed;

	printf("***\t PRODOTTO COMPONENTE PER COMPONENTE DI DUE ARRAY \t***\n");
	
	if(argn<4){
		printf("Numero di parametri insufficiente!!!\n");
		printf("Uso corretto: %s <NumElementi> <NumThreadPerBlocco> <flag per la Stampa>\n",argv[0]);
		printf("Uso dei valori di default\n");
		N = 128;
		flag = 0;
	}
	else{
		N = atoi(argv[1]);
		num = atoi(argv[2]);	
		flag = atoi(argv[3]);
	}

	blockDim.x = num;

	//determinazione esatta del numero di blocchi - se la divisione ha resto dobbiamo aggiungere un blocco in più -> load balancing
	gridDim = N / blockDim.x + ((N % blockDim.x) == 0 ? 0:1); 
	
	//size in byte di ogni array
	size = N*sizeof(int);
	blocResidentiSM = SM / blockDim.x;

	//stampa delle info sull'esecuzione del kernel
	printf("Taglia dell'array N = %d \n", N);
	printf("Numero di thread per blocco = %d\n", blockDim.x);
	printf("Numero di blocchi = %d\n", gridDim.x);
	printf("Numero di blocchi residenti per SM in totale = %d \n", blocResidentiSM);
	printf("Numero di SM usati in totale = %d \n", blocResidentiSM/8);

	threadEffettiviSM = blockDim.x * 8;
	if(threadEffettiviSM == SM)
		printf("Uso ottimale degli SM \n");
	else
		printf("Usati solo %d thread di %d per ogni SM \n", threadEffettiviSM, SM);
	
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

	//***STRATEGIA 1***//

	// calcolo su CPU
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	elapsed = 0;
	//chiamata alla funzione seriale per il prodotto di due array
	prodottoArrayCompPerCompCPU(A_host, B_host, C_host, N);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop); // assicura che tutti siano arrivati all'evento stop prima di registrare il tempo
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("Tempo CPU=%f\n", elapsed);
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	//invocazione del kernel
	dotProdGPU<<<gridDim, blockDim>>>(C_device, A_device, B_device, N); //STRATEGIA 1

	cudaEventRecord(stop);
	cudaEventSynchronize(stop); // assicura che tutti siano arrivati all'evento stop prima di registrare il tempo
	cudaEventElapsedTime(&elapsed, start, stop);// tempo tra i due eventi in millisecondi
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	//copia dei risultati dal device all'host
	cudaMemcpy(copy, C_device, size, cudaMemcpyDeviceToHost);

	//test di correttezza: verifichiamo che le due somme corrispondano
	sumC_host = 0;
	sumC_device = 0;

	for(i=0; i<N; i++){
		sumC_host += C_host[i]; 
		sumC_device += copy[i];
	}

	if(sumC_host==sumC_device)
		printf("Le somme coincidono: host (%d) - device (%d) \n", sumC_host, sumC_device);
	else
		printf("Le somme NON coincidono: host (%d) - device (%d) \n", sumC_host, sumC_device);

	printf("Tempo GPU I strategia = %f\n", elapsed);

	//*** STRATEGIA 2 - shared memory ***//

	shared = (int*) calloc (N, sizeof(int));//vettore somme parziali
	cudaFree(C_device);
	cudaMalloc((void **)&C_device, gridDim.x*sizeof(int)); //C_Device deve avere un numero di elementi pari al numero di blocchi
	cudaMemset(C_device, 0, size); //setta a 0 l'array C_device

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	//invocazione del kernel
	reduce2<<<gridDim, blockDim, blockDim.x * sizeof(int)>>>(C_device, A_device, B_device, N);
		
	cudaEventRecord(stop);
	cudaEventSynchronize(stop); //assicura che tutti siano arrivati all'evento stop prima di registrare il tempo
	elapsed = 0;
	cudaEventElapsedTime(&elapsed, start, stop); 	// tempo tra i due eventi in millisecondi
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//copia dei risultati dal device all'host
	cudaMemcpy(shared, C_device, gridDim.x*sizeof(int), cudaMemcpyDeviceToHost);

	sumReduce = 0;
	for(i=0; i<gridDim.x; i++)
		sumReduce+=shared[i];

	if(sumC_host==sumReduce)
		printf("Le somme coincidono: host (%d) - device (%d) \n", sumC_host, sumReduce);
	else
		printf("Le somme NON coincidono: host (%d) - device (%d) \n", sumC_host, sumReduce);

	printf("Tempo GPU II strategia = %f\n", elapsed);

	//***STRATEGIA 3 - shared memory per evitare divergenza e conflitti di accesso ai banchi del shared mem ***//
	
	memset(shared, 0, size);
	cudaMemset(C_device, 0, size); //setta a 0 l'array C_device

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	//invocazione del kernel
	reduce3<<<gridDim, blockDim, blockDim.x * sizeof(int)>>>(C_device, A_device, B_device, N);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop); //assicura che tutti siano arrivati all'evento stop prima di registrare il tempo
	elapsed = 0;
	cudaEventElapsedTime(&elapsed, start, stop);// tempo tra i due eventi in millisecondi
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//copia dei risultati dal device all'host
	cudaMemcpy(shared, C_device, gridDim.x*sizeof(int), cudaMemcpyDeviceToHost);

	sumReduce = 0;
	for(i=0; i<gridDim.x; i++)
		sumReduce+=shared[i];

	if(sumC_host==sumReduce)
		printf("Le somme coincidono: host (%d) - device (%d) \n", sumC_host, sumReduce);
	else
		printf("Le somme NON coincidono: host (%d) - device (%d) \n", sumC_host, sumReduce);

	printf("Tempo GPU III strategia = %f\n", elapsed);

	//de-allocazione host
	free(A_host);
	free(B_host);
	free(C_host);
	free(copy);
	free(shared);

	//de-allocazione device
	cudaFree(A_device);
	cudaFree(B_device);
	cudaFree(C_device);
	exit(0);
	}

	void initializeArray(int *array, int n){
		int i;
		for(i=0;i<n;i++)
			array[i] = rand() % 5;
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
			printf("I risultati dell'host e del device sono diversi \n");
		else
			printf("I risultati dell'host e del device coincidono \n");
	}

	//Seriale
	void prodottoArrayCompPerCompCPU(int *a, int *b, int *c, int N){
		int i;
		for(i=0;i<N;i++)
			c[i]=a[i]*b[i];
	}

	//Parallelo

	__global__ void dotProdGPU(int *c, int *a, int *b, int N){
		// global index
	   	int index = blockDim.x * blockIdx.x + threadIdx.x;

	   	if (index < N)
	      c[index] = a[index]*b[index];
	}

	__global__ void reduce2(int *c, int *a, int *b, int N){
		
		extern __shared__ int sdata[];

		// global index
		int index = blockDim.x * blockIdx.x + threadIdx.x;

		//calcolo prodotto
		if (index < N)
			sdata[threadIdx.x] = a[index]*b[index];

		__syncthreads();

		//do reduction in shared mem
		for (unsigned int s = 1; s < blockDim.x; s *= 2){ 
			// step = x*2
			if(threadIdx.x % (2*s) == 0) { // only threadIDs divisible by step participate
				sdata[threadIdx.x] += sdata[threadIdx.x + s];
			}

		__syncthreads();
		}

		// write result for this block to global mem		
		if (threadIdx.x == 0) 
			c[blockIdx.x] = sdata[threadIdx.x];
	}

	__global__ void reduce3(int *c, int *a, int *b, int N){
		
		extern __shared__ int sdata[];

		// global index
		int index = blockDim.x * blockIdx.x + threadIdx.x;

		//calcolo prodotto
		if (index < N)
	   		sdata[threadIdx.x] = a[index]*b[index];

		__syncthreads();

		// do reduction in shared mem
		for (unsigned int s = blockDim.x / 2; s>0; s >>= 1){ // s è la distanza
			// s = s/2
			if(threadIdx.x < s) { 
				sdata[threadIdx.x] += sdata[threadIdx.x + s];
			}

		__syncthreads();
		}

		// writeresultfor this block to global mem
		if (threadIdx.x == 0) 
			c[blockIdx.x] = sdata[threadIdx.x];
	}


