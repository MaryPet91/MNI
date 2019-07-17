/*
*
* Studente: Petraglia Mariangela 0522500473
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NANOSECONDS_PER_SECOND 1E9

int main(int argc, char *argv[]){

	int num=0,somma=0,i,nuovoNum=0;
	struct timespec fine,inizio;
	double tempoTot;
	int *vett;

	printf("Inserisci il numero degli elementi da sommare: ");
	fflush(stdout);
	scanf("%d",&num);
	vett=(int*)calloc(num,sizeof(int));
	for(i=0;i<num;i++){
		nuovoNum=rand() % 50+1; //Numeri tra 0 e 50
		vett[i]=nuovoNum;
	}
	clock_gettime(CLOCK_REALTIME, &inizio);
	for(i=0;i<num;i++)
		somma+=vett[i];
	clock_gettime(CLOCK_REALTIME, &fine);
	tempoTot = (fine.tv_sec - inizio.tv_sec) + (fine.tv_nsec - inizio.tv_nsec)/NANOSECONDS_PER_SECOND;
    printf("Il tempo totale per sommare %d e' %lf \n",num,tempoTot);
    return 0;
}
