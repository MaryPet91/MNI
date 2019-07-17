/* --------------------------------------------------------------------------
  SCOPO
  Calcolo della somma di n numeri interi
  su un calcolatore parallelo tipo MIMD
  con p processori, con p potenza di 2.
--------------------------------------------------------------------------
  DESCRIZIONE
  Inizialmente il processo zero legge i dati di input e li distribuisce
  ai vari processori, in modo che ognuno di essi abbia lo stesso numero
  di elementi a meno di una unità.
  Tali elementi sono  memorizzati in un vettore x di 
  dimensione nloc (intero dipendente dal processore).
  Quindi ogni processore effettua la somma parziale e la invia ad un altro processore,
  quindi per passi si giunge alla somma totale degli elementi dati in input.
  Attraverso la funzione di MPI Wtime, ciascun processore calcola il
  tempo di esecuzione della funzione somma. Il processore 0 riceve tali 
  tempi e, ne calcola il massimo, stima del tempo di esecuzione della somma totale.
  Il numero di processi deve essere pari a 2^p con p>=1.
  Il numero di elementi da sommare deve essere maggiore o uguale al numero
  di processi.
----------------------------------------------------------------------------*/ 
/*----------------------------------------------------------------------------
  Inclusione dei file di intestazione usati
----------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
/*--------------------------------------------------------------------------
  Inclusione del file che contiene le definizioni necessarie al preprocessore
  per l'utilizzo di MPI.
-----------------------------------------------------------------------------*/
#include <mpi.h>

int main (int argc, char **argv)
{
	/*dichiarazioni variabili*/
 	int menum,nproc,tag;
	int n,nloc,i,resto,nlocgen;
	int ind,p,r,sendTo,recvBy;
	int *dist,passi=0;
	double T_inizio,T_fine,T_max;
	float sommaloc=0,tmp,*vett,*vett_loc;

	MPI_Status info;
	
	/*Inizializzazione dell'ambiente di calcolo MPI*/
	MPI_Init(&argc,&argv);
	/*assegnazione IdProcessore a menum*/
	MPI_Comm_rank(MPI_COMM_WORLD, &menum);
	/*assegna numero processori a nproc*/
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	/* lettura e inserimento dati*/
	if (menum==0)
	{
		/*printf("Inserire il numero di elementi da sommare: \n");
		fflush(stdout);
		scanf("%d",&n);
		*/
		if (argc < 2) {
      		printf("Numero di parametri insufficiente!");
      		exit(0);;
    	}
   		else {
     		n = atoi(argv[1]);
    	}
    	printf("Addizione di %d numeri reali.\n", n);
       	vett=(float*)calloc(n,sizeof(float));
	}

	/*invio del valore di n a tutti i processori appartenenti a MPI_COMM_WORLD*/
	MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    
    /*numero di addendi da assegnare a ciascun processore*/
	nlocgen=n/nproc; // divisione intera
	
    resto=n%nproc; // resto della divisione

	/* Se resto è non nullo, i primi resto processi ricevono un addento in più */
	if(menum<resto)
	{
		nloc=nlocgen+1;
	}
	else
	{
		nloc=nlocgen;
	}
	
	
    /*allocazione di memoria del vettore per le somme parziali */
    
	vett_loc=(float*)calloc(nloc, sizeof(float));

	// P0 genera e assegna gli elementi da sommare
    
	if (menum==0)
	{
        /*Inizializza la generazione random degli addendi utilizzando l'ora attuale del sistema*/                
       srand((unsigned int) time(0)); 
		int num=0;
		float dec=0;
        for(i=0; i<n; i++)
		{
			/*creazione del vettore contenente numeri casuali */
			num = (int)rand()%50+1;
			dec = ((float)rand())/(float)(RAND_MAX/50);
			vett[i]= num+dec;
		}
		
   		// Stampa del vettore che contiene i dati da sommare, se sono meno di 100 
		if (n<100)
		{
			for (i=0; i<n; i++)
			{
				printf("\n\nElemento %d del vettore = %f",i,*(vett+i));
			}
        }

	// assegnazione dei primi addendi a P0
        
        for(i=0;i<nloc;i++)
		{
			*(vett_loc+i)=*(vett+i);
		}
  
  	// ind è il numero di addendi già assegnati     
			ind=nloc;
        
		/* P0 assegna i restanti addendi agli altri processori */
		for(i=1; i<nproc; i++)
		{
			tag=i; /* tag del messaggio uguale all'id del processo che riceve*/
			/*SE ci sono addendi in sovrannumero da ripartire tra i processori*/
            if (i<resto) 
			{
				/*il processore P0 gli invia il corrispondete vettore locale considerando un addendo in piu'*/
				MPI_Send(vett+ind,nloc,MPI_INT,i,tag,MPI_COMM_WORLD);
				ind=ind+nloc;
			} 
			else 
			{
				/*il processore P0 gli invia il corrispondete vettore locale*/
				MPI_Send(vett+ind,nlocgen,MPI_INT,i,tag,MPI_COMM_WORLD);
				ind=ind+nlocgen;
			}// end if
		}//end for
	}
	 
    /*SE non siamo il processore P0 riceviamo i dati trasmessi dal processore P0*/
    else
    {
		// tag è uguale numero di processore che riceve
		tag=menum;
  
		/*fase di ricezione*/
		MPI_Recv(vett_loc,nloc,MPI_INT,0,tag,MPI_COMM_WORLD,&info);
	}// end if

   
	
	/* sincronizzazione dei processori del contesto MPI_COMM_WORLD*/
	
	MPI_Barrier(MPI_COMM_WORLD);
 
	T_inizio=MPI_Wtime(); //inizio del cronometro per il calcolo del tempo di inizio

	for(i=0;i<nloc;i++)
	{
		/*ogni processore effettua la somma parziale*/
		sommaloc=sommaloc+*(vett_loc+i);
	}

	//  calcolo di p=log_2 (nproc)
	p=nproc;

	while(p!=1)
	{
		/*shifta di un bit a destra*/
		p=p>>1;
		passi++;
	}
 
	/* creazione del vettore dist, che contiene le potenze di 2*/
	dist=(int*)calloc(passi+1,sizeof(int));
		
	for(i=0;i<=passi;i++)
	{
		dist[i]=p<<i;
	}

	/* calcolo delle somme parziali e combinazione dei risultati parziali */
	for(i=0;i<passi;i++)
	{
		// ... calcolo identificativo del processore
		r=menum%dist[i+1];
		
		// Se il resto è uguale a 2^i, il processore menum invia
		if(r==dist[i])
		{
			// calcolo dell'id del processore a cui spedire la somma locale
			sendTo=menum-dist[i];
			tag=sendTo;
			MPI_Send(&sommaloc,1,MPI_INT,sendTo,tag,MPI_COMM_WORLD);
		}
		else if(r==0) // se il resto è uguale a 0, il processore menum riceve
		{
			recvBy=menum+dist[i];
			tag=menum;
			MPI_Recv(&tmp,1,MPI_INT,recvBy,tag,MPI_COMM_WORLD,&info);
			/*calcolo della somma parziale al passo i*/
			sommaloc=sommaloc+tmp;
		}//end
	}// end for

	MPI_Barrier(MPI_COMM_WORLD); // sincronizzazione
	T_fine=MPI_Wtime()-T_inizio; // calcolo del tempo di fine
 
	/* calcolo del tempo totale di esecuzione*/
	MPI_Reduce(&T_fine,&T_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

	/*stampa a video dei risultati finali*/
	if(menum==0)
	{
		printf("Processori impegnati: %d\n", nproc);
		printf("Tempo calcolo locale: %lf\n", T_fine);
		printf("MPI_Reduce max time: %f\n",T_max);
	}// end if
 
	/*routine chiusura ambiente MPI*/
	MPI_Finalize();
}// fine programma

