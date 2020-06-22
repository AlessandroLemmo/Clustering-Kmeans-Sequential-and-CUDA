#include "Point.h"

class KMeansCPU
{
	Point* points;
	int* clusters;			//indice del cluster di appartenenza di ogni punto
	int* clusterCounts;		//numero di punti asseggnati ad ogni cluster
	Point* result;
	int size;				//numero di punti
	int k;					//numero di clusters

public:
	KMeansCPU(Point* points, int size, int k) : size(size), k(k)
	{
		this->points = new Point[size];
		this->clusters = new int[size];

		for (int i = 0; i < size; i++) {
			this->points[i] = points[i];
		}
		this->result = new Point[k];
		this->clusterCounts = new int[k];
	}

	~KMeansCPU()
	{
		delete[] points;
		delete[] result;
	}

	Point* GetResult() {
		Point* resultCopy = new Point[k];
		for (int i = 0; i < k; i++) {
			resultCopy[i] = result[i];
		}
		return resultCopy;
	}

	void Solve()
	{
		if (k > size)
			return;

		//inizializzazione clusters a -1
		for (int i = 0; i < size; i++) 										
			clusters[i] = -1;

		for (int i = 0; i < k; i++) {
			//result contiene i centroidi dei clusters; inizializzati uguali ai primi k punti
			result[i] = points[i];											
			clusters[i] = i;
		}

		int iteration = 0;
		int delta;
		do
		{
			for (int i = 0; i < k; i++)
				clusterCounts[i] = 0;

			//numero di punti che ad ogni iterazione vengo cambiati di cluster
			delta = 0;														

			//assegnazione di ogni punto ad un cluster
			//per ogni punto si scorre tutti i centroidi e si determina quello più vicino
			for (int i = 0; i < size; i++)									
			{
				double minDist = DBL_MAX;
				int bestCluster = -1;

				//per ogni cluster
				for (int j = 0; j < k; j++)									
				{
					//calcolo distanza punto i-esimo dal centroide j-esimo
					int dist = Distance(points[i], result[j]);				
					if (dist < minDist)
					{
						minDist = dist;
						//assegno a bestCluster l'indice del centroide del cluster j
						bestCluster = j;									
					}
				}

				if (bestCluster != clusters[i])
				{
					//alla fine dei due for contiene per ogni punto l'indice del cluster a cui appartiene
					clusters[i] = bestCluster;								
					delta++;
				}
				//alla fine dei due for contiene il numero di punti assegnati ad ogni cluster
				clusterCounts[bestCluster]++;								
			}

			//nuovi posizionamenti dei centroidi
			Point* newResult = new Point[k];								

			for (int i = 0; i < size; i++)
			{
				//cluster di appartenenza del punto i-esimo
				int cluster = clusters[i];	
				//somma componente per componente
				newResult[cluster] = newResult[cluster] + points[i];		
			}

			for (int i = 0; i < k; i++)
			{
				if (clusterCounts[i] == 0)
				{
					continue;
				}
				//calcolo del nuovo centroide del cluster i-esimo
				newResult[i] = newResult[i] / clusterCounts[i];				
			}
			delete[] result;
			result = newResult;

			printf("Iteration: %d,     delta: %d\n", iteration++, delta);
		} while (delta > 0);
	}
};
