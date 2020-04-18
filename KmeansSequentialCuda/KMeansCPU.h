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

		for (int i = 0; i < size; i++) 										//inizializzazione clusters a -1
			clusters[i] = -1;

		for (int i = 0; i < k; i++) {
			result[i] = points[i];											//result contiene i centroidi dei clusters; inizializzati uguali ai primi k punti
			clusters[i] = i;
		}

		int iteration = 0;
		int delta;
		do
		{
			for (int i = 0; i < k; i++)
				clusterCounts[i] = 0;

			delta = 0;														//numero di punti che ad ogni iterazione vengo cambiati di cluster

			//assegnazione di ogni punto ad un cluster
			//per ogni punto si scorre tutti i centroidi e si determina quello più vicino
			for (int i = 0; i < size; i++)									//per ogni punto
			{
				double minDist = DBL_MAX;
				int bestCluster = -1;

				for (int j = 0; j < k; j++)									//per ogni cluster
				{
					int dist = Distance(points[i], result[j]);				//calcolo distanza punto i-esimo dal centroide j-esimo
					if (dist < minDist)
					{
						minDist = dist;
						bestCluster = j;									//assegno a bestCluster l'indice del centroide del cluster j
					}
				}

				if (bestCluster != clusters[i])
				{
					clusters[i] = bestCluster;								//alla fine dei due for contiene per ogni punto l'indice del cluster a cui appartiene
					delta++;
				}
				clusterCounts[bestCluster]++;								//alla fine dei due for contiene il numero di punti assegnati ad ogni cluster
			}

			Point* newResult = new Point[k];								//nuovi posizionamenti dei centroidi

			for (int i = 0; i < size; i++)
			{
				int cluster = clusters[i];									//cluster di appartenenza del punto i-esimo
				newResult[cluster] = newResult[cluster] + points[i];		//somma componente per componente
			}

			for (int i = 0; i < k; i++)
			{
				if (clusterCounts[i] == 0)
				{
					continue;
				}
				newResult[i] = newResult[i] / clusterCounts[i];				//calcolo del nuovo centroide del cluster i-esimo
			}
			delete[] result;
			result = newResult;

			printf("Iteration: %d,     delta: %d\n", iteration++, delta);
		} while (delta > 0);
	}
};
