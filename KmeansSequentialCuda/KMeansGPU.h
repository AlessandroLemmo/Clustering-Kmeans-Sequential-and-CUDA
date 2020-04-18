#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include "Point.h"

#define POINTS_FOR_THREAD 32
using namespace std;



struct SumPoints : public thrust::binary_function<Point, Point, Point>
{
	__host__ __device__ Point operator()(Point p1, Point p2) { return Point(p1.X + p2.X, p1.Y + p2.Y, p1.Z + p2.Z); }
};



inline __device__ double PointDistance(const Point& p1, const Point& p2) {
	return sqrt((p1.X - p2.X) * (p1.X - p2.X) + (p1.Y - p2.Y) * (p1.Y - p2.Y) + (p1.Z - p2.Z) * (p1.Z - p2.Z));
}

/*
inline __global__ void Reset(Point* d_new_result, int* d_counts, int* d_k)
{
	int k = *d_k;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= k)
		return;

	d_counts[i] = 0;
	d_new_result[i] = Point();
}*/



inline __global__ void FindCluster(Point* d_points, int* d_clusters, Point* d_result, Point* d_new_result, int* d_clusterCounts, int* d_delta, int* d_k, int* d_size, int threads)
{
	int k = *d_k;
	int size = *d_size;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int cluster = 0; cluster < k; cluster++)
	{
		d_new_result[cluster * threads + i] = Point();
		d_clusterCounts[cluster * threads + i] = 0;						//numero di punti appartenenti ad ogni cluster inizializzati a 0
	}

	//assegnazione del punto i al cluster j
	for (int pointNr = 0; pointNr < POINTS_FOR_THREAD; pointNr++)		//ciclo sul numero di punti di un thread
	{
		int index = i * POINTS_FOR_THREAD + pointNr;

		if (index >= size)
			return;

		double minDist = DBL_MAX;
		int bestCluster = -1;
		d_delta[index] = 0;

		Point p = d_points[index];

		for (int j = 0; j < k; j++)										//scorro tutti i clusters
		{
			int dist = PointDistance(p, d_result[j]);
			if (dist < minDist)
			{
				minDist = dist;
				bestCluster = j;										//trovo per il punto index-esimo il miglior cluster
			}
		}

		if (bestCluster != d_clusters[index])
		{
			d_clusters[index] = bestCluster;
			d_delta[index] = 1;											//d_delta indica il numero di punti cambiati di cluster ad ogni iterazione
		}
		d_clusterCounts[bestCluster * threads + i]++;					//numero di punti per cluster


		//se due o più punti dello stesso thread dello stesso blocco appartengono allo stesso centroide
		//allora si ha una somma altrimenti si somma le coordinate del punto con 0
		Point point = d_new_result[bestCluster * threads + i];
		d_new_result[bestCluster * threads + i] = Point(point.X + p.X, point.Y + p.Y, point.Z + p.Z);  //i valori che ci sono più il nuovo punto
	}
}

inline __global__ void CalculateResult(Point* d_new_result_final, int* d_counts_final, Point* d_result, int* d_k)
{
	int k = *d_k;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= k)
		return;

	Point p = d_new_result_final[i];
	int count = d_counts_final[i];

	d_result[i] = Point(p.X / count, p.Y / count, p.Z / count);
}



inline void SolveGPU(Point* h_points, int size, int k)
{
	int* h_clusters;										//indice del cluster di appartenenza di ogni punto
	Point* h_result;										//centroidi risultati

	//variabili GPU
	Point* d_points;
	Point* d_result;
	Point* d_new_result;
	Point* d_new_result_final;
	int* d_clusters;
	int* d_clusterCounts;
	int* d_counts_final;
	int* d_k;
	int* d_size;
	int* d_delta;
	int nThreads = 128;

	//blocchi per punti da clusterizzare
	int nBlocks = size / POINTS_FOR_THREAD / nThreads;		//   size/POINTS_PER_THREAD =  numero di threads da usare
															//   /nThreads = numero di threads per blocco
	nBlocks += (size % nThreads == 0) ? 0 : 1;

	//blocchi per clusters
	int kBlocks = k / nThreads;								//numero di blocchi per gestire tutti i centroidi 
	kBlocks += (k % nThreads == 0) ? 0 : 1;
	int iteration = 0;

	//come nella versione sequenziale
	h_clusters = new int[size];
	h_result = new Point[k];

	//inizializzazione identica a quella sequenziale
	for (int i = 0; i < size; i++)
	{
		h_clusters[i] = -1;
	}

	for (int i = 0; i < k; i++)
	{
		h_result[i] = h_points[i];
		h_clusters[i] = i;
	}


	cudaMalloc((void**)&d_points, size * sizeof(Point));
	cudaMemcpy(d_points, h_points, size * sizeof(Point), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_clusters, size * sizeof(int));
	cudaMemcpy(d_clusters, h_clusters, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_result, k * sizeof(Point));
	cudaMemcpy(d_result, h_result, k * sizeof(Point), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_k, sizeof(int));
	cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_size, sizeof(int));
	cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_delta, size * sizeof(int));
	cudaMalloc((void**)&d_new_result, nThreads * nBlocks * k * sizeof(Point));
	cudaMalloc((void**)&d_new_result_final, k * sizeof(Point));
	cudaMalloc((void**)&d_clusterCounts, nThreads * nBlocks * k * sizeof(int));
	cudaMalloc((void**)&d_counts_final, k * sizeof(int));


	while (true)
	{

		//Reset << < kBlocks, nThreads >> > (d_new_result_final, d_counts_final, d_k);

		//nBlocks * nThreads = numero totale di thread da usare
		FindCluster << <nBlocks, nThreads >> > (d_points, d_clusters, d_result, d_new_result, d_clusterCounts, d_delta, d_k, d_size, nBlocks * nThreads);

		//somma di tutti i punti appartenenti ad uno stesso cluster
		//riduzione vettore contenente l'indice del cluster di appartenenza di ogni punto 
		//e del vettore contenente per ogni cluster il numero di punti assegnati
		//in quanto per ogni thread eseguito si hanno dei valori che devono essere raggruppati
		for (int i = 0; i < k; i++)
		{
			thrust::device_ptr<Point> dev_new_result_ptr(d_new_result + i * nBlocks * nThreads);
			thrust::device_ptr<int> dev_count_ptr(d_clusterCounts + i * nBlocks * nThreads);

			thrust::device_ptr<Point> dev_new_result_final_ptr(d_new_result_final);
			thrust::device_ptr<int> dev_count_final_ptr(d_counts_final);

			dev_new_result_final_ptr[i] = thrust::reduce(dev_new_result_ptr, dev_new_result_ptr + nThreads * nBlocks, Point(), SumPoints());
			dev_count_final_ptr[i] = thrust::reduce(dev_count_ptr, dev_count_ptr + nThreads * nBlocks);
		}

		//calcolo risultati finali
		CalculateResult << <kBlocks, nThreads >> > (d_new_result_final, d_counts_final, d_result, d_k);

		//riduzione del delta, che indica il numero di punti spostati di cluster ad ogni iterazione, per ottenere un valore unico
		thrust::device_ptr<int> dev_delta_ptr(d_delta);
		int delta = thrust::reduce(thrust::device, dev_delta_ptr, dev_delta_ptr + size);

		cout << "Iteration: " << iteration++ << ",     delta: " << delta << endl;
		if (delta == 0)
			break;
	}


	cudaMemcpy(h_result, d_result, k * sizeof(Point), cudaMemcpyDeviceToHost);

	cout << "\nGPU centroids coordinates:" << endl;
	for (int i = 0; i < k; i++)
	{
		cout << h_result[i] << endl;
	}
}
