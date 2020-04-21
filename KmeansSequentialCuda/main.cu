#include <iostream>
#include <fstream>
#include "Point.h"
#include "KMeansCPU.h"
#include "KMeansGPU.h"

#include <stdio.h>
#include <ctime>



#define POINTS_FOR_THREAD 32

using namespace std;



int main(int argc, char** argv)
{
	std::clock_t c_start, c_end;
	double time_elapsed_ms;
	int size;
	int k;

	srand(time(NULL));

	size = 500000;			//numero punti
	k = 10;				//numero clusters



	//generazione punti casuali e scrittura su file
	fstream myfile;
	myfile.open("points.txt");

	Point* points = new Point[size];
	for (int i = 0; i < size; i++)
	{
		points[i] = Point(rand() % 1000, rand() % 1000, rand() % 1000);
		if (i >= 1000000)
			myfile << points[i].X << " " << points[i].Y << " " << points[i].Z << "\n";
	}
	myfile.close();



	//utilizzo punti sul file
	/*
	fstream myfile;
	myfile.open("points.txt");

	Point* points = new Point[size];
	double a;
	int count = 0;
	int index = 0;
	int i = 0;
	while (myfile >> a && i < size)
	{
		if (count == 0) {
			points[index].X = a;
		}
		else if (count == 1) {
			points[index].Y = a;

		}
		else {
			points[index].Z = a;
		}

		count++;
		if (count == 3) {
			count = 0;
			index++;
		}
		i++;
	}
	*/


	//calcolo su GPU
	c_start = clock();
	SolveGPU(points, size, k);
	c_end = clock();
	time_elapsed_ms = c_end - c_start;
	cout << "\nTime for GPU: " << time_elapsed_ms << endl << endl;



	//calcolo su CPU
	c_start = clock();
	KMeansCPU solver = KMeansCPU(points, size, k);
	solver.Solve();

	std::cout << "\nCPU centroids coordinates:" << std::endl;
	auto result = solver.GetResult();
	for (int i = 0; i < k; i++)
		cout << result[i] << endl;

	c_end = std::clock();
	time_elapsed_ms = c_end - c_start;
	cout << "\nTime for CPU: " << time_elapsed_ms << endl;


	delete[] points;
	delete[] result;
	return 0;
}
