#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


class Point
{
public:
	double X, Y, Z;

	__host__ __device__ Point() : X(0), Y(0), Z(0) {}
	__host__ __device__ Point(double x, double y, double z) : X(x), Y(y), Z(z) {}

	friend std::ostream& operator<<(std::ostream& ostream, const Point& p) {
		ostream << "(" << p.X << ", " << p.Y << ", " << p.Z << ")";
		return ostream;
	}

	friend Point operator+(const Point& p1, const Point& p2) {
		return Point(p1.X + p2.X, p1.Y + p2.Y, p1.Z + p2.Z);
	}

	friend Point operator/(const Point& p1, const double& a) {
		return Point(p1.X / a, p1.Y / a, p1.Z / a);
	}

	friend double Distance(const Point& p1, const Point& p2) {
		return sqrt((p1.X - p2.X) * (p1.X - p2.X) + (p1.Y - p2.Y) * (p1.Y - p2.Y) + (p1.Z - p2.Z) * (p1.Z - p2.Z));
	}
};


