#pragma once

//#include <algorithm>
#include <Eigen/Core>
//
//#ifndef min
//#define min(X, Y)  ((X) < (Y) ? (X) : (Y))
//#endif
//
//#ifndef max
//#define max(X, Y)  ((X) > (Y) ? (X) : (Y))
//#endif

struct AABB
{
private:
	float calculateSurfaceArea() const { return 2.0f * (getWidth() * getHeight() + getWidth()*getDepth() + getHeight()*getDepth()); }

public:
	unsigned long index;
	float minX;
	float minY;
	float minZ;
	float maxX;
	float maxY;
	float maxZ;
	float surfaceArea;

	AABB() : minX(0.0f), minY(0.0f), minZ(0.0f), maxX(0.0f), maxY(0.0f), maxZ(0.0f), surfaceArea(0.0f) { }
	AABB(unsigned minX, unsigned minY, unsigned minZ, unsigned maxX, unsigned maxY, unsigned maxZ) :
		AABB(static_cast<float>(minX), static_cast<float>(minY), static_cast<float>(minZ), static_cast<float>(maxX), static_cast<float>(maxY), static_cast<float>(maxZ)) { }
	AABB(float minX, float minY, float minZ, float maxX, float maxY, float maxZ) :
		minX(minX), minY(minY), minZ(minZ), maxX(maxX), maxY(maxY), maxZ(maxZ)
	{
		surfaceArea = calculateSurfaceArea();
	}

	bool overlaps(const AABB& other) const
	{
		return (maxX > other.minX &&
			minX < other.maxX &&
			maxY > other.minY &&
			minY < other.maxY &&
			maxZ > other.minZ &&
			minZ < other.maxZ);
	}

	bool overlapsOR(const AABB& other) const
	{
		return ((maxX > other.minX ||
			minX < other.maxX) &&
			(maxY > other.minY ||
				minY < other.maxY) &&
				(maxZ > other.minZ ||
					minZ < other.maxZ));
	}

	bool ray_trace(
		Eigen::Vector3f& origin,
		Eigen::Vector3f& r
	) const {
		//Implementation of : https://tavianator.com/2011/ray_box.html
		Eigen::Vector3f inv_r;
		// r is unit direction vector or ray
		// inv_r is inverse of unit direction vector of ray
		inv_r(0) = 1.0f / r(0);
		inv_r(1) = 1.0f / r(1);
		inv_r(2) = 1.0f / r(2);
		// origin is camera center
		float t1 = (minX - origin(0))*inv_r(0);
		float t2 = (maxX - origin(0))*inv_r(0);
		float t3 = (minY - origin(1))*inv_r(1);
		float t4 = (maxY - origin(1))*inv_r(1);
		float t5 = (minZ - origin(2))*inv_r(2);
		float t6 = (maxZ - origin(2))*inv_r(2);

		float tmin = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
		float tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));

		int length = 0;
		// if tmax < 0, ray is intersecting AABB, but the whole AABB is behind us
		if (tmax < 0)
		{
			length = tmax;
			return false;
		}

		// if tmin > tmax, ray doesn't intersect AABB
		if (tmin > tmax)
		{
			length = tmax;
			return false;
		}

		length = tmin;
		return true;
	}

	bool contains(const AABB& other) const
	{
		return (other.minX >= minX &&
			other.maxX <= maxX &&
			other.minY >= minY &&
			other.maxY <= maxY &&
			other.minZ >= minZ &&
			other.maxZ <= maxZ);
	}

	AABB merge(const AABB& other) const
	{
		return AABB(
			std::min(minX, other.minX), std::min(minY, other.minY), std::min(minZ, other.minZ),
			std::max(maxX, other.maxX), std::max(maxY, other.maxY), std::max(maxZ, other.maxZ)
		);
	}

	AABB intersection(const AABB& other) const
	{
		return AABB(
			std::max(minX, other.minX), std::max(minY, other.minY), std::max(minZ, other.minZ),
			std::min(maxX, other.maxX), std::min(maxY, other.maxY), std::min(maxZ, other.maxZ)
		);
	}

	float getWidth() const { return maxX - minX; }
	float getHeight() const { return maxY - minY; }
	float getDepth() const { return maxZ - minZ; }
};