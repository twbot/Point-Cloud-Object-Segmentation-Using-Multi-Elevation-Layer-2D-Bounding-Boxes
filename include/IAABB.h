#pragma once

#include <Eigen/Dense>

struct AABB;

class IAABB
{
public:
	virtual ~IAABB() = default;
	virtual AABB getAABB() const = 0;
	virtual Eigen::Vector3f getCentroid() const = 0;
};