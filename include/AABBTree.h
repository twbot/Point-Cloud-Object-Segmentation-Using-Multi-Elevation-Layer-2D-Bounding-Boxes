#pragma once

#include <memory>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <forward_list>

#include "AABB.h"
#include "IAABB.h"

#define AABB_NULL_NODE 0xffffffff

struct AABBNode
{
	AABB aabb;
	std::shared_ptr<IAABB> object;
	// tree links
	unsigned long long parentNodeIndex;
	unsigned long long leftNodeIndex;
	unsigned long long rightNodeIndex;
	// node linked list link
	unsigned long long nextNodeIndex;

	bool isLeaf() const { return leftNodeIndex == AABB_NULL_NODE; }

	AABBNode() : object(nullptr), parentNodeIndex(AABB_NULL_NODE), leftNodeIndex(AABB_NULL_NODE), rightNodeIndex(AABB_NULL_NODE), nextNodeIndex(AABB_NULL_NODE)
	{

	}
};

class RR_3DLib_API AABBTree
{
private:
	std::map<std::shared_ptr<IAABB>, unsigned long long> _objectNodeIndexMap;
	std::vector<AABBNode> _nodes;
	unsigned long long _rootNodeIndex;
	unsigned long long _allocatedNodeCount;
	unsigned long long _nextFreeNodeIndex;
	unsigned long long _nodeCapacity;
	unsigned long long _growthSize;

	unsigned long long allocateNode();
	void deallocateNode(unsigned long long nodeIndex);
	void insertLeaf(unsigned long long leafNodeIndex);
	void removeLeaf(unsigned long long leafNodeIndex);
	void updateLeaf(unsigned long long leafNodeIndex, const AABB& newAaab);
	void fixUpwardsTree(unsigned long long treeNodeIndex);

public:
	AABBTree(unsigned long long initialSize);
	~AABBTree();

	void insertObject(const std::shared_ptr<IAABB>& object);
	void removeObject(const std::shared_ptr<IAABB>& object);
	void updateObject(const std::shared_ptr<IAABB>& object);
	std::forward_list<std::shared_ptr<IAABB>> queryOverlaps(const std::shared_ptr<IAABB>& object) const;
	size_t numberIntersections(
		const std::shared_ptr<IAABB>& object,
		Eigen::Vector3f& camera
	) const;
	std::vector<std::shared_ptr<IAABB>> getListIntersected(
		const::std::shared_ptr<IAABB>& object,
		Eigen::Vector3f& camera
	) const;
};