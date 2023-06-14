/*
 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
 holder of all proprietary rights on this computer program.
 You can only use this computer program if you have closed
 a license agreement with MPG or you get the right to use the computer
 program from someone who is authorized to grant you that right.
 Any use of the computer program without a valid license is prohibited and
 liable to prosecution.

 Copyright©2019 Max-Planck-Gesellschaft zur Förderung
 der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
 for Intelligent Systems. All rights reserved.

 Contact: ps-license@tuebingen.mpg.de
*/

#include <torch/extension.h>
#include <torch/types.h>

#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include <cfloat>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "aabb.hpp"
#include "defs.hpp"
#include "double_vec_ops.hpp"
#include "helper_math.h"
#include "triangle.hpp"

// Size of the stack used to traverse the Bounding Volume Hierarchy tree
#ifndef STACK_SIZE
#define STACK_SIZE 64
#endif /* ifndef STACK_SIZE */

// Upper bound for the number of possible collisions
#ifndef MAX_COLLISIONS
#define MAX_COLLISIONS 16
#endif

#ifndef EPSILON
#define EPSILON 1e-4
#endif /* ifndef EPSILON */

// Number of threads per block for CUDA kernel launch
#ifndef NUM_THREADS
#define NUM_THREADS 128
#endif

#ifndef COLLISION_ORDERING
#define COLLISION_ORDERING 1
#endif

#ifndef FORCE_INLINE
#define FORCE_INLINE 1
#endif /* ifndef FORCE_INLINE */

#ifndef ERROR_CHECKING
#define ERROR_CHECKING 1
#endif /* ifndef ERROR_CHECKING */

// Macro for checking cuda errors following a cuda launch or api call
#if ERROR_CHECKING == 1
#define cudaCheckError()                                                       \
  {                                                                            \
    cudaDeviceSynchronize();                                                   \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(e));                                           \
      exit(0);                                                                 \
    }                                                                          \
  }
#else
#define cudaCheckError()
#endif

#define CMP(x, y)                                                              \
  (fabsf(x - y) <= FLT_EPSILON * fmaxf(1.0f, fmaxf(fabsf(x), fabsf(y))))

typedef unsigned int MortonCode;

template <typename T>
using vec3 = typename std::conditional<std::is_same<T, float>::value, float3,
                                       double3>::type;

template <typename T>
using vec2 = typename std::conditional<std::is_same<T, float>::value, float2,
                                       double2>::type;

template <typename T>
std::ostream &operator<<(std::ostream &os, const vec3<T> &x) {
  os << x.x << ", " << x.y << ", " << x.z;
  return os;
}

std::ostream &operator<<(std::ostream &os, const vec3<float> &x) {
  os << x.x << ", " << x.y << ", " << x.z;
  return os;
}

std::ostream &operator<<(std::ostream &os, const vec3<double> &x) {
  os << x.x << ", " << x.y << ", " << x.z;
  return os;
}

template <typename T> std::ostream &operator<<(std::ostream &os, vec3<T> x) {
  os << x.x << ", " << x.y << ", " << x.z;
  return os;
}

__host__ __device__ inline double3 fmin(const double3 &a, const double3 &b) {
  return make_double3(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
}

__host__ __device__ inline double3 fmax(const double3 &a, const double3 &b) {
  return make_double3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));
}

struct is_valid_cnt : public thrust::unary_function<long2, int> {
public:
  __host__ __device__ int operator()(long2 vec) const {
    return vec.x >= 0 && vec.y >= 0;
  }
};

template <typename T>
__global__ void compute_tri_bboxes(Triangle<T> *triangles, int num_triangles,
                                   AABB<T> *bboxes) {

  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_triangles;
       idx += blockDim.x * gridDim.x) {
    bboxes[idx] = triangles[idx].bbox();
  }
  return;
}

template <typename T>
__host__ __device__ vec3<T> SatCrossEdge(const vec3<T> &a, const vec3<T> &b,
                                         const vec3<T> &c, const vec3<T> &d) {
  vec3<T> ab = b - a;
  vec3<T> cd = d - c;

  vec3<T> result = cross(ab, cd);
  // if (dot(ab, cd) > EPSILON)
  if (!CMP(dot(ab, cd), 0))
    return result;
  else {
    vec3<T> axis = cross(ab, c - a);
    result = cross(ab, axis);
    // if (dot(ab, cd) > EPSILON)
    if (!CMP(dot(result, result), 0))
      return result;
  }
  return vec3<T>();
}

template <typename T>
__device__ __host__ void point_to_barycentric(vec3<T> p, vec3<T> a, vec3<T> b,
                                              vec3<T> c, vec3<T> barycentrics) {
  vec3<T> v0 = b - a, v1 = c - a, v2 = p - a;
  T d00 = dot(v0, v0);
  T d01 = dot(v0, v1);
  T d11 = dot(v1, v1);
  T d20 = dot(v2, v0);
  T d21 = dot(v2, v1);
  T denom = d00 * d11 - d01 * d01;
  barycentrics.y = (d11 * d20 - d01 * d21) / denom;
  barycentrics.z = (d00 * d21 - d01 * d20) / denom;
  barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;
}

template <typename T>
__device__ __host__ void point_to_barycentric(vec3<T> p, vec3<T> a, vec3<T> b,
                                              vec3<T> c,
                                              vec3<T> *barycentrics) {
  vec3<T> v0 = b - a, v1 = c - a, v2 = p - a;
  T d00 = dot(v0, v0);
  T d01 = dot(v0, v1);
  T d11 = dot(v1, v1);
  T d20 = dot(v2, v0);
  T d21 = dot(v2, v1);
  T denom = d00 * d11 - d01 * d01;
  barycentrics->y = (d11 * d20 - d01 * d21) / denom;
  barycentrics->z = (d00 * d21 - d01 * d20) / denom;
  barycentrics->x = 1.0 - barycentrics->y - barycentrics->z;
}

template <typename T>
__host__ __device__ bool
ray_triangle_intersect(const vec3<T> &orig, const vec3<T> &dir,
                       const vec3<T> &v0, const vec3<T> &v1, const vec3<T> &v2,
                       T &t, vec3<T> &isect_point) {
  vec3<T> v0v1 = v1 - v0;
  vec3<T> v0v2 = v2 - v0;
  vec3<T> pvec = cross(dir, v0v2);
  T det = dot(v0v1, pvec);

  // ray and triangle are parallel if det is close to 0
  if (fabs(det) < EPSILON)
    return false;

  T invDet = 1 / det;

  vec3<T> tvec = orig - v0;
  T u = dot(tvec, pvec) * invDet;
  if (u < 0 || u > 1)
    return false;

  vec3<T> qvec = cross(tvec, v0v1);
  T v = dot(dir, qvec) * invDet;
  if (v < 0 || u + v > 1)
    return false;

  t = dot(v0v2, qvec) * invDet;
  isect_point = t * dir + orig;

  return true;
}

template <typename T>
__device__ inline vec2<T> isect_interval(const vec3<T> &sep_axis,
                                         const Triangle<T> &tri) {
  // Check the separating sep_axis versus the first point of the triangle
  T proj_distance = dot(sep_axis, tri.v0);

  vec2<T> interval;
  interval.x = proj_distance;
  interval.y = proj_distance;

  proj_distance = dot(sep_axis, tri.v1);
  interval.x = fminf(interval.x, proj_distance);
  interval.y = fmaxf(interval.y, proj_distance);

  proj_distance = dot(sep_axis, tri.v2);
  interval.x = fminf(interval.x, proj_distance);
  interval.y = fmaxf(interval.y, proj_distance);

  return interval;
}

template <typename T>
__device__ inline bool TriangleTriangleOverlap(const Triangle<T> &tri1,
                                               const Triangle<T> &tri2,
                                               const vec3<T> &sep_axis) {
  // Calculate the projected segment of each triangle on the separating
  // axis.
  vec2<T> tri1_interval = isect_interval(sep_axis, tri1);
  vec2<T> tri2_interval = isect_interval(sep_axis, tri2);

  // In order for the triangles to overlap then there must exist an
  // intersection of the two intervals
  return (tri1_interval.x <= tri2_interval.y) &&
         (tri2_interval.x <= tri1_interval.y);
}

template <typename T>
__device__ bool TriangleTriangleIsectSepAxis(const Triangle<T> &tri1,
                                             const Triangle<T> &tri2) {
  // Calculate the edges and the normal for the first triangle
  vec3<T> tri1_edge0 = tri1.v1 - tri1.v0;
  vec3<T> tri1_edge1 = tri1.v2 - tri1.v0;
  vec3<T> tri1_edge2 = tri1.v2 - tri1.v1;
  vec3<T> tri1_normal = cross(tri1_edge1, tri1_edge2);

  // Calculate the edges and the normal for the second triangle
  vec3<T> tri2_edge0 = tri2.v1 - tri2.v0;
  vec3<T> tri2_edge1 = tri2.v2 - tri2.v0;
  vec3<T> tri2_edge2 = tri2.v2 - tri2.v1;
  vec3<T> tri2_normal = cross(tri2_edge1, tri2_edge2);

  // If the triangles are coplanar then the first 11 cases are all the same,
  // since the cross product will just give us the normal vector
  vec3<T> axes[] = {
      // tri1_normal,
      // tri2_normal,
      // cross(tri1_edge0, tri2_edge0),
      // cross(tri1_edge0, tri2_edge1),
      // cross(tri1_edge0, tri2_edge2),
      // cross(tri1_edge1, tri2_edge0),
      // cross(tri1_edge1, tri2_edge1),
      // cross(tri1_edge1, tri2_edge2),
      // cross(tri1_edge2, tri2_edge0),
      // cross(tri1_edge2, tri2_edge1),
      // cross(tri1_edge2, tri2_edge2),

      // Normals
      SatCrossEdge<T>(tri1.v0, tri1.v1, tri1.v1, tri1.v2),
      SatCrossEdge<T>(tri2.v0, tri2.v1, tri2.v1, tri2.v2),

      SatCrossEdge<T>(tri1.v0, tri1.v1, tri2.v0, tri2.v1),
      SatCrossEdge<T>(tri1.v0, tri1.v1, tri2.v1, tri2.v2),
      SatCrossEdge<T>(tri1.v0, tri1.v1, tri2.v2, tri2.v0),

      SatCrossEdge<T>(tri1.v1, tri1.v2, tri2.v0, tri2.v1),
      SatCrossEdge<T>(tri1.v1, tri1.v2, tri2.v1, tri2.v2),
      SatCrossEdge<T>(tri1.v1, tri1.v2, tri2.v2, tri2.v0),

      SatCrossEdge<T>(tri1.v2, tri1.v0, tri2.v0, tri2.v1),
      SatCrossEdge<T>(tri1.v2, tri1.v0, tri2.v1, tri2.v2),
      SatCrossEdge<T>(tri1.v2, tri1.v0, tri2.v2, tri2.v0)
      // Triangles are coplanar
      // Check the axis created by the normal of the triangle and the edges of
      // both triangles.
      // cross(tri1_normal, tri1_edge0),
      // cross(tri1_normal, tri1_edge1),
      // cross(tri1_normal, tri1_edge2),
      // cross(tri1_normal, tri2_edge0),
      // cross(tri1_normal, tri2_edge1),
      // cross(tri1_normal, tri2_edge2),
  };

  bool isect_flag = false;
#pragma unroll
  for (int i = 0; i < 11; ++i) {
    // return false;
    bool out = TriangleTriangleOverlap(tri1, tri2, axes[i]);
    if (!out) {
      // if (dot(axes[i], axes[i]) > EPSILON)
      if (!CMP(dot(axes[i], axes[i]), 0))
        return false;
    }
    // isect_flag = isect_flag || ();
  }

  return true;
  return isect_flag;
}

template <typename T> struct BVHNode {
public:
  AABB<T> bbox;
  TrianglePtr<T> triangle_ptr;

  BVHNode<T> *left;
  BVHNode<T> *right;
  BVHNode<T> *parent;
  // Stores the rightmost leaf node that can be reached from the current
  // node.
  BVHNode<T> *rightmost;

  __host__ __device__ inline bool isLeaf() { return !left && !right; };

  // The index of the object contained in the node
  int idx;
};

template <typename T> using BVHNodePtr = BVHNode<T> *;

template <typename T>
__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
    bool
    checkOverlap(const AABB<T> &bbox1, const AABB<T> &bbox2) {
  return (bbox1.min_t.x <= bbox2.max_t.x) && (bbox1.max_t.x >= bbox2.min_t.x) &&
         (bbox1.min_t.y <= bbox2.max_t.y) && (bbox1.max_t.y >= bbox2.min_t.y) &&
         (bbox1.min_t.z <= bbox2.max_t.z) && (bbox1.max_t.z >= bbox2.min_t.z);
}

template <typename T>
__device__ void find_triangle_triangle_intersection_points(
    const Triangle<T> &query_triangle, const Triangle<T> &target_triangle,
    // vec3<T> *isect_point1, vec3<T> *isect_point2) {
    vec3<T> *isect1_bcs, vec3<T> *isect2_bcs) {
  // Triangle 1
  vec3<T> query_edges[] = {
      query_triangle.v1 - query_triangle.v0,
      query_triangle.v2 - query_triangle.v1,
      query_triangle.v0 - query_triangle.v2,
  };
  vec3<T> query_origins[] = {
      query_triangle.v0,
      query_triangle.v1,
      query_triangle.v2,
  };

  // Triangle 2
  vec3<T> target_edges[] = {
      target_triangle.v1 - target_triangle.v0,
      target_triangle.v2 - target_triangle.v1,
      target_triangle.v0 - target_triangle.v2,
  };
  vec3<T> target_origins[] = {
      target_triangle.v0,
      target_triangle.v1,
      target_triangle.v2,
  };

  T tmin = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;
  T tmax = -std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;

  bool found_first = false;
  bool found_second = false;

  vec3<T> isect_point, isect_point1, isect_point2;

  T t;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    // Find the current intersection point of the triangles
    bool does_intersect = ray_triangle_intersect(
        query_origins[i], query_edges[i], target_triangle.v0,
        target_triangle.v1, target_triangle.v2, t, isect_point);
    if (t > 1 || t < 0) {
      continue;
    }
    if (does_intersect) {
      if (!found_first) {
        isect_point1.x = isect_point.x;
        isect_point1.y = isect_point.y;
        isect_point1.z = isect_point.z;
        found_first = true;
        tmin = t;
      }
    }
    does_intersect = ray_triangle_intersect(
        query_origins[i] + (t + EPSILON) * query_edges[i], query_edges[i],
        target_triangle.v1, target_triangle.v1, target_triangle.v2, t,
        isect_point2);
    if (t > 1 || t < 0) {
      continue;
    }
    if (does_intersect) {
      // bool cond = found_first && t > tmin && !found_second;
      // printf("%f %f %d %d %d\n", t, tmin, found_first, found_second, cond);
      if (found_first && t > tmin && !found_second) {
        // *isect_point2 = origins[i] + t * query_edges[i];
        // *isect2_bcs = bcs;
        isect_point2.x = isect_point.x;
        isect_point2.y = isect_point.y;
        isect_point2.z = isect_point.z;
        found_second = true;
      }
    }
  }
  if (found_first) {
    point_to_barycentric<T>(isect_point1, target_triangle.v0,
                            target_triangle.v1, target_triangle.v2, isect1_bcs);
  }
  if (found_second) {
    point_to_barycentric<T>(isect_point2, target_triangle.v0,
                            target_triangle.v1, target_triangle.v2, isect2_bcs);
    return;
  }
  tmin = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;

#pragma unroll
  for (int i = 0; i < 3; ++i) {
    // Find the current intersection point of the triangles
    bool does_intersect = ray_triangle_intersect(
        target_origins[i], target_edges[i], query_triangle.v0,
        query_triangle.v1, query_triangle.v2, t, isect_point);
    if (t > 1 || t < 0) {
      continue;
    }
    if (does_intersect) {
      if (!found_first) {
        isect_point1.x = isect_point.x;
        isect_point1.y = isect_point.y;
        isect_point1.z = isect_point.z;
        tmin = t;
        found_first = true;
      }
    }

    does_intersect = ray_triangle_intersect(
        target_origins[i] + (t + EPSILON) * target_edges[i], target_edges[i],
        query_triangle.v0, query_triangle.v1, query_triangle.v2, t,
        isect_point);
    if (t > 1 || t < 0) {
      continue;
    }
    if (does_intersect) {
      if (found_first && t > tmin && !found_second) {
        isect_point2.x = isect_point.x;
        isect_point2.y = isect_point.y;
        isect_point2.z = isect_point.z;
        found_second = true;
      }
    }
  }

  if (found_first) {

    point_to_barycentric<T>(isect_point1, target_triangle.v0,
                            target_triangle.v1, target_triangle.v2, isect1_bcs);
  }
  if (found_second) {
    point_to_barycentric<T>(isect_point2, target_triangle.v0,
                            target_triangle.v1, target_triangle.v2, isect2_bcs);
    return;
  }

  if (found_first && !found_second) {
    isect2_bcs->x = isect1_bcs->x;
    isect2_bcs->y = isect1_bcs->y;
    isect2_bcs->z = isect1_bcs->z;
  }
  // printf("%f %f %f, %f %f %f\n", isect1_bcs->x, isect1_bcs->y, isect1_bcs->z,
         // isect2_bcs->x, isect2_bcs->y, isect2_bcs->z);

  return;
}

template <typename T>
__device__ int
traverse_bvh(long *collisionIndices, vec3<T> *intersection_bcs,
             BVHNodePtr<T> root, const Triangle<T> &query_triangle,
             int max_collisions, bool collision_ordering = true) {
  int num_collisions = 0;
  // Allocate traversal stack from thread-local memory,
  // and push NULL to indicate that there are no postponed nodes.
  BVHNodePtr<T> stack[STACK_SIZE];
  BVHNodePtr<T> *stackPtr = stack;
  *stackPtr++ = nullptr; // push

  const AABB<T> query_aabb = query_triangle.bbox();
  // Traverse nodes starting from the root.
  BVHNodePtr<T> node = root;
  do {
    // Check each child node for overlap.
    BVHNodePtr<T> childL = node->left;
    BVHNodePtr<T> childR = node->right;
    bool overlapL = checkOverlap<T>(query_aabb, childL->bbox);
    bool overlapR = checkOverlap<T>(query_aabb, childR->bbox);

    // Query overlaps a leaf node => report collision.
    if (overlapL && childL->isLeaf()) {
      // Append the collision to the main array
      // Increase the number of detection collisions
      // int coll_idx = atomicAdd(counter, 1);
      bool does_collide =
          TriangleTriangleIsectSepAxis(query_triangle, *childL->triangle_ptr);
      if (does_collide) {
        // Add the index of the target mesh to the collision indices
        collisionIndices[num_collisions] = childL->idx;
        // Compute the intersection points with the target mesh
        find_triangle_triangle_intersection_points(
            query_triangle, *childL->triangle_ptr,
            intersection_bcs + 2 * num_collisions,
            intersection_bcs + 2 * num_collisions + 1);
        num_collisions++;
      }
    }

    if (overlapR && childR->isLeaf()) {
      bool does_collide =
          TriangleTriangleIsectSepAxis(query_triangle, *childR->triangle_ptr);
      if (does_collide) {
        collisionIndices[num_collisions] = childR->idx;
        find_triangle_triangle_intersection_points(
            query_triangle, *childR->triangle_ptr,
            intersection_bcs + 2 * num_collisions,
            intersection_bcs + 2 * num_collisions + 1);
        num_collisions++;
      }
    }

    // Query overlaps an internal node => traverse.
    bool traverseL = (overlapL && !childL->isLeaf());
    bool traverseR = (overlapR && !childR->isLeaf());

    if (!traverseL && !traverseR) {
      node = *--stackPtr; // pop
    } else {
      node = (traverseL) ? childL : childR;
      if (traverseL && traverseR) {
        *stackPtr++ = childR; // push
      }
    }
  } while (node != nullptr);

  return num_collisions;
}

template <typename T>
__global__ void
findPotentialCollisions(long *collisionIndices, vec3<T> *intersection_bcs_ptr,
                        BVHNodePtr<T> root, BVHNodePtr<T> leaves,
                        Triangle<T> *query_triangles, int num_query_triangles,
                        int max_collisions) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < num_query_triangles; idx += blockDim.x * gridDim.x) {
    Triangle<T> query_triangle = query_triangles[idx];

    long *curr_collision_idxs = collisionIndices + idx * max_collisions;
    vec3<T> *curr_intersection_bcs_ptr =
        intersection_bcs_ptr + idx * max_collisions * 2;
    int num_collisions =
        traverse_bvh<T>(curr_collision_idxs, curr_intersection_bcs_ptr, root,
                        query_triangle, max_collisions);
  }
  return;
}

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
        MortonCode
        expandBits(MortonCode v) {
  // Shift 16
  v = (v * 0x00010001u) & 0xFF0000FFu;
  // Shift 8
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  // Shift 4
  v = (v * 0x00000011u) & 0xC30C30C3u;
  // Shift 2
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
template <typename T>
__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
        MortonCode
        morton3D(T x, T y, T z) {
  x = min(max(x * 1024.0f, 0.0f), 1023.0f);
  y = min(max(y * 1024.0f, 0.0f), 1023.0f);
  z = min(max(z * 1024.0f, 0.0f), 1023.0f);
  MortonCode xx = expandBits((MortonCode)x);
  MortonCode yy = expandBits((MortonCode)y);
  MortonCode zz = expandBits((MortonCode)z);
  return xx * 4 + yy * 2 + zz;
}

template <typename T>
__global__ void compute_morton_codes(Triangle<T> *triangles, int num_triangles,
                                     AABB<T> *scene_bb,
                                     MortonCode *morton_codes) {

  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_triangles;
       idx += blockDim.x * gridDim.x) {
    Triangle<T> tri = triangles[idx];
    vec3<T> centroid = (tri.v0 + tri.v1 + tri.v2) / (T)3.0;

    T x = (centroid.x - scene_bb->min_t.x) /
          (scene_bb->max_t.x - scene_bb->min_t.x);
    T y = (centroid.y - scene_bb->min_t.y) /
          (scene_bb->max_t.y - scene_bb->min_t.y);
    T z = (centroid.z - scene_bb->min_t.z) /
          (scene_bb->max_t.z - scene_bb->min_t.z);

    morton_codes[idx] = morton3D<T>(x, y, z);
  }
  return;
}

__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
    int
    LongestCommonPrefix(int i, int j, MortonCode *morton_codes,
                        int num_triangles, int *triangle_ids) {
  // This function will be called for i - 1, i, i + 1, so we might go beyond
  // the array limits
  if (i < 0 || i > num_triangles - 1 || j < 0 || j > num_triangles - 1)
    return -1;

  MortonCode key1 = morton_codes[i];
  MortonCode key2 = morton_codes[j];

  if (key1 == key2) {
    // Duplicate key:__clzll(key1 ^ key2) will be equal to the number of
    // bits in key[1, 2]. Add the number of leading zeros between the
    // indices
    return __clz(key1 ^ key2) + __clz(triangle_ids[i] ^ triangle_ids[j]);
  } else {
    // Keys are different
    return __clz(key1 ^ key2);
  }
}

template <typename T>
__global__ void BuildRadixTree(MortonCode *morton_codes, int num_triangles,
                               int *triangle_ids, BVHNodePtr<T> internal_nodes,
                               BVHNodePtr<T> leaf_nodes) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= num_triangles - 1)
    return;

  int delta_next = LongestCommonPrefix(idx, idx + 1, morton_codes,
                                       num_triangles, triangle_ids);
  int delta_last = LongestCommonPrefix(idx, idx - 1, morton_codes,
                                       num_triangles, triangle_ids);
  // Find the direction of the range
  int direction = delta_next - delta_last >= 0 ? 1 : -1;

  int delta_min = LongestCommonPrefix(idx, idx - direction, morton_codes,
                                      num_triangles, triangle_ids);

  // Do binary search to compute the upper bound for the length of the range
  int lmax = 2;
  while (LongestCommonPrefix(idx, idx + lmax * direction, morton_codes,
                             num_triangles, triangle_ids) > delta_min) {
    lmax *= 2;
  }

  // Use binary search to find the other end.
  int l = 0;
  int divider = 2;
  for (int t = lmax / divider; t >= 1; divider *= 2) {
    if (LongestCommonPrefix(idx, idx + (l + t) * direction, morton_codes,
                            num_triangles, triangle_ids) > delta_min) {
      l = l + t;
    }
    t = lmax / divider;
  }
  int j = idx + l * direction;

  // Find the length of the longest common prefix for the current node
  int node_delta =
      LongestCommonPrefix(idx, j, morton_codes, num_triangles, triangle_ids);
  int s = 0;
  divider = 2;
  // Search for the split position using binary search.
  for (int t = (l + (divider - 1)) / divider; t >= 1; divider *= 2) {
    if (LongestCommonPrefix(idx, idx + (s + t) * direction, morton_codes,
                            num_triangles, triangle_ids) > node_delta) {
      s = s + t;
    }
    t = (l + (divider - 1)) / divider;
  }
  // gamma in the Karras paper
  int split = idx + s * direction + min(direction, 0);

  // Assign the parent and the left, right children for the current node
  BVHNodePtr<T> curr_node = internal_nodes + idx;
  if (min(idx, j) == split) {
    curr_node->left = leaf_nodes + split;
    (leaf_nodes + split)->parent = curr_node;
  } else {
    curr_node->left = internal_nodes + split;
    (internal_nodes + split)->parent = curr_node;
  }
  if (max(idx, j) == split + 1) {
    curr_node->right = leaf_nodes + split + 1;
    (leaf_nodes + split + 1)->parent = curr_node;
  } else {
    curr_node->right = internal_nodes + split + 1;
    (internal_nodes + split + 1)->parent = curr_node;
  }
}

template <typename T>
__global__ void create_hierarchy(BVHNodePtr<T> internal_nodes,
                                 BVHNodePtr<T> leaf_nodes, int num_triangles,
                                 Triangle<T> *triangles, int *triangle_ids,
                                 int *atomic_counters) {
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_triangles;
       idx += blockDim.x * gridDim.x) {

    BVHNodePtr<T> leaf = leaf_nodes + idx;
    // Assign the index to the primitive
    leaf->idx = triangle_ids[idx];

    Triangle<T> tri = triangles[triangle_ids[idx]];
    // Assign the bounding box of the triangle to the leaves
    leaf->bbox = tri.bbox();
    leaf->rightmost = leaf;
    leaf->triangle_ptr = &triangles[triangle_ids[idx]];

    BVHNodePtr<T> curr_node = leaf->parent;
    int current_idx = curr_node - internal_nodes;

    // Increment the atomic counter
    int curr_counter = atomicAdd(atomic_counters + current_idx, 1);
    while (true) {
      // atomicAdd returns the old value at the specified address. Thus the
      // first thread to reach this point will immediately return
      if (curr_counter == 0)
        break;

      // Calculate the bounding box of the current node as the union of the
      // bounding boxes of its children.
      AABB<T> left_bb = curr_node->left->bbox;
      AABB<T> right_bb = curr_node->right->bbox;
      curr_node->bbox = left_bb + right_bb;
      // Store a pointer to the right most node that can be reached from this
      // internal node.
      curr_node->rightmost =
          curr_node->left->rightmost > curr_node->right->rightmost
              ? curr_node->left->rightmost
              : curr_node->right->rightmost;

      // If we have reached the root break
      if (curr_node == internal_nodes)
        break;

      // Proceed to the parent of the node
      curr_node = curr_node->parent;
      // Calculate its position in the flat array
      current_idx = curr_node - internal_nodes;
      // Update the visitation counter
      curr_counter = atomicAdd(atomic_counters + current_idx, 1);
    }
  }
  return;
}

template <typename T>
void buildBVH(BVHNodePtr<T> internal_nodes, BVHNodePtr<T> leaf_nodes,
              Triangle<T> *__restrict__ triangles,
              thrust::device_vector<int> *triangle_ids, int num_triangles,
              int batch_size, bool print_timings = false) {

  cudaEvent_t start, stop;
  float milliseconds = 0;
  if (print_timings) {
    // Create the CUDA events used to estimate the execution time of each
    // kernel.
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  thrust::device_vector<AABB<T>> bounding_boxes(num_triangles);

  int blockSize = NUM_THREADS;
  int gridSize = (num_triangles + blockSize - 1) / blockSize;
  if (print_timings)
    cudaEventRecord(start);

  compute_tri_bboxes<T><<<gridSize, blockSize>>>(triangles, num_triangles,
                                                 bounding_boxes.data().get());
  if (print_timings)
    cudaEventRecord(stop);

  cudaCheckError();

  if (print_timings) {
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Compute Triangle Bounding boxes = " << milliseconds << " (ms)"
              << std::endl;
  }

  if (print_timings)
    cudaEventRecord(start);
  // Compute the union of all the bounding boxes
  AABB<T> host_scene_bb = thrust::reduce(
      bounding_boxes.begin(), bounding_boxes.end(), AABB<T>(), MergeAABB<T>());
  if (print_timings)
    cudaEventRecord(stop);

  cudaCheckError();

  if (print_timings) {
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Scene bounding box reduction = " << milliseconds << " (ms)"
              << std::endl;
  }

  // TODO: Custom reduction ?
  // Copy the bounding box back to the GPU
  AABB<T> *scene_bb_ptr;
  cudaMalloc(&scene_bb_ptr, sizeof(AABB<T>));
  cudaMemcpy(scene_bb_ptr, &host_scene_bb, sizeof(AABB<T>),
             cudaMemcpyHostToDevice);

  thrust::device_vector<MortonCode> morton_codes(num_triangles);

  if (print_timings)
    cudaEventRecord(start);
  // Compute the morton codes for the centroids of all the primitives
  compute_morton_codes<T><<<gridSize, blockSize>>>(
      triangles, num_triangles, scene_bb_ptr, morton_codes.data().get());
  if (print_timings)
    cudaEventRecord(stop);

  cudaCheckError();

  if (print_timings) {
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Morton code calculation = " << milliseconds << " (ms)"
              << std::endl;
  }

  // Construct an array of triangle ids.
  thrust::sequence(triangle_ids->begin(), triangle_ids->end());

  try {
    if (print_timings)
      cudaEventRecord(start);
    thrust::sort_by_key(morton_codes.begin(), morton_codes.end(),
                        triangle_ids->begin());
    if (print_timings)
      cudaEventRecord(stop);
    if (print_timings) {
      cudaEventSynchronize(stop);
      milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      std::cout << "Morton code sorting = " << milliseconds << " (ms)"
                << std::endl;
    }
  } catch (thrust::system_error e) {
    std::cout << "Error inside sort: " << e.what() << std::endl;
  }

  if (print_timings)
    cudaEventRecord(start);
  // Construct the radix tree using the sorted morton code sequence
  BuildRadixTree<T><<<gridSize, blockSize>>>(
      morton_codes.data().get(), num_triangles, triangle_ids->data().get(),
      internal_nodes, leaf_nodes);
  if (print_timings)
    cudaEventRecord(stop);

  cudaCheckError();

  if (print_timings) {
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Building radix tree = " << milliseconds << " (ms)"
              << std::endl;
  }
  // Create an array that contains the atomic counters for each node in the
  // tree
  thrust::device_vector<int> counters(num_triangles);

  // Build the Bounding Volume Hierarchy in parallel from the leaves to the
  // root
  create_hierarchy<T><<<gridSize, blockSize>>>(
      internal_nodes, leaf_nodes, num_triangles, triangles,
      triangle_ids->data().get(), counters.data().get());

  cudaCheckError();

  if (print_timings) {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Hierarchy generation = " << milliseconds << " (ms)"
              << std::endl;
  }

  cudaFree(scene_bb_ptr);
  return;
}

void mesh_mesh_intersection_forward(const torch::Tensor &query_triangles,
                                    const torch::Tensor &target_triangles,
                                    torch::Tensor &collision_faces,
                                    torch::Tensor &collision_bcs,
                                    int max_collisions = 16,
                                    bool print_timings = false) {
  const auto batch_size = query_triangles.size(0);
  const auto num_query_triangles = query_triangles.size(1);
  const auto num_target_triangles = target_triangles.size(1);

  thrust::device_vector<int> triangle_ids(num_target_triangles);

  int blockSize = NUM_THREADS;
  int gridSize = (num_query_triangles + blockSize - 1) / blockSize;

  thrust::device_vector<long> coll_faces_vector(num_query_triangles *
                                                max_collisions);
  cudaEvent_t start, stop;
  float milliseconds = 0;
  if (print_timings) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  auto collision_faces_ptr = collision_faces.data_ptr<long>();

  // Construct the bvh tree
  AT_DISPATCH_FLOATING_TYPES(
      query_triangles.type(), "bvh_tree_building", ([&] {
        thrust::device_vector<BVHNode<scalar_t>> leaf_nodes(
            num_target_triangles);
        thrust::device_vector<BVHNode<scalar_t>> internal_nodes(
            num_target_triangles - 1);
        thrust::device_vector<vec3<scalar_t>> intersection_bcs(
            num_query_triangles * max_collisions * 2);

        auto query_triangles_float_ptr = query_triangles.data<scalar_t>();
        auto target_triangles_float_ptr = target_triangles.data<scalar_t>();

        auto collision_bcs_ptr = collision_bcs.data_ptr<scalar_t>();

        // Iterate over the batch
        for (int bidx = 0; bidx < batch_size; ++bidx) {

          Triangle<scalar_t> *curr_target_triangles_ptr =
              (TrianglePtr<scalar_t>)target_triangles_float_ptr +
              num_target_triangles * bidx;
          Triangle<scalar_t> *curr_query_triangles_ptr =
              (TrianglePtr<scalar_t>)query_triangles_float_ptr +
              num_query_triangles * bidx;

          buildBVH<scalar_t>(internal_nodes.data().get(),
                             leaf_nodes.data().get(), curr_target_triangles_ptr,
                             &triangle_ids, num_target_triangles, batch_size);

          thrust::fill(coll_faces_vector.begin(), coll_faces_vector.end(), -1);
          if (print_timings)
            cudaEventRecord(start);

          findPotentialCollisions<scalar_t><<<gridSize, blockSize>>>(
              coll_faces_vector.data().get(), intersection_bcs.data().get(),
              internal_nodes.data().get(), leaf_nodes.data().get(),
              curr_query_triangles_ptr, num_query_triangles, max_collisions);
          cudaDeviceSynchronize();

          if (print_timings)
            cudaEventRecord(stop);

          cudaCheckError();

          if (print_timings) {
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "FindPotentialCollisions = " << milliseconds << " (ms)"
                      << std::endl;
          }

          cudaMemcpy(collision_faces_ptr +
                         bidx * num_query_triangles * max_collisions,
                     (long *)coll_faces_vector.data().get(),
                     coll_faces_vector.size() * sizeof(long),
                     cudaMemcpyDeviceToDevice);
          cudaCheckError();

          cudaMemcpy(collision_bcs_ptr +
                         bidx * num_query_triangles * max_collisions * 2 * 3,
                     (scalar_t *)intersection_bcs.data().get(),
                     intersection_bcs.size() * sizeof(scalar_t) * 3,
                     cudaMemcpyDeviceToDevice);
          cudaCheckError();

          // thrust::host_vector<vec3<scalar_t>> host_vec(intersection_bcs);
          // for (int i = 0; i < host_vec.size(); i++) {
            // std::cout << host_vec[i] << std::endl;
          // }


          if (print_timings)
            cudaEventRecord(stop);

          if (print_timings) {
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "Copy CUDA array to tensor " << milliseconds << " (ms)"
                      << std::endl;
          }
        }
      }));
}
