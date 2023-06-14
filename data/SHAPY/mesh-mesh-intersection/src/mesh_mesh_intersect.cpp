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
#include <vector>

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)


void mesh_mesh_intersection_forward(const torch::Tensor &query_triangles,
                                    const torch::Tensor &target_triangles,
                                    torch::Tensor &collision_faces,
                                    torch::Tensor &collision_bcs,
                                    int max_collisions = 16,
                                    bool print_timings = false);

std::vector<torch::Tensor>
mesh_to_mesh_intersection(torch::Tensor query_triangles,
                          torch::Tensor target_triangles,
                          int max_collisions = 16, bool print_timings = false) {
  CHECK_INPUT(query_triangles);
  CHECK_INPUT(target_triangles);
  torch::Tensor collision_faces =
      -1 * torch::ones({query_triangles.size(0),
                        query_triangles.size(1) * max_collisions},
                       torch::device(query_triangles.device())
                           .dtype(torch::ScalarType::Long));
  torch::Tensor collision_bcs = torch::zeros(
      {query_triangles.size(0), query_triangles.size(1) * max_collisions, 2, 3},
      torch::device(query_triangles.device()).dtype(query_triangles.dtype()));

  mesh_mesh_intersection_forward(query_triangles, target_triangles,
                                 collision_faces, collision_bcs,
                                 max_collisions);

  return {torch::autograd::make_variable(collision_faces, false),
          torch::autograd::make_variable(collision_bcs, false)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mesh_to_mesh_forward", &mesh_to_mesh_intersection,
        "BVH mesh-to-mesh intersection forward (CUDA)",
        py::arg("query_triangles"), py::arg("target_triangles"),
        py::arg("max_collisions") = 16, py::arg("print_timings") = false);
}
