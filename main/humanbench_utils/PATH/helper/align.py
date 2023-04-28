import cv2
import numpy as np
from skimage import transform as trans

def affine_align(img, landmark=None, **kwargs):
    M = None
    src = np.array([
     [38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041] ], dtype=np.float32 )
    # src=src * 224 / 112

    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    warped = cv2.warpAffine(img, M, (112, 112), borderValue = 0.0)
    return warped


def kestrel_get_similar_matrix(src_points, dst_points):
  if src_points.size != dst_points.size:
    print("error: the size of src_points and dst_points must be same",
      "which is {0} vs. {1}".format(src_points.size, dst_points.size))
    exit(-1)

  dst_points = dst_points.T.reshape(-1)

  point_num = src_points.shape[0]
  new_src_points = np.zeros((point_num * 2, 4))
  new_src_points[:point_num, :2] = src_points
  new_src_points[:point_num, 2] = 1
  new_src_points[:point_num, 3] = 0

  new_src_points[point_num:, 0] = src_points[:, 1]
  new_src_points[point_num:, 1] = -src_points[:, 0]
  new_src_points[point_num:, 2] = 0
  new_src_points[point_num:, 3] = 1

  min_square_solution = np.linalg.lstsq(new_src_points, dst_points,
    rcond=-1)[0]

  trans_matrix = np.array([
      [ min_square_solution[0], -min_square_solution[1], 0 ],
      [ min_square_solution[1], min_square_solution[0], 0 ],
      [ min_square_solution[2], min_square_solution[3], 1 ],
    ])

  return trans_matrix.T[:2]

def transform(pts, M):
    dst = np.matmul(pts, M[:, :2].T)
    dst[:, 0] += M[0, 2]
    dst[:, 1] += M[1, 2]
    return dst


def affine_alignSDK(img, landmark=None, borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR):
    M = None
    dst_points = np.array([[70.745156, 111.9996875], [108.23625, 111.9996875], [89.700875, 153.514375]], dtype=np.float32)
    default_shape = (178,218)
    lmk = landmark.astype(np.float32)
    src_points = np.array([
        lmk[0], lmk[1],
        (lmk[3] + lmk[4]) / 2
    ], dtype=np.float32)
    # src_points = get_trans_points(landmarks)
    trans_matrix = kestrel_get_similar_matrix(src_points, dst_points)
    trans_matrix = np.concatenate((trans_matrix, [[0, 0, 1]]), axis=0)
    # print(rotate_points_106)
    return cv2.warpPerspective(img, trans_matrix, default_shape, borderMode, flags=flags), trans_matrix
