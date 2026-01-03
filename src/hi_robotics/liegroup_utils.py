# Python Native LieGroup operations (SO3, SE3, RxSO3, Sim3) for PyTorch

# See: "A micro Lie theory for state estimation in robotics" pp. 13 for linearized optimization used in SLAM
# See: "A tutorial on SE(3) transformation parameterizations and on-manifold optimization" for derived exp/log map and its jacobians of SE3
# See: "Lie Groups for 2D and 3D Transformations" for Sim3 exp/log maps derivation

# Copy from: https://github.com/facebookresearch/pytorch3d/blob/33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba/pytorch3d/transforms/se3.py
# See: https://github.com/borglab/gtsam/blob/ef33d45aea433da506447759ec949af30dc8e38f/gtsam/geometry/Pose3.cpp

import torch

from torch.autograd.function import Function, FunctionCtx


def so3_vee(h: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse Hat operator [1] of a batch of 3x3 matrices.

    Args:
        h: Batch of skew-symmetric matrices of shape `(minibatch, 3, 3)`.

    Returns:
        Batch of 3d vectors of shape `(minibatch, 3)`.

    Raises:
        ValueError if `h` is of incorrect shape.
        ValueError if `h` not skew-symmetric.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim1, dim2 = h.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    x = (h[:, 2, 1] - h[:, 1, 2]) / 2
    y = (h[:, 0, 2] - h[:, 2, 0]) / 2
    z = (h[:, 1, 0] - h[:, 0, 1]) / 2
    return torch.stack((x, y, z), dim=1)


def so3_hat(phi: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        phi: Batch of vectors of shape `(minibatch, 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3, 3)` where each matrix is of the form:
            `[      0 -phi_z  phi_y ]
             [  phi_z      0 -phi_x ]
             [ -phi_y  phi_x      0 ]`

    Raises:
        ValueError if `phi` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = phi.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    rx, ry, rz = phi[..., 0], phi[..., 1], phi[..., 2]
    zeros = torch.zeros_like(rx)
    return torch.stack([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1).view(phi.shape + (3,)) # shape == (N,3,3)


def _get_so3_v_c(theta: torch.Tensor, eps = 1e-2):
    # torch where will cause NaN in backward if we do not handle the zero case here
    # See: https://discuss.pytorch.org/t/gradients-of-torch-where/26835
    mask0 = torch.abs(theta) < eps # TODO test
    mask1 = ~mask0
    theta0 = theta[mask0]
    theta1 = theta[mask1]
    c = torch.empty_like(theta)
    c[mask0] = 1/2 - theta0**2/24 + theta0**4/720 - theta0**6/40320 # Tayler expansion for better numeric stability
    c[mask1] = (1. - torch.cos(theta1)) / theta1**2
    return c


def so3_expmap(phi: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of rotation matrices `phi`
    to a batch of 3x3 rotation matrices using Rodrigues formula [1].

    In the logarithmic representation, each rotation matrix is represented as
    a 3-dimensional vector (`phi`) who's l2-norm and direction correspond
    to the magnitude of the rotation angle and the axis of rotation respectively.

    Args:
        phi: Batch of vectors of shape `(minibatch, 3)`.

    Returns:
        Batch of rotation matrices of shape `(minibatch, 3, 3)`.

    Raises:
        ValueError if `phi` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    _, dim = phi.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    theta = torch.norm(phi, dim=-1)

    b = torch.sinc(theta / torch.pi)
    c = _get_so3_v_c(theta)
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).unsqueeze(0)
    W = so3_hat(phi)
    WW = W @ W

    return I + b * W + c * WW


def so3_logmap(R: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    Convert a batch of 3x3 rotation matrices `R`
    to a batch of 3-dimensional matrix logarithms of rotation matrices
    The conversion has a singularity around `(R=I)`.

    Args:
        R: batch of rotation matrices of shape `(minibatch, 3, 3)`.

    Returns:
        Batch of logarithms of input rotation matrices
        of shape `(minibatch, 3)`.
    """
    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    if R.size(-1) != 3 or R.size(-2) != 3:
        raise ValueError(f"Invalid rotation R shape {R.shape}.")

    # See:
    # https://dfki-ric.github.io/pytransform3d/_modules/pytransform3d/rotations/_matrix.html#quaternion_from_matrix
    # Source:
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions
    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    mask0 = trace > 0.0
    mask1 = ~mask0 & ((R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]))
    mask2 = ~mask0 & ~mask1 & (R[:, 1, 1] > R[:, 2, 2])
    mask3 = ~mask0 & ~mask1 & ~mask2

    qw = torch.empty_like(trace)
    qx = torch.empty_like(trace)
    qy = torch.empty_like(trace)
    qz = torch.empty_like(trace)

    div = torch.sqrt(1.0 + R[mask0, 0, 0] + R[mask0, 1, 1] + R[mask0, 2, 2])
    qw[mask0] = 0.5 * div
    qx[mask0] = 0.5 / div * (R[mask0, 2, 1] - R[mask0, 1, 2])
    qy[mask0] = 0.5 / div * (R[mask0, 0, 2] - R[mask0, 2, 0])
    qz[mask0] = 0.5 / div * (R[mask0, 1, 0] - R[mask0, 0, 1])

    div = torch.sqrt(1.0 + R[mask1, 0, 0] - R[mask1, 1, 1] - R[mask1, 2, 2])
    qw[mask1] = 0.5 / div * (R[mask1, 2, 1] - R[mask1, 1, 2])
    qx[mask1] = 0.5 * div
    qy[mask1] = 0.5 / div * (R[mask1, 1, 0] + R[mask1, 0, 1])
    qz[mask1] = 0.5 / div * (R[mask1, 0, 2] + R[mask1, 2, 0])

    div = torch.sqrt(1.0 + R[mask2, 1, 1] - R[mask2, 0, 0] - R[mask2, 2, 2])
    qw[mask2] = 0.5 / div * (R[mask2, 0, 2] - R[mask2, 2, 0])
    qx[mask2] = 0.5 / div * (R[mask2, 1, 0] + R[mask2, 0, 1])
    qy[mask2] = 0.5 * div
    qz[mask2] = 0.5 / div * (R[mask2, 2, 1] + R[mask2, 1, 2])

    div = torch.sqrt(1.0 + R[mask3, 2, 2] - R[mask3, 0, 0] - R[mask3, 1, 1])
    qw[mask3] = 0.5 / div * (R[mask3, 1, 0] - R[mask3, 0, 1])
    qx[mask3] = 0.5 / div * (R[mask3, 0, 2] + R[mask3, 2, 0])
    qy[mask3] = 0.5 / div * (R[mask3, 2, 1] + R[mask3, 1, 2])
    qz[mask3] = 0.5 * div

    qvec = torch.stack((qx, qy, qz), dim=-1)
    qvec_norm = torch.norm(qvec, dim=-1)
    # theta = 2. * torch.atan2(qvec_norm, qw)

    # See: https://github.com/strasdat/Sophus/blob/main/sophus/so3.hpp
    # Atan-based log thanks to C. Hertzberg et al.: "Integrating Generic Sensor Fusion Algorithms with Sound State Representation through Encapsulation of Manifolds" Information Fusion, 2011
    maskkk0 = torch.abs(qvec_norm) < eps
    maskkk10 = ~maskkk0 & (qw < 0)
    maskkk11 = ~maskkk0 & ~maskkk10
    qw0 = qw[maskkk0]
    qw10 = qw[maskkk10]
    qw11 = qw[maskkk11]
    qvec_norm0 = qvec_norm[maskkk0]
    qvec_norm10 = qvec_norm[maskkk10]
    qvec_norm11 = qvec_norm[maskkk11]

    qvec_ratio = torch.empty_like(qvec_norm)

    qvec_ratio[maskkk0] = 2/qw0 - 2*qvec_norm0**2/(3*qw0**3) + 2*qvec_norm0**4/(5*qw0**5) - 2*qvec_norm0**6/(7*qw0**7)
    # w < 0 ==> cos(theta/2) < 0 ==> theta > pi
    #
    # By convention, the condition |theta| < pi is imposed by wrapping theta
    # to pi; The wrap operation can be folded inside evaluation of atan2
    #
    # theta - pi = atan(sin(theta - pi), cos(theta - pi))
    #            = atan(-sin(theta), -cos(theta))
    #
    qvec_ratio[maskkk10] = 2. * torch.atan2(-qvec_norm10, -qw10) / qvec_norm10
    qvec_ratio[maskkk11] = 2. * torch.atan2(qvec_norm11, qw11) / qvec_norm11

    return qvec_ratio * qvec


def _get_se3_v_c(theta: torch.Tensor, eps = 1e-2):
    mask0 = torch.abs(theta) < eps
    mask1 = ~mask0
    theta0 = theta[mask0]
    theta1 = theta[mask1]
    c = torch.empty_like(theta)
    c[mask0] = 1/6 - theta0**2/120 + theta0**4/5040 - theta0**6/362880 # Taylor expansion for better numeric stability
    c[mask1] = (1. - torch.sin(theta1) / theta1) / theta1**2
    return c


# Copy from: https://github.com/princeton-vl/lietorch/blob/e7df86554156b36846008d8ddbcc4d8521a16554/lietorch/include/rxso3.h
# See: https://github.com/borglab/gtsam/blob/ef33d45aea433da506447759ec949af30dc8e38f/gtsam/geometry/Similarity3.cpp
# See: https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf


def _get_se3_v_abc(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # See: "Lie Groups for 2D and 3D Transformations" pp. 10
    a = torch.tensor(1., dtype=theta.dtype, device=theta.device).unsqueeze(0)
    b = _get_so3_v_c(theta)
    c = _get_se3_v_c(theta)
    return a, b, c


def _get_v_inv_def(theta: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # See: https://www.notion.so/hilookas/2d93c52e9ba58091ba6def60f248d132
    d = 1 / a
    e = - b / ((a - theta**2 * c)**2 + theta**2 * b**2)
    f = (b**2 + theta**2 * c**2 - a * c) / (a * ((a - theta**2 * c)**2 + theta**2 * b**2))
    return d, e, f


def se3_expmap(log_transform: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of SE(3) matrices `log_transform`
    to a batch of 4x4 SE(3) matrices using the exponential map.
    See e.g. [1], Sec 9.4.2. for more detailed description.

    A SE(3) matrix has the following form:
        ```
        [ R T ]
        [ 0 1 ] ,
        ```
    where `R` is a 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.

    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[phi | log_translation]`,
    i.e. a concatenation of two 3D vectors `phi` and `log_translation`.

    The conversion from the 6D representation to a 4x4 SE(3) matrix `transform`
    is done as follows:
        ```
        transform = exp( [ so3_hat(phi) log_translation ]
                         [   0 1 ] ) ,
        ```
    where `exp` is the matrix exponential and `hat` is the Hat operator [2].

    Note that for any `log_transform` with `0 <= ||phi|| < 2pi`
    (i.e. the rotation angle is between 0 and 2pi), the following identity holds:
    ```
    se3_logmap(se3_exponential_map(log_transform)) == log_transform
    ```

    Args:
        log_transform: Batch of vectors of shape `(minibatch, 6)`.

    Returns:
        Batch of transformation matrices of shape `(minibatch, 4, 4)`.

    Raises:
        ValueError if `log_transform` is of incorrect shape.

    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    if log_transform.ndim != 2 or log_transform.shape[1] != 6:
        raise ValueError("Expected input to be of shape (N, 6).")

    N, _ = log_transform.shape

    log_translation = log_transform[..., 3:]
    phi = log_transform[..., :3]

    R = so3_expmap(phi)

    # A helper function that computes the "V" matrix from [1], Sec 9.4.2. [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf

    theta = torch.norm(phi, dim=-1) # shape == (B,)

    a, b, c = _get_se3_v_abc(theta)
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).unsqueeze(0)
    W = so3_hat(phi)
    WW = W @ W
    V = a * I + b * W + c * WW # shape == (B,3,3)

    # translation is V @ T
    T = (V @ log_translation.unsqueeze(-1)).squeeze(-1)

    transform = torch.zeros(
        N, 4, 4, dtype=log_transform.dtype, device=log_transform.device
    )

    transform[:, :3, :3] = R
    transform[:, :3, 3] = T
    transform[:, 3, 3] = 1.0

    return transform


def se3_logmap(transform: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 4x4 transformation matrices `transform`
    to a batch of 6-dimensional SE(3) logarithms of the SE(3) matrices.
    See e.g. [1], Sec 9.4.2. for more detailed description.

    A SE(3) matrix has the following form:
        ```
        [ R T ]
        [ 0 1 ] ,
        ```
    where `R` is an orthonormal 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.

    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[phi | log_translation]`,
    i.e. a concatenation of two 3D vectors `phi` and `log_translation`.

    The conversion from the 4x4 SE(3) matrix `transform` to the
    6D representation `log_transform = [phi | log_translation]`
    is done as follows:
        ```
        log_transform = log(transform)
        phi = so3_vee(log_transform[:3, :3])
        log_translation = log_transform[:3, 3]
        ```
    where `log` is the matrix logarithm
    and `inv_hat` is the inverse of the Hat operator [2].

    Note that for any valid 4x4 `transform` matrix, the following identity holds:
    ```
    se3_expmap(se3_logmap(transform)) == transform
    ```

    Args:
        transform: batch of SE(3) matrices of shape `(minibatch, 4, 4)`.

    Returns:
        Batch of logarithms of input SE(3) matrices
        of shape `(minibatch, 6)`.

    Raises:
        ValueError if `transform` is of incorrect shape.
        ValueError if `R` has an unexpected trace.

    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    if transform.ndim != 3:
        raise ValueError("Input tensor shape has to be (N, 4, 4).")

    N, dim1, dim2 = transform.shape
    if dim1 != 4 or dim2 != 4:
        raise ValueError("Input tensor shape has to be (N, 4, 4).")

    if not torch.allclose(transform[:, 3, :3], torch.zeros_like(transform[:, 3, :3])):
        raise ValueError("All elements of `transform[:, 3, :3]` should be 0.")

    # log_rot is just so3_logmap of the upper left 3x3 block
    R = transform[:, :3, :3]
    phi = so3_logmap(R)

    # log_translation is V^-1 @ T
    T = transform[:, :3, 3]

    theta = torch.norm(phi, dim=-1)

    d, e, f = _get_v_inv_def(theta, *_get_se3_v_abc(theta))
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).unsqueeze(0)
    W = so3_hat(phi)
    WW = W @ W
    V_inv = d * I + e * W + f * WW

    log_translation = (V_inv @ T.unsqueeze(-1)).squeeze(-1)

    return torch.cat((phi, log_translation), dim=1)


def se3_inv(A: torch.Tensor) -> torch.Tensor:
    """
    se3_inv

    Faster version of torch.linalg.inv for SE3 matrices.
    ( R t )^-1   ( R^T -R^T*t )
    ( 0 1 )    = ( 0    1     )

    :param A: A.shape == (..., 4, 4)
    :type A: torch.Tensor
    :return: A_inv.shape == (..., 4, 4)
    :rtype: Tensor
    """

    R = A[:, :3, :3]
    t = A[:, :3, 3]
    R_inv = R.permute(0, 2, 1)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)
    A_inv = torch.zeros_like(A)
    A_inv[:, :3, :3] = R_inv
    A_inv[:, :3, 3] = t_inv
    A_inv[:, 3, 3] = 1.0
    return A_inv


def sim3_inv(A: torch.Tensor, eps = 1e-8) -> torch.Tensor:
    """
    sim3_inv

    Faster version of torch.linalg.inv for Sim3 matrices (similarity transformation).
    Sim3矩阵结构：(sR  t)，其中s是缩放因子，R是3x3正交旋转矩阵，t是3维平移向量
                  (0^T 1)
    逆矩阵公式：  ( (1/s)R^T   -R^T*t/s )
                  ( 0^T        1        )

    :param A: Sim3矩阵，shape == (..., 4, 4)
    :type A: torch.Tensor
    :return: Sim3逆矩阵，shape == (..., 4, 4)
    :rtype: Tensor
    """
    sR = A[..., :3, :3]  # (..., 3, 3)  s*R部分
    t = A[..., :3, 3]    # (..., 3)     平移向量

    norm_sR = torch.norm(sR, dim=(-2, -1))  # (...,) 计算每个sR的Frobenius范数
    s = norm_sR / torch.sqrt(torch.tensor(3.0, device=sR.device, dtype=sR.dtype))

    R = sR / (s[..., None, None] + eps)  # (..., 3, 3)  还原纯旋转矩阵
    R_inv = R.permute(*tuple(range(R.ndim-2)), -1, -2)  # (..., 3, 3)  R^T，适配任意批量维度

    inv_s = 1.0 / (s + eps)
    sR_inv = inv_s[..., None, None] * R_inv  # (..., 3, 3)

    t_inv = -torch.matmul(sR_inv, t[..., None])[..., 0]  # (..., 3)

    A_inv = torch.zeros_like(A)
    A_inv[..., :3, :3] = sR_inv
    A_inv[..., :3, 3] = t_inv
    A_inv[..., 3, 3] = 1.0

    return A_inv


# Copy from: https://github.com/princeton-vl/lietorch/blob/e7df86554156b36846008d8ddbcc4d8521a16554/lietorch/include/rxso3.h
# Which is copied from: https://github.com/strasdat/Sophus/blob/d0b7315a0d90fc6143defa54596a3a95d9fa10ec/sophus/so3.hpp
# MAGIC CODE!
# See: https://github.com/borglab/gtsam/blob/ef33d45aea433da506447759ec949af30dc8e38f/gtsam/geometry/Similarity3.cpp


def _get_sim3_v_abc(theta: torch.Tensor, sigma: torch.Tensor, eps = 1e-2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Not same as "Lie Groups for 2D and 3D Transformations" pp. 22
    # Different defination of sigma
    # See: https://www.notion.so/hilookas/2d93c52e9ba58091ba6def60f248d132

    mask0_sigma = torch.abs(sigma) < eps
    mask1_sigma = ~mask0_sigma
    mask0_theta = torch.abs(theta) < eps
    mask1_theta = ~mask0_theta

    mask00 = mask0_sigma & mask0_theta
    mask01 = mask0_sigma & mask1_theta
    mask10 = mask1_sigma & mask0_theta
    mask11 = mask1_sigma & mask1_theta

    sigma00 = sigma[mask00]
    sigma01 = sigma[mask01]
    sigma10 = sigma[mask10]
    sigma11 = sigma[mask11]

    theta00 = theta[mask00]
    theta01 = theta[mask01]
    theta10 = theta[mask10]
    theta11 = theta[mask11]

    a = torch.empty_like(sigma)
    b = torch.empty_like(sigma)
    c = torch.empty_like(sigma)

    a[mask00] = 1 + sigma00/2 + sigma00**2/6 + sigma00**3/24 + sigma00**4/120 + sigma00**5/720 + sigma00**6/5040 + sigma00**7/40320
    b[mask00] = 1/2 - theta00**2/24 + sigma00/3 - sigma00*theta00**2/30 + sigma00**2/8 - sigma00**2*theta00**2/72 + sigma00**3/30 - sigma00**3*theta00**2/252
    c[mask00] = 1/6 - theta00**2/120 + sigma00/8 - sigma00*theta00**2/144 + sigma00**2/20 - sigma00**2*theta00**2/336 + sigma00**3/72 - sigma00**3*theta00**2/1152

    a[mask01] = 1 + sigma01/2 + sigma01**2/6 + sigma01**3/24 + sigma01**4/120 + sigma01**5/720 + sigma01**6/5040 + sigma01**7/40320
    b[mask01] = theta01**(-2) - torch.cos(theta01)/theta01**2 + sigma01*(-torch.cos(theta01)/theta01**2 + torch.sin(theta01)/theta01**3) + sigma01**2*(-torch.cos(theta01)/(2*theta01**2) + torch.sin(theta01)/theta01**3 + torch.cos(theta01)/theta01**4 - 1/theta01**4) + sigma01**3*(-torch.cos(theta01)/(6*theta01**2) + torch.sin(theta01)/(2*theta01**3) + torch.cos(theta01)/theta01**4 - torch.sin(theta01)/theta01**5)
    c[mask01] = -torch.sin(theta01)/theta01**3 + theta01**(-2) + sigma01*(1/(2*theta01**2) - torch.sin(theta01)/theta01**3 - torch.cos(theta01)/theta01**4 + theta01**(-4)) + sigma01**2*(1/(6*theta01**2) - torch.sin(theta01)/(2*theta01**3) - torch.cos(theta01)/theta01**4 + torch.sin(theta01)/theta01**5) + sigma01**3*(1/(24*theta01**2) - torch.sin(theta01)/(6*theta01**3) - torch.cos(theta01)/(2*theta01**4) + torch.sin(theta01)/theta01**5 + torch.cos(theta01)/theta01**6 - 1/theta01**6)

    a[mask10] = torch.expm1(sigma10) / sigma10
    b[mask10] = sigma10**(-2) - torch.exp(sigma10)/sigma10**2 + torch.exp(sigma10)/sigma10 + theta10**2*(-torch.exp(sigma10)/(6*sigma10) + torch.exp(sigma10)/(2*sigma10**2) - torch.exp(sigma10)/sigma10**3 + torch.exp(sigma10)/sigma10**4 - 1/sigma10**4)
    c[mask10] = (sigma10**5*torch.exp(sigma10)/2 - sigma10**4*torch.exp(sigma10) + sigma10**3*torch.exp(sigma10) - sigma10**3)/sigma10**6 + theta10**2*(-torch.exp(sigma10)/(24*sigma10) + torch.exp(sigma10)/(6*sigma10**2) - torch.exp(sigma10)/(2*sigma10**3) + torch.exp(sigma10)/sigma10**4 - torch.exp(sigma10)/sigma10**5 + sigma10**(-5))

    a[mask11] = torch.expm1(sigma11) / sigma11
    b[mask11] = (torch.exp(sigma11) * (sigma11 * torch.sin(theta11) - theta11 * torch.cos(theta11)) + theta11) / (theta11 * theta11 + sigma11 * sigma11) / theta11
    c[mask11] = (torch.expm1(sigma11) / sigma11 - (torch.exp(sigma11) * (sigma11 * torch.cos(theta11) + theta11 * torch.sin(theta11)) - sigma11) / (theta11 * theta11 + sigma11 * sigma11)) / (theta11 * theta11)

    return a, b, c


def sim3_expmap(v: torch.Tensor):
    N, dim1 = v.shape
    if dim1 != 7:
        raise ValueError("Input tensor shape has to be Nx7.")

    sigma = v[:,0] # shape == (B,)
    phi = v[:,1:4] # shape == (B,3)
    rho = v[:,4:7] # shape == (B,3)

    R = so3_expmap(phi) # shape == (B,3,3)

    theta = torch.norm(phi, dim=-1) # shape == (B,)

    a, b, c = _get_sim3_v_abc(theta, sigma)
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).unsqueeze(0)
    W = so3_hat(phi)
    WW = W @ W
    V = a * I + b * W + c * WW # shape == (B,3,3)

    t = (V @ rho.unsqueeze(-1)).squeeze(-1) # shape == (B,3)
    s = torch.exp(sigma) # shape == (B,)

    T = torch.zeros((N, 4, 4), dtype=v.dtype, device=v.device)
    T[:, :3, :3] = R * s.unsqueeze(-1).unsqueeze(-1)
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0
    return T


def sim3_logmap(T: torch.Tensor):
    # To get the logmap, calculate phi and sigma, then solve for u as shown by Ethan at
    # See: https://www.ethaneade.org/latex2html/lie/node29.html
    # See: https://www.ethaneade.com/lie.pdf
    # See: https://www.cis.upenn.edu/~jean/interp-SIM.pdf
    s = torch.norm(T[:, :3, :3], dim=(1,2)) / torch.sqrt(torch.tensor(3.0, dtype=T.dtype, device=T.device).unsqueeze(0)) # shape == (B,)
    R = T[:, :3, :3] / s.unsqueeze(-1).unsqueeze(-1) # shape == (B,3,3)
    t = T[:, :3, 3] # shape == (B,3)

    sigma = torch.log(s) # shape == (B,)
    phi = so3_logmap(R) # shape == (B,3)

    theta = torch.norm(phi, dim=-1) # shape == (B,)

    d, e, f = _get_v_inv_def(theta, *_get_sim3_v_abc(theta, sigma))
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).unsqueeze(0)
    W = so3_hat(phi)
    WW = W @ W
    V_inv = d * I + e * W + f * WW

    rho = (V_inv @ t.unsqueeze(-1)).squeeze(-1) # shape == (B,3)

    v = torch.concat([ sigma.unsqueeze(-1), phi, rho ], dim=-1)
    return v

