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
    return torch.where(torch.abs(theta) < eps, # TODO test
        1/2 - theta**2/24 + theta**4/720 - theta**6/40320, # Tayler expansion for better numeric stability
        (1. - torch.cos(theta)) / theta**2,
    )


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


def test_so3_expmap():
    assert (so3_expmap(torch.tensor([[0., 0., 0.]])) == torch.eye(3).unsqueeze(0)).all()  # Should be identity
    # print(so3_expmap(torch.tensor([[0., 0., 1e-13]])))

    # import ipdb; ipdb.set_trace()

test_so3_expmap()


def so3_logmap(R: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
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
    div = torch.where(trace > 0.0,
        torch.sqrt(1.0 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]),
        torch.where((R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]),
            torch.sqrt(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]),
            torch.where(R[:, 1, 1] > R[:, 2, 2],
                torch.sqrt(1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]),
                torch.sqrt(1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]),
            )
        )
    )
    qw = torch.where(trace > 0.0,
        0.5 * div,
        torch.where((R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]),
            0.5 / div * (R[:, 2, 1] - R[:, 1, 2]),
            torch.where(R[:, 1, 1] > R[:, 2, 2],
                0.5 / div * (R[:, 0, 2] - R[:, 2, 0]),
                0.5 / div * (R[:, 1, 0] - R[:, 0, 1]),
            )
        )
    )
    qx = torch.where(trace > 0.0,
        0.5 / div * (R[:, 2, 1] - R[:, 1, 2]),
        torch.where((R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]),
            0.5 * div,
            torch.where(R[:, 1, 1] > R[:, 2, 2],
                0.5 / div * (R[:, 1, 0] + R[:, 0, 1]),
                0.5 / div * (R[:, 0, 2] + R[:, 2, 0]),
            )
        )
    )
    qy = torch.where(trace > 0.0,
        0.5 / div * (R[:, 0, 2] - R[:, 2, 0]),
        torch.where((R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]),
            0.5 / div * (R[:, 1, 0] + R[:, 0, 1]),
            torch.where(R[:, 1, 1] > R[:, 2, 2],
                0.5 * div,
                0.5 / div * (R[:, 2, 1] + R[:, 1, 2]),
            )
        )
    )
    qz = torch.where(trace > 0.0,
        0.5 / div * (R[:, 1, 0] - R[:, 0, 1]),
        torch.where((R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]),
            0.5 / div * (R[:, 0, 2] + R[:, 2, 0]),
            torch.where(R[:, 1, 1] > R[:, 2, 2],
                0.5 / div * (R[:, 2, 1] + R[:, 1, 2]),
                0.5 * div,
            )
        )
    )
    qvec = torch.stack((qx, qy, qz), dim=-1)
    qvec_norm = torch.norm(qvec, dim=-1, keepdim=True)
    # theta = 2. * torch.atan2(qvec_norm, qw)

    # See:
    # https://github.com/strasdat/Sophus/blob/main/sophus/so3.hpp
    # Atan-based log thanks to
    # C. Hertzberg et al.: "Integrating Generic Sensor Fusion Algorithms with Sound State Representation through Encapsulation of Manifolds" Information Fusion, 2011
    return torch.where(torch.abs(qvec_norm) < eps,
        (2/qw - 2*qvec_norm**2/(3*qw**3)) * qvec,
        2. * torch.atan2(qvec_norm, qw) / qvec_norm * qvec,
    )

so3_logmap(so3_expmap(torch.tensor([[0., 0., torch.pi - 1e-7]])))


def plot_so3_singularity_near_pi():
    # Copy from: https://github.com/dfki-ric/pytransform3d/issues/43
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 10))
    diffs = np.logspace(-13, -1, 101)
    eps_candidates = np.logspace(-10, -3, 9)
    for plot_idx, eps in enumerate(eps_candidates):
        ax = plt.subplot(3, 3, plot_idx + 1)
        axis_dists = []
        for diff in diffs:
            theta = np.pi - diff
            a = np.array([np.sqrt(1/3) * theta, np.sqrt(1/3) * theta, np.sqrt(1/3) * theta], dtype=np.float32)
            R = so3_expmap(torch.tensor(a).unsqueeze(0))
            a2 = so3_logmap(R, eps).squeeze(0).numpy()
            print(a, a2)
            axis_dist = np.linalg.norm((a - a2)) / theta
            axis_dists.append(axis_dist)
        plt.plot(diffs, axis_dists, label="eps = %g" % eps)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim((10e-16, 10e-1))
        ax.legend()
    plt.tight_layout()
    plt.savefig("so3_singularity_near_pi.png")

plot_so3_singularity_near_pi()


def plot_so3_singularity_near_zero():
    # Copy from: https://github.com/dfki-ric/pytransform3d/issues/43
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 10))
    diffs = np.logspace(-13, -1, 101)
    eps_candidates = np.logspace(-10, -3, 9)
    for plot_idx, eps in enumerate(eps_candidates):
        ax = plt.subplot(3, 3, plot_idx + 1)
        axis_dists = []
        for diff in diffs:
            theta = diff
            a = np.array([np.sqrt(1/3) * theta, np.sqrt(1/3) * theta, np.sqrt(1/3) * theta], dtype=np.float32)
            R = so3_expmap(torch.tensor(a).unsqueeze(0))
            a2 = so3_logmap(R, eps).squeeze(0).numpy()
            axis_dist = np.linalg.norm((a - a2)) / theta
            axis_dists.append(axis_dist)
        plt.plot(diffs, axis_dists, label="eps = %g" % eps)
        ax.set_xscale("log")
        ax.set_yscale("log")
        # ax.set_ylim((10e-16, 10e-1))
        ax.legend()
    plt.tight_layout()
    plt.savefig("so3_singularity_near_zero.png")

# plot_so3_singularity_near_zero()


def test_so3_logmap():
    assert (so3_logmap(torch.eye(3).unsqueeze(0)) == torch.zeros(3).unsqueeze(0)).all()  # Should be close to zero vector
    import numpy as np

    for diff in np.logspace(-13, -1, 101):
        theta = np.pi - diff
        a = np.array([np.sqrt(1/3) * theta, np.sqrt(1/3) * theta, np.sqrt(1/3) * theta], dtype=np.float64)
        R = so3_expmap(torch.tensor(a).unsqueeze(0))
        a2 = so3_logmap(R).squeeze(0).numpy()
        axis_dist = np.linalg.norm((a - a2)) / theta
        assert axis_dist < 1e-7, f"axis_dist={axis_dist}, diff={diff}"

    for diff in np.logspace(-13, -1, 101):
        theta = np.pi - diff
        a = np.array([np.sqrt(1/3) * theta, np.sqrt(1/3) * theta, np.sqrt(1/3) * theta], dtype=np.float32)
        R = so3_expmap(torch.tensor(a).unsqueeze(0))
        a2 = so3_logmap(R).squeeze(0).numpy()
        axis_dist = np.linalg.norm((a - a2)) / theta
        assert axis_dist < 1e-3

    for diff in np.logspace(-13, -1, 101):
        theta = diff
        a = np.array([np.sqrt(1/3) * theta, np.sqrt(1/3) * theta, np.sqrt(1/3) * theta], dtype=np.float64)
        R = so3_expmap(torch.tensor(a).unsqueeze(0))
        a2 = so3_logmap(R).squeeze(0).numpy()
        axis_dist = np.linalg.norm((a - a2)) / theta
        assert axis_dist < 1e-7

    for diff in np.logspace(-13, -1, 101):
        theta = diff
        a = np.array([np.sqrt(1/3) * theta, np.sqrt(1/3) * theta, np.sqrt(1/3) * theta], dtype=np.float32)
        R = so3_expmap(torch.tensor(a).unsqueeze(0))
        a2 = so3_logmap(R).squeeze(0).numpy()
        axis_dist = np.linalg.norm((a - a2)) / theta
        assert axis_dist < 1e-6

    # Test cases for edge cases
    assert (so3_logmap(torch.tensor([[[-1,0,0],[0,-1,0],[0,0,1.]]])) == torch.tensor([[0,0,np.pi]])).all()  # 180 degree rotation around z axis
    assert (so3_logmap(torch.tensor([[[1,0,0],[0,-1,0],[0,0,-1.]]])) == torch.tensor([[np.pi,0,0]])).all()  # 180 degree rotation around z axis

# test_so3_logmap()


def _get_se3_v_c(theta: torch.Tensor, eps = 1e-2):
    return torch.where(torch.abs(theta) < eps,
        1/6 - theta**2/120 + theta**4/5040 - theta**6/362880, # Taylor expansion for better numeric stability
        (1. - torch.sin(theta) / theta) / theta**2
    )


# Plot _get_se3_v_c vs taylor
def plot__get_se3_v_c():
    import numpy as np
    import matplotlib.pyplot as plt

    thetas = np.logspace(-13, 0, 101).astype(np.float32)
    # non_taylor - taylor
    diff = np.abs(_get_se3_v_c(torch.tensor(thetas), eps=0) - _get_se3_v_c(torch.tensor(thetas), eps=100)).numpy()

    print(torch.tensor(thetas))
    print(_get_se3_v_c(torch.tensor(thetas), eps=0))
    print(_get_se3_v_c(torch.tensor(thetas), eps=100))

    plt.figure()
    plt.plot(thetas, diff, label="_get_se3_v_c - taylor")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim((1e-20, 1e-1))
    plt.savefig("_get_se3_v_c_vs_taylor.png")

# plot__get_se3_v_c()


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


def sim3_inv(A: torch.Tensor) -> torch.Tensor:
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

    EPS = 1e-8

    R = sR / (s[..., None, None] + EPS)  # (..., 3, 3)  还原纯旋转矩阵
    R_inv = R.permute(*tuple(range(R.ndim-2)), -1, -2)  # (..., 3, 3)  R^T，适配任意批量维度

    inv_s = 1.0 / (s + EPS)
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

    a = torch.where(torch.abs(sigma) < eps,
        1 + sigma/2 + sigma**2/6 + sigma**3/24 + sigma**4/120 + sigma**5/720 + sigma**6/5040 + sigma**7/40320,
        torch.expm1(sigma) / sigma,
    )
    b = torch.where(torch.abs(sigma) < eps,
        torch.where(torch.abs(theta) < eps,
            1/2 - theta**2/24 + sigma/3 - sigma*theta**2/30 + sigma**2/8 - sigma**2*theta**2/72 + sigma**3/30 - sigma**3*theta**2/252,
            theta**(-2) - torch.cos(theta)/theta**2 + sigma*(-torch.cos(theta)/theta**2 + torch.sin(theta)/theta**3) + sigma**2*(-torch.cos(theta)/(2*theta**2) + torch.sin(theta)/theta**3 + torch.cos(theta)/theta**4 - 1/theta**4) + sigma**3*(-torch.cos(theta)/(6*theta**2) + torch.sin(theta)/(2*theta**3) + torch.cos(theta)/theta**4 - torch.sin(theta)/theta**5),
        ),
        torch.where(torch.abs(theta) < eps,
            sigma**(-2) - torch.exp(sigma)/sigma**2 + torch.exp(sigma)/sigma + theta**2*(-torch.exp(sigma)/(6*sigma) + torch.exp(sigma)/(2*sigma**2) - torch.exp(sigma)/sigma**3 + torch.exp(sigma)/sigma**4 - 1/sigma**4),
            (torch.exp(sigma) * (sigma * torch.sin(theta) - theta * torch.cos(theta)) + theta) / (theta * theta + sigma * sigma) / theta,
        ),
    )
    c = torch.where(torch.abs(sigma) < eps,
        torch.where(torch.abs(theta) < eps,
            1/6 - theta**2/120 + sigma/8 - sigma*theta**2/144 + sigma**2/20 - sigma**2*theta**2/336 + sigma**3/72 - sigma**3*theta**2/1152 ,
            -torch.sin(theta)/theta**3 + theta**(-2) + sigma*(1/(2*theta**2) - torch.sin(theta)/theta**3 - torch.cos(theta)/theta**4 + theta**(-4)) + sigma**2*(1/(6*theta**2) - torch.sin(theta)/(2*theta**3) - torch.cos(theta)/theta**4 + torch.sin(theta)/theta**5) + sigma**3*(1/(24*theta**2) - torch.sin(theta)/(6*theta**3) - torch.cos(theta)/(2*theta**4) + torch.sin(theta)/theta**5 + torch.cos(theta)/theta**6 - 1/theta**6),
        ),
        torch.where(torch.abs(theta) < eps,
            (sigma**5*torch.exp(sigma)/2 - sigma**4*torch.exp(sigma) + sigma**3*torch.exp(sigma) - sigma**3)/sigma**6 + theta**2*(-torch.exp(sigma)/(24*sigma) + torch.exp(sigma)/(6*sigma**2) - torch.exp(sigma)/(2*sigma**3) + torch.exp(sigma)/sigma**4 - torch.exp(sigma)/sigma**5 + sigma**(-5)),
            (torch.expm1(sigma) / sigma - (torch.exp(sigma) * (sigma * torch.cos(theta) + theta * torch.sin(theta)) - sigma) / (theta * theta + sigma * sigma)) / (theta * theta),
        ),
    )

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


def sympy_series():
    import sympy as sp

    class sp_expm1(sp.Function):
        def fdiff(self, argindex=1):
            return sp.exp(self.args[0])

    sigma, theta = sp.symbols('sigma theta', real=True)

    expr = ((sp.exp(sigma) - 1) / sigma - (sp.exp(sigma) * (sigma * sp.cos(theta) + theta * sp.sin(theta)) - sigma) / (theta * theta + sigma * sigma)) / (theta * theta)

    taylor_theta = sp.series(expr, theta, 0, 1, "+")

    # expr_diff = sp.diff(expr)
    # print("expr_diff", expr_diff.evalf(subs={'theta': 0}))
    # print(sp.limit(expr_diff, theta, 0, "+"))
    # print(sp.limit(expr, theta, 0, "+"))

    taylor_sigma = sp.series(expr, sigma, 0, 4, "+")

    taylor_theta_sigma = sp.series(sp.series(expr, theta, 0, 4, "+").removeO(), sigma, 0, 4, "+").removeO().expand().simplify()

    taylor_sigma_theta = sp.series(sp.series(expr, sigma, 0, 4, "+").removeO(), theta, 0, 4, "+").removeO().expand().simplify()

    print(taylor_theta)
    print(taylor_sigma)
    print(taylor_theta_sigma)
    print(taylor_sigma_theta)

    # print(sp.latex(simplified_expr))
    # print(simplified_expr)

    # print(expr.diff(x))   # exp(x)
    # f = lambdify(x, expr)
    # print(f(1))        # 1.718281828459045
    # print(f(1e-20))    # 1e-20, unlike exp(x)-1 which would evaluate to 0

    import ipdb; ipdb.set_trace()

# sympy_series()


def test_se3():
    # SE3 test
    import pytransform3d.transformations as pt
    import pytransform3d.rotations as pr
    import numpy as np

    T = pt.random_transform(np.random.default_rng(None))

    assert torch.isclose(torch.tensor(pt.exponential_coordinates_from_transform(T), dtype=torch.float32), se3_logmap(torch.tensor(T[None,], dtype=torch.float32))).all()

    assert torch.isclose(torch.tensor(pt.transform_from_exponential_coordinates(pt.exponential_coordinates_from_transform(T)), dtype=torch.float32), se3_expmap(se3_logmap(torch.tensor(T[None,], dtype=torch.float32)))).all()

    assert torch.isclose(torch.tensor(pt.invert_transform(T), dtype=torch.float32), se3_inv(torch.tensor(T, dtype=torch.float32)[None,])).all()

test_se3()


def test_sim3():
    # Sim3 test
    import lietorch

    print(lietorch.Sim3.exp(torch.tensor([[0,0,0,0,0,0,0.]])).matrix())
    #                  scale  rot   trans              trans   rot scale
    print(torch.tensor([[0., 0,0,0, 0,3,0]]))
    print(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,0,0, 0.]])).matrix()))
    print(torch.tensor([[0., 0,2,0, 0,3,0]]))
    print(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,2,0, 0.]])).matrix()))
    print(torch.tensor([[1., 0,0,0, 0,3,0]]))
    print(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,0,0, 1.]])).matrix()))
    print(torch.tensor([[1., 0,2,0, 0,3,0]]))
    print(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,2,0, 1.]])).matrix()))
    print()
    print(torch.tensor([[1., 0,0,0, 0,0,2]]))
    print(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,0,2, 0,0,0, 1.]])).matrix()))
    print()
    print(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,0,0, 0.]])).matrix())
    print(sim3_expmap(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,0,0, 0.]])).matrix())))
    print(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,2,0, 0.]])).matrix())
    print(sim3_expmap(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,2,0, 0.]])).matrix())))
    print(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,0,0, 1.]])).matrix())
    print(sim3_expmap(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,0,0, 1.]])).matrix())))
    print(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,2,0, 1.]])).matrix())
    print(sim3_expmap(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,3,0, 0,2,0, 1.]])).matrix())))
    print()
    print(lietorch.Sim3.exp(torch.tensor([[0,0,2, 0,0,0, 1.]])).matrix())
    print(sim3_expmap(sim3_logmap(lietorch.Sim3.exp(torch.tensor([[0,0,2, 0,0,0, 1.]])).matrix())))

    import ipdb; ipdb.set_trace()

test_sim3()


if __name__ == "__main__":
    pass