from hi_robotics.liegroup_utils import *


def test_so3_expmap():
    assert (so3_expmap(torch.tensor([[0., 0., 0.]])) == torch.eye(3).unsqueeze(0)).all()  # Should be identity
    print(so3_expmap(torch.tensor([[0., 0., 1e-13]])))
    print(so3_logmap(so3_expmap(torch.tensor([[0., 0., torch.pi - 1e-7]]))))

test_so3_expmap()


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

# plot_so3_singularity_near_pi()


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

test_so3_logmap()


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

test_sim3()


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