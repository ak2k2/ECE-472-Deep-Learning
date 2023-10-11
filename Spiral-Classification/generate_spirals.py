import numpy as np


def generate_spiral_data(N=400, K=4, seed=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate spiral data.
    Input:
        N: number of points per class
        K: number of spiral arms
        seed: random seed
    Output:
        x_data: (2*N x 2)
        y_data: (2*N x 1)

    https://gist.github.com/45deg/e731d9e7f478de134def5668324c44c5
    """

    np.random.seed(seed)

    pi = np.pi
    theta = np.sqrt(np.random.rand(N)) * K * pi

    r_a = K * theta + pi
    data_a = np.array([np.cos(theta) * r_a, -np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(N, 2)

    r_b = -K * theta - pi
    data_b = np.array([np.cos(theta) * r_b, -np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(N, 2)

    y_a = np.zeros((N, 1))
    y_b = np.ones((N, 1))

    x_data = np.vstack([x_a, x_b])
    y_data = np.vstack([y_a, y_b])

    return x_data, y_data
