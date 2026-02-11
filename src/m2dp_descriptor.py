"""
Implementation of a Multive 2D Projection (M2DP) Point Cloud Descriptor. M2DP projects a 3D point cloud to a series of 2D planes.

Original Paper: M2DP: A Novel 3D Point Cloud Descriptor and Its Application in Loop Closure Detection (https://ieeexplore.ieee.org/document/7759060)
"""

import numpy as np
from sklearn.decomposition import PCA

def preprocess_point_cloud(pc):
    """
    Shift point cloud and perform PCA to create a coordinate system.
    """
    # extract position values
    P = pc[:, :3].astype(np.float32)

    # shift x,y,z - points
    centroid = P.mean(axis=0)
    P_shifted = P - centroid

    # get orientation of point cloud with pca
    pca = PCA(n_components=3)
    P_pca = pca.fit_transform(P_shifted)

    centroid_pca = np.zeros(3, dtype=np.float32) # P_pca.mean(axis=0) is almost equal
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    return P_pca, centroid_pca, x_axis

def create_normal_vectors(theta, phi):
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    sin_phi = np.sin(phi)

    m_x = cos_theta * cos_phi
    m_y = cos_theta * sin_phi
    m_z = sin_theta

    return np.stack([m_x, m_y, m_z], axis=0)

def plane_basis(m):
    m = m / np.linalg.norm(m, axis=1, keepdims=True)

    a = np.zeros_like(m)
    mask = np.abs(m[:, 0]) < 0.9
    a[mask] = np.array([1.0, 0.0, 0.0])
    a[~mask] = np.array([0.0, 1.0, 0.0])

    # first basis vector
    u = np.cross(m, a)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)

    # second basis vector
    v = np.cross(m, u)

    return u, v

def project_to_plane(theta, phi, P, centroid, x_axis):
    """
    Create 2D plane where the points of the point cloud are projected to
    """
    # normal vector for 2D plane
    m = create_normal_vectors(theta, phi) # shape (3, N)
    u, v = plane_basis(m.T)

    point_proj = P[:, None, :] - (P @ m)[:, :, None] * m.T[None, :, :]
    P_2d = np.stack([
        np.sum(point_proj * u[None, :, :], axis=2),
        np.sum(point_proj * v[None, :, :], axis=2)], 
        axis=2)

    x_axis_proj = x_axis[None, :] - (x_axis @ m)[:, None] * m.T
    x_axis_2d = np.stack([
        np.sum(x_axis_proj * u, axis=1),
        np.sum(x_axis_proj * v, axis=1)
    ], axis=1)

    centroid_2d = centroid[:2]

    return P_2d, centroid_2d, x_axis_2d

def compute_bins(P_x, centroid_2d, l, t, p, q):
    # distance to centroid to determine its radius, i.e. circle_id
    centroid_proj = centroid_2d[None, None, :]
    diff = P_x - centroid_proj
    x_diff = diff[..., 0]
    y_diff = diff[..., 1]
    dist = np.linalg.norm(diff, axis=2)

    # get dist (i.e. radius) of farthest point of 2D plane
    idx = np.argmax(dist, axis=0)
    max_dist = dist[idx, np.arange(dist.shape[1])]
    
    # max radius
    r = max_dist / (l**2)

    # create l cirlces with radii [r, 2**2 * r, ..., l**2 * r]
    radii = (np.arange(1, l + 1)[None, :]**2) * r[:, None] # shape (p*q, l)

    # # split circle in t bins
    delta_theta = (2 * np.pi) / t

    rho = np.sqrt(x_diff*x_diff + y_diff*y_diff)
    theta = np.arctan2(y_diff, x_diff) # x_axis_proj not needed here as its already the "normal" x-axis
    theta = np.mod(theta, 2*np.pi) # convert to [0, 2pi]

    # shape (len(P_x), p*q) -> returns for each point the ring and sector index
    ring_idx = np.sum(rho[:, :, None] > radii[None, :, :], axis=2)
    sector_idx = (theta / delta_theta).astype(int)
    sector_idx = np.clip(sector_idx, 0, t-1)

    # compute A(p*q, l*t) matrix - for each angle combination the number of points for each bin
    bin_id = ring_idx * t + sector_idx
    plane_offsets = (np.arange(p*q) * (l * t))[None, :]
    flat = (bin_id + plane_offsets)
    counts = np.bincount(flat.ravel(), minlength=(p*q) * (l * t)).reshape((p*q), (l * t))

    return counts

def generate_angles(p, q):
    theta_rng = np.arange(0, np.pi, np.pi / p)
    phi_rng = np.arange(0, np.pi/2, np.pi /(2*q))

    theta, phi = np.meshgrid(theta_rng, phi_rng)
    theta, phi = theta.flatten(), phi.flatten()

    return theta, phi

def compute_m2dp_signature_vectors(pc, l, t, p, q):
    """
    Main M2DP computation. Creates for a point cloud a descriptors matrix.

    Args:
        pc (np.array): Point cloud
        l (int): Number of concentric circles
        t (int): Number of bins per circle
        p (int): Number of 2D planes (azimuth)
        q (int): Number of 2D planes (elevation)

    Returns:
        np.array: Descriptor matrix with shape (p*q, l*t)
    """
    # preprocess point cloud
    P_pca, centroid_pca, x_axis = preprocess_point_cloud(pc)
    
    # generate angles
    theta, phi = generate_angles(p, q)

    # project point cloud, x-axis and centroid to 2D plane
    P_2d, centroid_2d, x_axis_2d = project_to_plane(theta, phi, P_pca, centroid_pca, x_axis)

    # generate bins (l circles with t bins each), reset to 0 for each (theta, phi) combination, count number of points in each bin, i.e. signature vector
    A = compute_bins(P_2d, centroid_2d, l, t, p, q) # 2d signature matrix

    # run SVD to get first left and right singular vectors
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    descriptor = np.concatenate([U[:, 0], U[:, 1]])

    # normalize
    descriptor /= np.linalg.norm(descriptor)

    return descriptor

