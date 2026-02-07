import numpy as np
from sklearn.decomposition import PCA
import math

class M2DP():

    """
    Implementation of a Multive 2D Projection (M2DP) Point Cloud Descriptor. M2DP projects a 3D point cloud to a series of 2D planes.
    
    Original Paper: M2DP: A Novel 3D Point Cloud Descriptor and Its Application in Loop Closure Detection (https://ieeexplore.ieee.org/document/7759060)
    """

    def __init__(self, l, t, p, q):
        """
        :param l: Number of concentric circles
        :param t: Number of bins per circle
        :param p: Number of 2D planes (azimuth)
        :param q: Number of 2D planes (elevation)
        """
        
        self.l = l 
        self.t = t
        self.p = p
        self.q = q

    def plane_basis_from_normal(self, m):
        m = np.asarray(m, dtype=float)
        m /= np.linalg.norm(m)

        v = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(v, m)) > 0.9:
            v = np.array([0.0, 1.0, 0.0])

        e1 = np.cross(m, v)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(m, e1)
        return e1, e2, m

    def preprocess_point_cloud(self, pc):
        """
        Shift point cloud and perform PCA to create a coordinate system.
        """

        # shift x,y,z - points
        centroid = np.mean(pc[:, :3], axis=0)
        pc_shifted = pc[:,:3].copy()
        pc_shifted -= centroid

        # get orientation of point cloud with pca
        pca = PCA(n_components=3)
        pca.fit(pc_shifted)
        axes = pca.components_
        
        x_axis, y_axis = axes[0], axes[1]

        return centroid, pc_shifted, x_axis
    
    def points_per_bin(self, P_x, centroid_proj, x_axis_proj, bins):
        # get dist (i.e. radius) of farthest point of 2D plane
        dists = np.linalg.norm(P_x - centroid_proj, axis=1)
        idx = np.argmax(dists)
        r = dists[idx] / (self.l**2) # max_dist = l**2 * r

        # create l cirlces with radii [r, 2**2 * r, ..., l**2 * r]
        radii = (np.arange(1, self.l+1)**2) * r

        # split circle in t bins
        delta_theta = (2 * math.pi) / self.t

        # for each point calculate:
        # - distance to centroid to determine its radius, i.e. circle_id
        # - angle between vector from centroid to point and x_axis to determine the bin it lies in, i.e. bin_id
        for point in P_x:
            help_v = point - centroid_proj

            dist = np.linalg.norm(help_v)
            angle = np.arctan2(
                help_v[0]*x_axis_proj[1] - help_v[1]*x_axis_proj[0],
                help_v[0]*x_axis_proj[0] + help_v[1]*x_axis_proj[1]
            )

            circle_id = int(np.searchsorted(radii, dist))
            bin_id = int(np.floor(angle / delta_theta))
            bins[circle_id][bin_id] += 1
        
        return bins.flatten()

    def calc_2d_signature(self, theta, phi, centroid, pc_shifted, x_axis):
        """
        Create 2D plane where the points of the point cloud are projected to
        """
        
        # create 2D plane with normal vector m
        m = np.array([math.cos(theta)*math.cos(phi), math.cos(theta)*math.sin(phi), math.sin(theta)])
        e1, e2, m = self.plane_basis_from_normal(m)

        # project shifted point cloud and x_axis to the plane
        P_x = pc_shifted @ np.stack([e1, e2], axis=1)
        x_axis_proj = x_axis @ np.stack([e1, e2], axis=1)
        
        # generate bins (l circles with t bins each), reset to 0 for each (theta, phi) combination
        bins = np.zeros((self.l, self.t))
        centroid_proj = centroid @ np.stack([e1, e2], axis=1)

        # count number of points in each bin, i.e. signature vector
        v_x = self.points_per_bin(P_x, centroid_proj, x_axis_proj, bins)

        return v_x

    def compute(self, pc):
        centroid, pc_shifted, x_axis = self.preprocess_point_cloud(pc)
        num_sign_vec = 0

        A = np.zeros((self.p*self.q, self.l*self.t)) # 2d signature matrix
        theta = 0
        while theta < math.pi:
            phi = 0
            while phi < (math.pi / 2):
                v_x = self.calc_2d_signature(theta, phi, centroid, pc_shifted, x_axis)
                A[num_sign_vec] = v_x
                num_sign_vec += 1

                phi += math.pi/(2*self.q)

            theta += math.pi/self.p

        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        u1 = U[:, 0]
        v1 = Vt[0, :]

        return np.concatenate([u1, v1])

class M2DP_Vectorized():

    """
    Implementation of a Multive 2D Projection (M2DP) Point Cloud Descriptor. M2DP projects a 3D point cloud to a series of 2D planes.
    
    Original Paper: M2DP: A Novel 3D Point Cloud Descriptor and Its Application in Loop Closure Detection (https://ieeexplore.ieee.org/document/7759060)
    """

    def __init__(self, pc, l, t, p, q):
        """
        :param l: Number of concentric circles
        :param t: Number of bins per circle
        :param p: Number of 2D planes (azimuth)
        :param q: Number of 2D planes (elevation)
        """
        self.pc = pc
        self.l = l 
        self.t = t
        self.p = p
        self.q = q

        self.compute(pc)
    
    def get_signature_vector(self):
        return self.v_x

    def preprocess_point_cloud(self, pc):
        """
        Shift point cloud and perform PCA to create a coordinate system.
        """

        # shift x,y,z - points
        centroid = np.mean(pc[:, :3], axis=0)
        pc_shifted = pc[:,:3].copy()
        pc_shifted -= centroid

        # get orientation of point cloud with pca
        pca = PCA(n_components=3)
        pca.fit(pc_shifted)
        axes = pca.components_
        
        x_axis, y_axis = axes[0], axes[1]

        return np.zeros(3), pc_shifted, x_axis
    
    def create_normal_vectors(self, theta, phi):
        cos_theta = np.cos(theta)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        sin_phi = np.sin(phi)

        m_x = cos_theta * cos_phi
        m_y = cos_theta * sin_phi
        m_z = sin_theta

        return np.stack([m_x, m_y, m_z], axis=1)

    def plane_basis_from_normal(self, m):
        ref_vec = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (16, 1))
        mask = np.abs(m[:, 0]) > 0.9
        ref_vec[mask] = np.array([0.0, 1.0, 0.0], dtype=np.float32) 

        h = np.sum(m * ref_vec, axis=1)
        e1 = ref_vec - h[:, np.newaxis] * m
        e2 = np.cross(m, e1)

        # normalize
        e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12)
        e2 = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12)

        return e1, e2
    
    def create_bins(self, P_x, centroid_proj, x_axis_proj, bins):
        # distance to centroid to determine its radius, i.e. circle_id
        diffs = P_x - centroid_proj[:, np.newaxis, :]
        dists = np.linalg.norm(diffs, axis=2)

        # get dist (i.e. radius) of farthest point of 2D plane
        idx = np.argmax(dists, axis=1)
        max_dists = dists[np.arange(dists.shape[0]), idx]
        
        r = max_dists / (self.l**2)

        # create l cirlces with radii [r, 2**2 * r, ..., l**2 * r]
        radii = (np.arange(1, self.l + 1)[None, :]**2) * r[:, None]

        # split circle in t bins
        delta_theta = (2 * np.pi) / self.t

        # angle between vector from centroid to point and x_axis to determine the bin it lies in, i.e. bin_id
        ref = x_axis_proj[:, None, :]
        angles = np.arctan2(
            diffs[..., 0]*ref[..., 1] - diffs[..., 1]*ref[..., 0], 
            diffs[..., 0]*ref[..., 0] + diffs[..., 1]*ref[..., 1]
        )
        angles = (angles + 2*np.pi) % (2*np.pi)

        for i in range(self.p * self.q):
            circle_id = np.searchsorted(radii[i], dists[i], side="right") - 1
            circle_id = np.clip(circle_id, 0, self.l-1)

            bin_id = np.floor(angles[i] / delta_theta).astype(int)
            flat = circle_id * self.t + bin_id
            counts = np.bincount(flat, minlength=self.l*self.t)
            bins[i] = counts.reshape(self.l, self.t)
        
        return bins

    def calc_2d_signature(self, theta, phi, centroid, pc_shifted, x_axis):
        """
        Create 2D plane where the points of the point cloud are projected to
        """
        
        # create 2D plane with normal vector m
        m = self.create_normal_vectors(theta, phi)
        e1, e2 = self.plane_basis_from_normal(m)

        # project shifted point cloud and x_axis to the plane
        P_x = np.einsum("nd,kde->kne", pc_shifted, np.stack([e1, e2], axis=2))
        x_axis_proj = np.stack([e1 @ x_axis, e2 @ x_axis], axis=1)

        # generate bins (l circles with t bins each), reset to 0 for each (theta, phi) combination
        bins = np.zeros((self.p * self.q, self.l, self.t))
        centroid_proj = np.stack([e1 @ centroid, e2 @ centroid], axis=1)

        # count number of points in each bin, i.e. signature vector
        bins = self.create_bins(P_x, centroid_proj, x_axis_proj, bins)

        return bins

    def compute(self, pc):
        centroid, pc_shifted, x_axis = self.preprocess_point_cloud(pc)
        
        # generate angles
        theta_rng = np.arange(0, np.pi, np.pi / self.p)
        phi_rng = np.arange(0, np.pi/2, np.pi /(2*self.q))

        theta, phi = np.meshgrid(theta_rng, phi_rng)
        theta, phi = theta.flatten(), phi.flatten()

        bins = self.calc_2d_signature(theta, phi, centroid, pc_shifted, x_axis)

        A = bins.reshape(self.p*self.q, self.l*self.t) # 2d signature matrix

        # run SVD to get first left and right singular vectors
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        u1 = U[:, 0]
        v1 = Vt[0, :]

        self.v_x = np.concatenate([u1, v1])
