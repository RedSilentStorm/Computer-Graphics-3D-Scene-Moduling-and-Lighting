import numpy as np
from OpenGL.GL import glBegin, glEnd, glColor4f, glVertex3f, glNormal3f, glPushMatrix, glPopMatrix, glMultMatrixf, GL_TRIANGLES

from utils import compose_matrix, normal_matrix, to_gl_matrix
from lighting import compute_phong


class Mesh:
    def __init__(self, vertices, normals, indices):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.int32)


class Object3D:
    def __init__(self, mesh, material, position=None, rotation=None, scale=None):
        self.mesh = mesh
        self.material = material
        self.position = np.array(position if position is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array(rotation if rotation is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        self.scale = np.array(scale if scale is not None else [1.0, 1.0, 1.0], dtype=np.float32)

    def get_model_matrix(self):
        return compose_matrix(self.position, self.rotation, self.scale)

    def draw(self, lights, view_pos, extra_matrix=None, alpha_override=None):
        model = self.get_model_matrix()
        if extra_matrix is not None:
            model = extra_matrix @ model
        n_matrix = normal_matrix(model)

        glPushMatrix()
        glMultMatrixf(to_gl_matrix(model))

        glBegin(GL_TRIANGLES)
        for idx in self.mesh.indices:
            v = self.mesh.vertices[idx]
            n = self.mesh.normals[idx]

            world_pos = (model @ np.array([v[0], v[1], v[2], 1.0], dtype=np.float32))[:3]
            world_n = n_matrix @ n

            color = compute_phong(world_pos, world_n, view_pos, self.material, lights)
            alpha = self.material.alpha if alpha_override is None else alpha_override
            glColor4f(color[0], color[1], color[2], alpha)
            glNormal3f(world_n[0], world_n[1], world_n[2])
            glVertex3f(v[0], v[1], v[2])
        glEnd()

        glPopMatrix()

    def draw_shadow(self, shadow_matrix, alpha=0.5):
        model = shadow_matrix @ self.get_model_matrix()

        glPushMatrix()
        glMultMatrixf(to_gl_matrix(model))

        glBegin(GL_TRIANGLES)
        for idx in self.mesh.indices:
            v = self.mesh.vertices[idx]
            glColor4f(0.0, 0.0, 0.0, alpha)
            glVertex3f(v[0], v[1], v[2])
        glEnd()

        glPopMatrix()

    def draw_shadow_on_planes(self, light, planes, alpha=0.5):
        """
        Project this object's vertices along the light onto the provided planes.
        `light` is either a numpy array (point light position with w=1.0) or
        a dict-like object with `direction` (for directional light) and a flag.
        `planes` is a list of dicts: { 'n': np.array, 'd': float, 'bounds': {axis:(min,max), ...} }
        """
        model = self.get_model_matrix()

        # Precompute world-space positions for vertices
        world_vertices = []
        for v in self.mesh.vertices:
            wp = (model @ np.array([v[0], v[1], v[2], 1.0], dtype=np.float32))[:3]
            world_vertices.append(wp)

        projected = [None] * len(world_vertices)

        eps = 1e-6

        # Helper to test bounds
        def in_bounds(pt, bounds):
            for axis, (mn, mx) in bounds.items():
                val = pt[{'x':0, 'y':1, 'z':2}[axis]]
                if val < mn - 1e-4 or val > mx + 1e-4:
                    return False
            return True

        # For directional lights, light will be dict with 'direction'
        is_directional = hasattr(light, 'direction') or (isinstance(light, dict) and 'direction' in light)
        if is_directional:
            dir_vec = None
            if hasattr(light, 'direction'):
                dir_vec = light.direction
            else:
                dir_vec = light['direction']
            r_dir = np.array(dir_vec, dtype=np.float32)
            r_dir = r_dir / (np.linalg.norm(r_dir) + eps)
            # ray travels in light's direction (from source toward scene)

        else:
            # point light: light is numpy array position
            light_pos = np.array(light, dtype=np.float32)

        # Compute projected positions per vertex: choose closest valid intersection (smallest positive t)
        # Also remember which plane the best projection belongs to so we can group by plane.
        projected = [None] * len(world_vertices)
        projected_plane_idx = [None] * len(world_vertices)

        for i, wp in enumerate(world_vertices):
            best = None
            best_t = None
            best_plane = None
            for p_idx, plane in enumerate(planes):
                n = plane['n']
                d = plane['d']

                if is_directional:
                    r = r_dir
                    denom = np.dot(n, r)
                    if abs(denom) < eps:
                        continue
                    t = -(d + np.dot(n, wp)) / denom
                    if t <= 1e-5:
                        continue
                    ip = wp + r * t
                    valid = in_bounds(ip, plane['bounds'])
                    if not valid:
                        continue
                    metric = t
                else:
                    r = wp - light_pos
                    denom = np.dot(n, r)
                    if abs(denom) < eps:
                        continue
                    u = -(d + np.dot(n, light_pos)) / denom
                    if u <= 1e-5:
                        continue
                    ip = light_pos + u * r
                    valid = in_bounds(ip, plane['bounds'])
                    if not valid:
                        continue
                    metric = u

                if best is None or metric < best_t:
                    best = ip
                    best_t = metric
                    best_plane = p_idx

            projected[i] = best
            projected_plane_idx[i] = best_plane

        # Group projected points by plane so we draw at most one silhouette per plane.
        groups = {}
        for i, p in enumerate(projected):
            if p is None or projected_plane_idx[i] is None:
                continue
            groups.setdefault(projected_plane_idx[i], []).append((i, p))

        from OpenGL.GL import glBegin, glEnd, glColor4f, glVertex3f, GL_TRIANGLES

        glColor4f(0.0, 0.0, 0.0, alpha)
        glBegin(GL_TRIANGLES)

        def convex_hull_indices_2d(pts2d_with_idx):
            # pts2d_with_idx: list of (idx, (x,y))
            pts = [(p[1][0], p[1][1], p[0]) for p in pts2d_with_idx]
            pts.sort()

            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

            lower = []
            for x, y, oi in pts:
                while len(lower) >= 2 and cross((lower[-2][0], lower[-2][1]), (lower[-1][0], lower[-1][1]), (x, y)) <= 0:
                    lower.pop()
                lower.append((x, y, oi))

            upper = []
            for x, y, oi in reversed(pts):
                while len(upper) >= 2 and cross((upper[-2][0], upper[-2][1]), (upper[-1][0], upper[-1][1]), (x, y)) <= 0:
                    upper.pop()
                upper.append((x, y, oi))

            hull = lower[:-1] + upper[:-1]
            return [h[2] for h in hull]

        for plane_idx, items in groups.items():
            if len(items) < 3:
                continue

            # Build plane-local basis (u,v) for 2D projection
            plane_n = planes[plane_idx]['n']
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(float(np.dot(plane_n, ref))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            u = np.cross(ref, plane_n)
            u /= (np.linalg.norm(u) + 1e-9)
            v = np.cross(plane_n, u)
            v /= (np.linalg.norm(v) + 1e-9)

            pts2d_with_idx = []
            idx_to_3d = {}
            for idx, p3 in items:
                x = float(np.dot(p3, u))
                y = float(np.dot(p3, v))
                pts2d_with_idx.append((idx, (x, y)))
                idx_to_3d[idx] = p3

            hull_indices = convex_hull_indices_2d(pts2d_with_idx)
            if len(hull_indices) < 3:
                continue

            # Emit triangle fan from first hull vertex
            center_idx = hull_indices[0]
            for k in range(1, len(hull_indices) - 1):
                a = idx_to_3d[center_idx]
                b = idx_to_3d[hull_indices[k]]
                c = idx_to_3d[hull_indices[k + 1]]
                glVertex3f(float(a[0]), float(a[1]), float(a[2]))
                glVertex3f(float(b[0]), float(b[1]), float(b[2]))
                glVertex3f(float(c[0]), float(c[1]), float(c[2]))

        glEnd()
