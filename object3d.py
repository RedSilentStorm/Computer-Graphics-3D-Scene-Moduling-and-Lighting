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
        for i, wp in enumerate(world_vertices):
            best = None
            best_t = None
            for plane in planes:
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
                    # For directional case t is distance along ray from vertex
                    valid = in_bounds(ip, plane['bounds'])
                    if not valid:
                        continue
                    # measure by t (smaller t == closer)
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

            projected[i] = best

        # Rasterize triangles formed by projected vertices where all three vertices have valid projection
        from OpenGL.GL import glBegin, glEnd, glColor4f, glVertex3f, GL_TRIANGLES

        glBegin(GL_TRIANGLES)
        for t_idx in range(0, len(self.mesh.indices), 3):
            ia = self.mesh.indices[t_idx]
            ib = self.mesh.indices[t_idx + 1]
            ic = self.mesh.indices[t_idx + 2]

            pa = projected[ia]
            pb = projected[ib]
            pc = projected[ic]

            if pa is None and pb is None and pc is None:
                continue
            # If any vertex missing, skip triangle to avoid stretched artifacts
            if pa is None or pb is None or pc is None:
                continue

            glColor4f(0.0, 0.0, 0.0, alpha)
            glVertex3f(float(pa[0]), float(pa[1]), float(pa[2]))
            glVertex3f(float(pb[0]), float(pb[1]), float(pb[2]))
            glVertex3f(float(pc[0]), float(pc[1]), float(pc[2]))
        glEnd()
