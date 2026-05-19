import numpy as np
from OpenGL.GL import glBegin, glEnd, glColor4f, glVertex3f, glNormal3f, glPushMatrix, glPopMatrix, glMultMatrixf, GL_TRIANGLES

from utils import compose_matrix, normal_matrix, to_gl_matrix, normalize
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
