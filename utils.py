import math
import numpy as np


def vec3(x, y, z):
    return np.array([x, y, z], dtype=np.float32)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return v
    return v / norm


def clamp01(v):
    return np.clip(v, 0.0, 1.0)


def identity_matrix():
    return np.identity(4, dtype=np.float32)


def translation_matrix(t):
    m = identity_matrix()
    m[0, 3] = t[0]
    m[1, 3] = t[1]
    m[2, 3] = t[2]
    return m


def scale_matrix(s):
    m = identity_matrix()
    m[0, 0] = s[0]
    m[1, 1] = s[1]
    m[2, 2] = s[2]
    return m


def rotation_x(deg):
    rad = math.radians(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    m = identity_matrix()
    m[1, 1] = c
    m[1, 2] = -s
    m[2, 1] = s
    m[2, 2] = c
    return m


def rotation_y(deg):
    rad = math.radians(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    m = identity_matrix()
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def rotation_z(deg):
    rad = math.radians(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    m = identity_matrix()
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def compose_matrix(position, rotation, scale):
    m = translation_matrix(position)
    m = m @ rotation_y(rotation[1])
    m = m @ rotation_x(rotation[0])
    m = m @ rotation_z(rotation[2])
    m = m @ scale_matrix(scale)
    return m


def look_at(eye, target, up):
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    m = identity_matrix()
    m[0, 0] = s[0]
    m[0, 1] = s[1]
    m[0, 2] = s[2]
    m[1, 0] = u[0]
    m[1, 1] = u[1]
    m[1, 2] = u[2]
    m[2, 0] = -f[0]
    m[2, 1] = -f[1]
    m[2, 2] = -f[2]

    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def normal_matrix(model_matrix):
    upper = model_matrix[:3, :3]
    return np.linalg.inv(upper).T


def to_gl_matrix(matrix):
    # Convert row-major to OpenGL column-major layout
    return matrix.T.flatten().astype(np.float32)


def make_shadow_matrix(plane, light_pos):
    a, b, c, d = plane
    lx, ly, lz, lw = light_pos
    dot = a * lx + b * ly + c * lz + d * lw

    m = np.array([
        [dot - lx * a, -lx * b, -lx * c, -lx * d],
        [-ly * a, dot - ly * b, -ly * c, -ly * d],
        [-lz * a, -lz * b, dot - lz * c, -lz * d],
        [-lw * a, -lw * b, -lw * c, dot - lw * d],
    ], dtype=np.float32)
    return m
