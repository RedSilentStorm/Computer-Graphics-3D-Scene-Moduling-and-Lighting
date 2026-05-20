import numpy as np
from dataclasses import dataclass

from utils import normalize, clamp01


@dataclass
class Material:
    ambient: np.ndarray
    diffuse: np.ndarray
    specular: np.ndarray
    shininess: float
    alpha: float = 1.0


@dataclass
class AmbientLight:
    color: np.ndarray
    enabled: bool = True


@dataclass
class DirectionalLight:
    direction: np.ndarray
    diffuse: np.ndarray
    specular: np.ndarray
    enabled: bool = True


@dataclass
class PointLight:
    position: np.ndarray
    diffuse: np.ndarray
    specular: np.ndarray
    attenuation: np.ndarray
    enabled: bool = True


def reflect(v, n):
    return v - 2.0 * np.dot(v, n) * n


def compute_phong(position, normal, view_pos, material, lights):
    n = normalize(normal)
    color = np.zeros(3, dtype=np.float32)

    for light in lights:
        if not light.enabled:
            continue

        if isinstance(light, AmbientLight):
            color += light.color * material.ambient
            continue

        if isinstance(light, DirectionalLight):
            l = normalize(-light.direction)
            ndotl = max(np.dot(n, l), 0.0)
            color += light.diffuse * material.diffuse * ndotl
            continue

        if isinstance(light, PointLight):
            l_vec = light.position - position
            dist = np.linalg.norm(l_vec)
            if dist > 0.0:
                l = l_vec / dist
            else:
                l = l_vec

            att = 1.0 / (light.attenuation[0] + light.attenuation[1] * dist + light.attenuation[2] * dist * dist)
            ndotl = max(np.dot(n, l), 0.0)
            color += (light.diffuse * material.diffuse * ndotl) * att

    return clamp01(color)
