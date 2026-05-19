import numpy as np

from object3d import Mesh
from utils import normalize


def create_cube(size=1.0):
    hs = size * 0.5
    vertices = []
    normals = []
    indices = []

    # Each face has its own normal for crisp edges
    faces = [
        ([[-hs, -hs, hs], [hs, -hs, hs], [hs, hs, hs], [-hs, hs, hs]], [0, 0, 1]),
        ([[hs, -hs, -hs], [-hs, -hs, -hs], [-hs, hs, -hs], [hs, hs, -hs]], [0, 0, -1]),
        ([[-hs, -hs, -hs], [-hs, -hs, hs], [-hs, hs, hs], [-hs, hs, -hs]], [-1, 0, 0]),
        ([[hs, -hs, hs], [hs, -hs, -hs], [hs, hs, -hs], [hs, hs, hs]], [1, 0, 0]),
        ([[-hs, hs, hs], [hs, hs, hs], [hs, hs, -hs], [-hs, hs, -hs]], [0, 1, 0]),
        ([[-hs, -hs, -hs], [hs, -hs, -hs], [hs, -hs, hs], [-hs, -hs, hs]], [0, -1, 0]),
    ]

    for face in faces:
        base_index = len(vertices)
        face_vertices, face_normal = face
        for v in face_vertices:
            vertices.append(v)
            normals.append(face_normal)
        indices += [base_index, base_index + 1, base_index + 2, base_index, base_index + 2, base_index + 3]

    return Mesh(vertices, normals, indices)


def create_pyramid(base=1.0, height=1.0):
    hs = base * 0.5
    vertices = []
    normals = []
    indices = []

    apex = np.array([0.0, height, 0.0], dtype=np.float32)
    base_verts = [
        np.array([-hs, 0.0, -hs], dtype=np.float32),
        np.array([hs, 0.0, -hs], dtype=np.float32),
        np.array([hs, 0.0, hs], dtype=np.float32),
        np.array([-hs, 0.0, hs], dtype=np.float32),
    ]

    # Base (two triangles) - normal facing up
    base_index = len(vertices)
    base_normal = [0, 1, 0]
    for v in base_verts:
        vertices.append(v)
        normals.append(base_normal)
    indices += [base_index, base_index + 1, base_index + 2, base_index, base_index + 2, base_index + 3]

    # Side faces
    for i in range(4):
        v0 = base_verts[i]
        v1 = base_verts[(i + 1) % 4]
        face_normal = normalize(np.cross(v1 - v0, apex - v0))

        base_index = len(vertices)
        vertices += [v0, v1, apex]
        normals += [face_normal, face_normal, face_normal]
        indices += [base_index, base_index + 1, base_index + 2]

    return Mesh(vertices, normals, indices)


def create_sphere(radius=1.0, stacks=16, slices=24):
    vertices = []
    normals = []
    indices = []

    for stack in range(stacks + 1):
        phi = np.pi * stack / stacks
        for slice_idx in range(slices + 1):
            theta = 2.0 * np.pi * slice_idx / slices
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.cos(phi)
            z = radius * np.sin(phi) * np.sin(theta)

            vertices.append([x, y, z])
            normals.append(normalize([x, y, z]))

    for stack in range(stacks):
        for slice_idx in range(slices):
            first = stack * (slices + 1) + slice_idx
            second = first + slices + 1
            indices += [first, first + 1, second + 1, first, second + 1, second]

    return Mesh(vertices, normals, indices)


def create_capsule(radius=1.0, height=2.0, segments=16, rings=4):
    """Create a capsule (cylinder with hemispherical ends)"""
    vertices = []
    normals = []
    indices = []
    
    half_height = height * 0.5
    
    # Top hemisphere
    for ring in range(rings + 1):
        phi = np.pi * ring / (2.0 * rings)
        for seg in range(segments):
            theta = 2.0 * np.pi * seg / segments
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.cos(phi)
            z = radius * np.sin(phi) * np.sin(theta)
            
            vertices.append([x, half_height + y, z])
            normals.append(normalize([x, y, z]))
    
    # Top hemisphere triangles
    for ring in range(rings):
        for seg in range(segments):
            a = ring * segments + seg
            b = ring * segments + (seg + 1) % segments
            c = (ring + 1) * segments + seg
            d = (ring + 1) * segments + (seg + 1) % segments
            
            indices += [a, b, d, a, d, c]
    
    # Cylinder
    cyl_base = len(vertices)
    for seg in range(segments):
        theta = 2.0 * np.pi * seg / segments
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        
        # Top ring
        vertices.append([x, half_height, z])
        normals.append(normalize([x, 0, z]))
        
        # Bottom ring
        vertices.append([x, -half_height, z])
        normals.append(normalize([x, 0, z]))
    
    # Cylinder triangles
    for seg in range(segments):
        a = cyl_base + seg * 2
        b = cyl_base + ((seg + 1) % segments) * 2
        c = a + 1
        d = b + 1
        
        indices += [a, b, d, a, d, c]
    
    # Bottom hemisphere
    bot_base = len(vertices)
    for ring in range(rings + 1):
        phi = np.pi * ring / (2.0 * rings)
        for seg in range(segments):
            theta = 2.0 * np.pi * seg / segments
            x = radius * np.sin(phi) * np.cos(theta)
            y = -radius * np.cos(phi)
            z = radius * np.sin(phi) * np.sin(theta)
            
            vertices.append([x, -half_height + y, z])
            normals.append(normalize([x, y, z]))
    
    # Bottom hemisphere triangles
    for ring in range(rings):
        for seg in range(segments):
            a = bot_base + ring * segments + seg
            b = bot_base + ring * segments + (seg + 1) % segments
            c = bot_base + (ring + 1) * segments + seg
            d = bot_base + (ring + 1) * segments + (seg + 1) % segments
            
            indices += [a, b, d, a, d, c]
    
    return Mesh(vertices, normals, indices)


def create_octahedron(size=1.0):
    h = size * 0.5
    vertices = []
    normals = []
    indices = []

    top = np.array([0.0, h, 0.0], dtype=np.float32)
    bottom = np.array([0.0, -h, 0.0], dtype=np.float32)
    ring = [
        np.array([h, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, h], dtype=np.float32),
        np.array([-h, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, -h], dtype=np.float32),
    ]

    # Upper faces
    for i in range(4):
        v0 = ring[i]
        v1 = ring[(i + 1) % 4]
        face_normal = normalize(np.cross(v1 - v0, top - v0))
        base_index = len(vertices)
        vertices += [v0, v1, top]
        normals += [face_normal, face_normal, face_normal]
        indices += [base_index, base_index + 1, base_index + 2]

    # Lower faces
    for i in range(4):
        v0 = ring[i]
        v1 = ring[(i + 1) % 4]
        face_normal = normalize(np.cross(v1 - v0, bottom - v0))
        base_index = len(vertices)
        vertices += [v0, v1, bottom]
        normals += [face_normal, face_normal, face_normal]
        indices += [base_index, base_index + 1, base_index + 2]

    return Mesh(vertices, normals, indices)


def create_floor(size=10.0):
    hs = size * 0.5
    vertices = [
        [-hs, 0.0, -hs],
        [hs, 0.0, -hs],
        [hs, 0.0, hs],
        [-hs, 0.0, hs],
    ]
    normals = [[0, 1, 0]] * 4
    indices = [0, 1, 2, 0, 2, 3]
    return Mesh(vertices, normals, indices)


def create_walls(size=10.0, height=6.0):
    hs = size * 0.5
    h = height
    vertices = []
    normals = []
    indices = []

    walls = [
        ([[-hs, 0.0, -hs], [hs, 0.0, -hs], [hs, h, -hs], [-hs, h, -hs]], [0, 0, 1]),
        ([[hs, 0.0, hs], [-hs, 0.0, hs], [-hs, h, hs], [hs, h, hs]], [0, 0, -1]),
        ([[-hs, 0.0, hs], [-hs, 0.0, -hs], [-hs, h, -hs], [-hs, h, hs]], [1, 0, 0]),
        ([[hs, 0.0, -hs], [hs, 0.0, hs], [hs, h, hs], [hs, h, -hs]], [-1, 0, 0]),
    ]

    for wall in walls:
        base_index = len(vertices)
        wall_vertices, wall_normal = wall
        for v in wall_vertices:
            vertices.append(v)
            normals.append(wall_normal)
        indices += [base_index, base_index + 1, base_index + 2, base_index, base_index + 2, base_index + 3]

    return Mesh(vertices, normals, indices)
