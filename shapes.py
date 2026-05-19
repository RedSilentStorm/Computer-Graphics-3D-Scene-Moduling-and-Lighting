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

    # Base (two triangles)
    base_index = len(vertices)
    base_normal = [0, -1, 0]
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
            indices += [first, second, first + 1, second, second + 1, first + 1]

    return Mesh(vertices, normals, indices)


def create_cylinder(radius=0.5, height=1.5, segments=24):
    half_height = height * 0.5
    vertices = []
    normals = []
    indices = []

    for segment in range(segments):
        angle0 = 2.0 * np.pi * segment / segments
        angle1 = 2.0 * np.pi * (segment + 1) / segments

        x0 = radius * np.cos(angle0)
        z0 = radius * np.sin(angle0)
        x1 = radius * np.cos(angle1)
        z1 = radius * np.sin(angle1)

        base_index = len(vertices)
        vertices += [
            np.array([x0, -half_height, z0], dtype=np.float32),
            np.array([x1, -half_height, z1], dtype=np.float32),
            np.array([x1, half_height, z1], dtype=np.float32),
            np.array([x0, half_height, z0], dtype=np.float32),
        ]
        normal0 = normalize([x0, 0.0, z0])
        normal1 = normalize([x1, 0.0, z1])
        normals += [normal0, normal1, normal1, normal0]
        indices += [base_index, base_index + 1, base_index + 2, base_index, base_index + 2, base_index + 3]

    top_center_index = len(vertices)
    vertices.append(np.array([0.0, half_height, 0.0], dtype=np.float32))
    normals.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))

    bottom_center_index = len(vertices)
    vertices.append(np.array([0.0, -half_height, 0.0], dtype=np.float32))
    normals.append(np.array([0.0, -1.0, 0.0], dtype=np.float32))

    for segment in range(segments):
        angle0 = 2.0 * np.pi * segment / segments
        angle1 = 2.0 * np.pi * (segment + 1) / segments

        x0 = radius * np.cos(angle0)
        z0 = radius * np.sin(angle0)
        x1 = radius * np.cos(angle1)
        z1 = radius * np.sin(angle1)

        base_index = len(vertices)
        vertices += [
            np.array([x0, half_height, z0], dtype=np.float32),
            np.array([x1, half_height, z1], dtype=np.float32),
        ]
        normals += [np.array([0.0, 1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)]
        indices += [top_center_index, base_index, base_index + 1]

        base_index = len(vertices)
        vertices += [
            np.array([x0, -half_height, z0], dtype=np.float32),
            np.array([x1, -half_height, z1], dtype=np.float32),
        ]
        normals += [np.array([0.0, -1.0, 0.0], dtype=np.float32), np.array([0.0, -1.0, 0.0], dtype=np.float32)]
        indices += [bottom_center_index, base_index + 1, base_index]

    return Mesh(vertices, normals, indices)


def create_capsule(radius=0.5, height=1.5, segments=24, rings=8):
    half_height = height * 0.5
    vertices = []
    normals = []
    indices = []

    ring_groups = []

    # Top hemisphere, from the pole toward the shoulder.
    for ring in range(1, rings):
        t = ring / rings
        phi = t * (np.pi * 0.5)
        y = half_height + radius * np.cos(phi)
        ring_radius = radius * np.sin(phi)
        group_vertices = []
        group_normals = []
        for segment in range(segments):
            theta = 2.0 * np.pi * segment / segments
            x = ring_radius * np.cos(theta)
            z = ring_radius * np.sin(theta)
            group_vertices.append(np.array([x, y, z], dtype=np.float32))
            group_normals.append(normalize([x, y - half_height, z]))
        ring_groups.append((group_vertices, group_normals))

    top_shoulder_vertices = []
    top_shoulder_normals = []
    for segment in range(segments):
        theta = 2.0 * np.pi * segment / segments
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        top_shoulder_vertices.append(np.array([x, half_height, z], dtype=np.float32))
        top_shoulder_normals.append(normalize([x, 0.0, z]))
    ring_groups.append((top_shoulder_vertices, top_shoulder_normals))

    bottom_shoulder_vertices = []
    bottom_shoulder_normals = []
    for segment in range(segments):
        theta = 2.0 * np.pi * segment / segments
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        bottom_shoulder_vertices.append(np.array([x, -half_height, z], dtype=np.float32))
        bottom_shoulder_normals.append(normalize([x, 0.0, z]))
    ring_groups.append((bottom_shoulder_vertices, bottom_shoulder_normals))

    # Bottom hemisphere, from the shoulder toward the pole.
    for ring in range(1, rings):
        t = ring / rings
        phi = (1.0 - t) * (np.pi * 0.5)
        y = -half_height - radius * np.cos(phi)
        ring_radius = radius * np.sin(phi)
        group_vertices = []
        group_normals = []
        for segment in range(segments):
            theta = 2.0 * np.pi * segment / segments
            x = ring_radius * np.cos(theta)
            z = ring_radius * np.sin(theta)
            group_vertices.append(np.array([x, y, z], dtype=np.float32))
            group_normals.append(normalize([x, y + half_height, z]))
        ring_groups.append((group_vertices, group_normals))

    top_pole_index = len(vertices)
    vertices.append(np.array([0.0, half_height + radius, 0.0], dtype=np.float32))
    normals.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))

    for group_vertices, group_normals in ring_groups:
        vertices.extend(group_vertices)
        normals.extend(group_normals)

    bottom_pole_index = len(vertices)
    vertices.append(np.array([0.0, -half_height - radius, 0.0], dtype=np.float32))
    normals.append(np.array([0.0, -1.0, 0.0], dtype=np.float32))

    ring_offsets = []
    offset = 1
    for group_vertices, _ in ring_groups:
        ring_offsets.append(offset)
        offset += len(group_vertices)

    for ring_index in range(len(ring_groups)):
        start0 = top_pole_index if ring_index == 0 else ring_offsets[ring_index - 1]
        start1 = ring_offsets[ring_index]
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            if ring_index == 0:
                indices += [top_pole_index, start1 + segment, start1 + next_segment]
            else:
                indices += [start0 + segment, start1 + segment, start1 + next_segment, start0 + segment, start1 + next_segment, start0 + next_segment]

    last_group_start = ring_offsets[-1]
    last_group_start_prev = ring_offsets[-2]
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        indices += [last_group_start + segment, bottom_pole_index, last_group_start + next_segment]

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
        v0 = ring[(i + 1) % 4]
        v1 = ring[i]
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
        ([[-hs, h, -hs], [hs, h, -hs], [hs, h, hs], [-hs, h, hs]], [0, -1, 0]),
    ]

    for wall in walls:
        base_index = len(vertices)
        wall_vertices, wall_normal = wall
        for v in wall_vertices:
            vertices.append(v)
            normals.append(wall_normal)
        indices += [base_index, base_index + 1, base_index + 2, base_index, base_index + 2, base_index + 3]

    return Mesh(vertices, normals, indices)
