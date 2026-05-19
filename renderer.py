import pygame
import numpy as np
from OpenGL.GL import (
    glClear, glClearColor, glEnable, glDisable, glBlendFunc, glDepthMask,
    glMatrixMode, glLoadIdentity, glCullFace, glPolygonOffset,
    glBegin, glEnd, glColor4f, glVertex3f, glPushMatrix, glPopMatrix, glMultMatrixf,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_CULL_FACE,
    GL_BACK, GL_BLEND, GL_PROJECTION, GL_MODELVIEW, GL_POLYGON_OFFSET_FILL,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_LINES, GL_TRIANGLES
)
from OpenGL.GLU import gluPerspective, gluLookAt

from camera import Camera
from lighting import Material, AmbientLight, DirectionalLight, PointLight
from object3d import Object3D
from shapes import create_cube, create_sphere, create_cylinder, create_capsule, create_floor, create_walls
from utils import vec3, make_shadow_matrix, compose_matrix, to_gl_matrix, rotation_x, rotation_y, rotation_z, scale_matrix


class Renderer:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.running = True

        self.camera = Camera()
        self.lights = []
        self.objects = []
        self.floor = None
        self.walls = None

        self.ambient_light = AmbientLight(color=vec3(0.12, 0.12, 0.12))
        self.point_light = PointLight(
            position=vec3(0.0, 4.5, 0.0),
            diffuse=vec3(1.0, 0.9, 0.8),
            specular=vec3(1.0, 1.0, 1.0),
            attenuation=np.array([1.0, 0.1, 0.03], dtype=np.float32),
        )
        self.directional_light = DirectionalLight(
            direction=vec3(-0.6, 0.9, -0.7),
            diffuse=vec3(0.6, 0.7, 0.9),
            specular=vec3(0.6, 0.6, 0.7),
        )

        self.lights = [self.ambient_light, self.point_light, self.directional_light]
        self.light_marker_mesh = create_sphere(0.12, stacks=8, slices=12)
        self.directional_light_anchor = vec3(-5.2, 5.0, -5.2)

    def initialize(self):
        pygame.init()
        pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("PyOpenGL Phong Scene")

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glClearColor(0.05, 0.06, 0.08, 1.0)
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        self.setup_scene()

    def setup_scene(self):
        # Room materials
        wall_material = Material(
            ambient=vec3(0.15, 0.15, 0.17),
            diffuse=vec3(0.4, 0.4, 0.45),
            specular=vec3(0.0, 0.0, 0.0),
            shininess=1.0,
        )
        floor_material = Material(
            ambient=vec3(0.12, 0.11, 0.1),
            diffuse=vec3(0.35, 0.32, 0.28),
            specular=vec3(0.2, 0.2, 0.2),
            shininess=20.0,
            alpha=0.9,
        )

        self.walls = Object3D(create_walls(12.0, 6.0), wall_material)
        self.floor = Object3D(create_floor(12.0), floor_material)

        # Object materials
        shiny_material = Material(
            ambient=vec3(0.12, 0.1, 0.12),
            diffuse=vec3(0.6, 0.2, 0.7),
            specular=vec3(0.9, 0.9, 0.9),
            shininess=64.0,
        )
        matte_material = Material(
            ambient=vec3(0.1, 0.12, 0.1),
            diffuse=vec3(0.3, 0.6, 0.2),
            specular=vec3(0.05, 0.05, 0.05),
            shininess=8.0,
        )
        cube_material = Material(
            ambient=vec3(0.12, 0.12, 0.1),
            diffuse=vec3(0.6, 0.45, 0.2),
            specular=vec3(0.25, 0.2, 0.15),
            shininess=12.0,
        )
        pyramid_material = Material(
            ambient=vec3(0.1, 0.12, 0.16),
            diffuse=vec3(0.2, 0.35, 0.65),
            specular=vec3(0.1, 0.1, 0.2),
            shininess=18.0,
        )

        # create meshes first so we can compute bottom offsets (min Y)
        cube_mesh = create_cube(1.2)
        sphere_mesh = create_sphere(0.9)
        cylinder_mesh = create_cylinder(0.7, 1.5)
        capsule_mesh = create_capsule(0.55, 1.2)

        def compute_bottom_y(mesh, rotation_deg, scale=(1.0, 1.0, 1.0)):
            # Build rotation+scale matrix (no translation)
            r = rotation_y(rotation_deg[1]) @ rotation_x(rotation_deg[0]) @ rotation_z(rotation_deg[2]) @ scale_matrix(scale)
            verts = mesh.vertices
            ys = []
            for v in verts:
                hv = np.array([v[0], v[1], v[2], 1.0], dtype=np.float32)
                tv = (r @ hv)[:3]
                ys.append(float(tv[1]))
            min_y = float(np.min(ys))
            return -min_y

        cube_y = compute_bottom_y(cube_mesh, [0.0, 25.0, 0.0])
        sphere_y = compute_bottom_y(sphere_mesh, [0.0, 0.0, 0.0])
        cylinder_y = compute_bottom_y(cylinder_mesh, [0.0, -20.0, 0.0])
        capsule_y = compute_bottom_y(capsule_mesh, [0.0, 35.0, 0.0])

        cube = Object3D(cube_mesh, cube_material, position=[-2.5, cube_y, -2.0], rotation=[0.0, 25.0, 0.0])
        sphere = Object3D(sphere_mesh, shiny_material, position=[2.0, sphere_y, -1.0], rotation=[0.0, 0.0, 0.0])
        cylinder = Object3D(cylinder_mesh, matte_material, position=[-1.0, cylinder_y, 2.2], rotation=[0.0, -20.0, 0.0])
        capsule = Object3D(capsule_mesh, matte_material, position=[2.4, capsule_y, 2.0], rotation=[0.0, 35.0, 0.0])

        self.objects = [cube, sphere, cylinder, capsule]

    def handle_input(self, dt):
        input_state = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "up": False,
            "down": False,
            "turn_x": 0.0,
            "turn_y": 0.0,
            "mouse_dx": 0.0,
            "mouse_dy": 0.0,
        }

        keys = pygame.key.get_pressed()
        input_state["forward"] = keys[pygame.K_w] or keys[pygame.K_UP]
        input_state["backward"] = keys[pygame.K_s] or keys[pygame.K_DOWN]
        input_state["left"] = keys[pygame.K_a] or keys[pygame.K_LEFT]
        input_state["right"] = keys[pygame.K_d] or keys[pygame.K_RIGHT]
        input_state["up"] = keys[pygame.K_e] or keys[pygame.K_SPACE]
        input_state["down"] = keys[pygame.K_q] or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        input_state["mouse_dx"], input_state["mouse_dy"] = pygame.mouse.get_rel()

        if keys[pygame.K_j]:
            input_state["turn_x"] = -1.0
        if keys[pygame.K_l]:
            input_state["turn_x"] = 1.0
        if keys[pygame.K_i]:
            input_state["turn_y"] = 1.0
        if keys[pygame.K_k]:
            input_state["turn_y"] = -1.0

        self.camera.update(dt, input_state)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                if event.key == pygame.K_1:
                    self.ambient_light.enabled = not self.ambient_light.enabled
                if event.key == pygame.K_2:
                    self.point_light.enabled = not self.point_light.enabled
                if event.key == pygame.K_3:
                    self.directional_light.enabled = not self.directional_light.enabled

    def draw_light_marker(self, position, color, scale=0.18):
        model = compose_matrix(position, [0.0, 0.0, 0.0], [scale, scale, scale])
        glPushMatrix()
        glMultMatrixf(to_gl_matrix(model))
        glBegin(GL_TRIANGLES)
        for idx in self.light_marker_mesh.indices:
            v = self.light_marker_mesh.vertices[idx]
            glColor4f(color[0], color[1], color[2], 1.0)
            glVertex3f(v[0], v[1], v[2])
        glEnd()
        glPopMatrix()

    def draw_directional_light_vector(self):
        direction = self.directional_light.direction
        if np.linalg.norm(direction) == 0.0:
            return
        direction = direction / np.linalg.norm(direction)
        start = self.directional_light_anchor
        length = 3.0
        end = start + (-direction) * length

        # arrow head
        side = np.array([direction[2], 0.0, -direction[0]], dtype=np.float32)
        if np.linalg.norm(side) > 0.0:
            side = side / np.linalg.norm(side)
        head_left = end + side * 0.18 + direction * 0.35
        head_right = end - side * 0.18 + direction * 0.35

        glColor4f(0.75, 0.9, 1.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(start[0], start[1], start[2])
        glVertex3f(end[0], end[1], end[2])
        glVertex3f(end[0], end[1], end[2])
        glVertex3f(head_left[0], head_left[1], head_left[2])
        glVertex3f(end[0], end[1], end[2])
        glVertex3f(head_right[0], head_right[1], head_right[2])
        glEnd()

    def set_camera(self):
        forward = self.camera.get_forward()
        target = self.camera.position + forward
        glLoadIdentity()
        gluLookAt(
            self.camera.position[0], self.camera.position[1], self.camera.position[2],
            target[0], target[1], target[2],
            0.0, 1.0, 0.0,
        )

    def draw_reflection(self):
        reflection = np.identity(4, dtype=np.float32)
        reflection[1, 1] = -1.0

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_CULL_FACE)

        for obj in self.objects:
            obj.draw(self.lights, self.camera.position, extra_matrix=reflection, alpha_override=0.35)

        glEnable(GL_CULL_FACE)
        glDisable(GL_BLEND)

    def draw_floor(self):
        self.floor.draw(self.lights, self.camera.position)

    def draw_shadows(self):
        if not self.point_light.enabled:
            return

        plane = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        light_pos = np.array([self.point_light.position[0], self.point_light.position[1], self.point_light.position[2], 1.0], dtype=np.float32)
        shadow_matrix = make_shadow_matrix(plane, light_pos)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-2.0, -2.0)
        glDepthMask(False)

        for obj in self.objects:
            obj.draw_shadow(shadow_matrix, alpha=0.45)

        glDepthMask(True)
        glDisable(GL_POLYGON_OFFSET_FILL)
        glDisable(GL_BLEND)

    def draw_scene(self):
        self.walls.draw(self.lights, self.camera.position)
        self.draw_reflection()
        self.draw_floor()
        self.draw_shadows()

        for obj in self.objects:
            obj.draw(self.lights, self.camera.position)

        glDisable(GL_DEPTH_TEST)
        if self.point_light.enabled:
            self.draw_light_marker(self.point_light.position, vec3(1.0, 0.95, 0.55), scale=0.22)
        if self.directional_light.enabled:
            self.draw_light_marker(self.directional_light_anchor, vec3(0.7, 0.9, 1.0), scale=0.14)
            self.draw_directional_light_vector()
        glEnable(GL_DEPTH_TEST)

    def run(self):
        self.initialize()
        clock = pygame.time.Clock()

        while self.running:
            dt = clock.tick(60) / 1000.0
            self.handle_events()
            self.handle_input(dt)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.set_camera()
            self.draw_scene()
            pygame.display.flip()

        pygame.quit()
