import pygame
import numpy as np
from OpenGL.GL import (
    glClear, glClearColor, glEnable, glDisable, glBlendFunc, glDepthMask,
    glMatrixMode, glLoadIdentity, glCullFace, glPolygonOffset,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_CULL_FACE,
    GL_BACK, GL_BLEND, GL_PROJECTION, GL_MODELVIEW, GL_POLYGON_OFFSET_FILL,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
)
from OpenGL.GLU import gluPerspective, gluLookAt

from camera import Camera
from lighting import Material, AmbientLight, DirectionalLight, PointLight
from object3d import Object3D
from shapes import create_cube, create_pyramid, create_sphere, create_octahedron, create_floor, create_walls
from utils import vec3, make_shadow_matrix


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
            direction=vec3(-0.4, -1.0, -0.2),
            diffuse=vec3(0.6, 0.7, 0.9),
            specular=vec3(0.6, 0.6, 0.7),
        )

        self.lights = [self.ambient_light, self.point_light, self.directional_light]

    def initialize(self):
        pygame.init()
        pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("PyOpenGL Phong Scene")

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glClearColor(0.05, 0.06, 0.08, 1.0)

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
            specular=vec3(0.05, 0.05, 0.05),
            shininess=4.0,
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

        cube = Object3D(create_cube(1.2), cube_material, position=[-2.5, 0.6, -2.0], rotation=[0.0, 25.0, 0.0])
        sphere = Object3D(create_sphere(0.9), shiny_material, position=[2.0, 1.0, -1.0], rotation=[0.0, 0.0, 0.0])
        pyramid = Object3D(create_pyramid(1.4, 1.2), pyramid_material, position=[-1.0, 0.0, 2.2], rotation=[0.0, -30.0, 0.0])
        octa = Object3D(create_octahedron(1.2), matte_material, position=[2.4, 0.6, 2.0], rotation=[0.0, 35.0, 0.0])

        self.objects = [cube, sphere, pyramid, octa]

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
        }

        keys = pygame.key.get_pressed()
        input_state["forward"] = keys[pygame.K_w] or keys[pygame.K_UP]
        input_state["backward"] = keys[pygame.K_s] or keys[pygame.K_DOWN]
        input_state["left"] = keys[pygame.K_a] or keys[pygame.K_LEFT]
        input_state["right"] = keys[pygame.K_d] or keys[pygame.K_RIGHT]
        input_state["up"] = keys[pygame.K_e]
        input_state["down"] = keys[pygame.K_q]

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
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.floor.draw(self.lights, self.camera.position)
        glDisable(GL_BLEND)

    def draw_shadows(self):
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
