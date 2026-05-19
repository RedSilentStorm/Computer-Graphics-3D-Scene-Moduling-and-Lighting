import pygame
import numpy as np
from OpenGL.GL import (
    glBegin, glEnd, glClear, glClearColor, glEnable, glDisable, glBlendFunc, glDepthMask,
    glMatrixMode, glLoadIdentity, glCullFace, glPolygonOffset, glDrawPixels, glWindowPos2f,
    glPushMatrix, glPopMatrix, glMultMatrixf, glColor4f, glVertex3f, glLineWidth,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_CULL_FACE,
    GL_BACK, GL_BLEND, GL_PROJECTION, GL_MODELVIEW, GL_POLYGON_OFFSET_FILL,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_RGBA, GL_UNSIGNED_BYTE, GL_TRIANGLES, GL_LINES
)
from OpenGL.GLU import gluPerspective, gluLookAt

from camera import Camera
from lighting import Material, AmbientLight, DirectionalLight, PointLight
from object3d import Object3D
from shapes import create_cube, create_pyramid, create_sphere, create_octahedron, create_floor, create_walls
from utils import vec3, make_shadow_matrix, compose_matrix, to_gl_matrix


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
        self.light_marker_mesh = create_octahedron(0.25)
        self.show_overlay = True

        self.ambient_light = AmbientLight(color=vec3(0.12, 0.12, 0.12))
        self.point_light = PointLight(
            position=vec3(0.0, 4.5, 0.0),
            diffuse=vec3(1.0, 0.9, 0.8),
            specular=vec3(1.0, 1.0, 1.0),
            attenuation=np.array([1.0, 0.1, 0.03], dtype=np.float32),
        )
        self.directional_light = DirectionalLight(
            direction=vec3(-0.7, -1.0, -0.6),
            diffuse=vec3(0.6, 0.7, 0.9),
            specular=vec3(0.6, 0.6, 0.7),
        )

        self.lights = [self.ambient_light, self.point_light, self.directional_light]

    def initialize(self):
        pygame.init()
        pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("PyOpenGL Phong Scene")
        pygame.font.init()
        self.overlay_font = pygame.font.SysFont("Consolas", 18)

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
            specular=vec3(0.0, 0.0, 0.0),
            shininess=1.0,
            alpha=1.0,
        )

        self.walls = Object3D(create_walls(12.0, 6.0), wall_material)
        self.floor = Object3D(create_floor(12.0), floor_material)

        # Object materials
        shiny_material = Material(
            ambient=vec3(0.12, 0.1, 0.12),
            diffuse=vec3(0.6, 0.2, 0.7),
            specular=vec3(0.0, 0.0, 0.0),
            shininess=1.0,
        )
        matte_material = Material(
            ambient=vec3(0.1, 0.12, 0.1),
            diffuse=vec3(0.3, 0.6, 0.2),
            specular=vec3(0.0, 0.0, 0.0),
            shininess=1.0,
        )
        cube_material = Material(
            ambient=vec3(0.12, 0.12, 0.1),
            diffuse=vec3(0.6, 0.45, 0.2),
            specular=vec3(0.0, 0.0, 0.0),
            shininess=1.0,
        )
        pyramid_material = Material(
            ambient=vec3(0.1, 0.12, 0.16),
            diffuse=vec3(0.2, 0.35, 0.65),
            specular=vec3(0.0, 0.0, 0.0),
            shininess=1.0,
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
                if event.key == pygame.K_h:
                    self.show_overlay = not self.show_overlay

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
        return

    def draw_floor(self):
        self.floor.draw(self.lights, self.camera.position)

    def draw_shadows(self):
        if not self.point_light.enabled and not self.directional_light.enabled:
            return
        plane = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        if self.point_light.enabled:
            light_pos = np.array([
                self.point_light.position[0],
                self.point_light.position[1],
                self.point_light.position[2],
                1.0,
            ], dtype=np.float32)
        else:
            # Directional light is represented with w=0 for planar projection
            light_pos = np.array([
                -self.directional_light.direction[0],
                -self.directional_light.direction[1],
                -self.directional_light.direction[2],
                0.0,
            ], dtype=np.float32)
        shadow_matrix = make_shadow_matrix(plane, light_pos)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-2.0, -2.0)
        glDepthMask(False)
        glDisable(GL_CULL_FACE)

        for obj in self.objects:
            obj.draw_shadow(shadow_matrix, alpha=0.45)

        glEnable(GL_CULL_FACE)
        glDepthMask(True)
        glDisable(GL_POLYGON_OFFSET_FILL)
        glDisable(GL_BLEND)

    def draw_scene(self):
        self.walls.draw(self.lights, self.camera.position)
        self.draw_floor()
        self.draw_shadows()
        self.draw_light_markers()

        for obj in self.objects:
            obj.draw(self.lights, self.camera.position)

        if self.show_overlay:
            self.draw_overlay()

    def draw_overlay(self):
        lines = [
            "Lights:",
            f"Ambient: {'ON' if self.ambient_light.enabled else 'OFF'}",
            f"Point:   {'ON' if self.point_light.enabled else 'OFF'} pos={self.point_light.position}",
            f"Dir:     {'ON' if self.directional_light.enabled else 'OFF'} dir={self.directional_light.direction}",
            "Toggle:  1=Ambient  2=Point  3=Directional  H=Panel",
        ]

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        line_height = 22
        panel_width = 520
        panel_height = line_height * len(lines) + 8
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((255, 255, 255, 210))

        y = 4
        for line in lines:
            text_surface = self.overlay_font.render(line, True, (30, 30, 30))
            panel.blit(text_surface, (8, y))
            y += line_height

        text_data = pygame.image.tostring(panel, "RGBA", True)
        glWindowPos2f(16, self.height - panel_height - 12)
        glDrawPixels(panel.get_width(), panel.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        glDisable(GL_BLEND)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)

    def draw_light_markers(self):
        marker_color = (1.0, 0.55, 0.1, 1.0)

        if self.point_light.enabled:
            self.draw_marker(self.point_light.position, marker_color)

        if self.directional_light.enabled:
            # Visualize directional light as a marker outside the room in the opposite direction
            dir_vec = -self.directional_light.direction
            pos = dir_vec / np.linalg.norm(dir_vec) * 7.0 + vec3(0.0, 2.5, 0.0)
            self.draw_marker(pos, marker_color)
            self.draw_directional_arrow(pos, self.directional_light.direction, marker_color)

    def draw_directional_arrow(self, origin, direction, color):
        dir_norm = direction / np.linalg.norm(direction)
        start = origin
        end = origin + dir_norm * 2.0

        glLineWidth(3.0)
        glBegin(GL_LINES)
        glColor4f(color[0], color[1], color[2], color[3])
        glVertex3f(start[0], start[1], start[2])
        glVertex3f(end[0], end[1], end[2])
        glEnd()
        glLineWidth(1.0)

    def draw_marker(self, position, color):
        model = compose_matrix(position, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        glPushMatrix()
        glMultMatrixf(to_gl_matrix(model))

        glBegin(GL_TRIANGLES)
        for idx in self.light_marker_mesh.indices:
            v = self.light_marker_mesh.vertices[idx]
            glColor4f(color[0], color[1], color[2], color[3])
            glVertex3f(v[0], v[1], v[2])
        glEnd()

        glPopMatrix()

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
