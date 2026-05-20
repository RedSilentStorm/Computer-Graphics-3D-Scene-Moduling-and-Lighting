import pygame
import numpy as np
from OpenGL.GL import (
    glBegin, glEnd, glClear, glClearColor, glEnable, glDisable, glBlendFunc, glDepthMask,
    glMatrixMode, glLoadIdentity, glCullFace, glPolygonOffset, glDrawPixels, glWindowPos2f,
    glPushMatrix, glPopMatrix, glMultMatrixf, glColor4f, glVertex3f, glLineWidth,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_CULL_FACE,
    GL_BACK, GL_BLEND, GL_PROJECTION, GL_MODELVIEW, GL_POLYGON_OFFSET_FILL,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_RGBA, GL_UNSIGNED_BYTE, GL_TRIANGLES, GL_LINES, GL_FRONT_AND_BACK
)
from OpenGL.GLU import gluPerspective, gluLookAt

from camera import Camera
from lighting import Material, AmbientLight, DirectionalLight, PointLight
from object3d import Object3D
from shapes import create_cube, create_pyramid, create_sphere, create_octahedron, create_capsule, create_floor, create_walls
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
        self.show_light_lines = True

        self.fps = 0
        self._overlay_surface = None
        self._overlay_dirty = True
        self._overlay_frame = 0

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
        
        # Enable mouse relative mode for camera look
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

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
        # Room materials - all fully matte and opaque for lighting testing
        wall_material = Material(
            ambient=vec3(0.15, 0.15, 0.17),
            diffuse=vec3(0.4, 0.4, 0.45),
            specular=vec3(0.0, 0.0, 0.0),
            shininess=1.0,
            alpha=1.0,
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

        # Object materials - all fully matte and opaque for lighting testing
        shiny_material = Material(
            ambient=vec3(0.12, 0.1, 0.12),
            diffuse=vec3(0.6, 0.2, 0.7),
            specular=vec3(0.0, 0.0, 0.0),
            shininess=1.0,
            alpha=1.0,
        )
        matte_material = Material(
            ambient=vec3(0.1, 0.12, 0.1),
            diffuse=vec3(0.3, 0.6, 0.2),
            specular=vec3(0.0, 0.0, 0.0),
            shininess=1.0,
            alpha=1.0,
        )
        cube_material = Material(
            ambient=vec3(0.12, 0.12, 0.1),
            diffuse=vec3(0.6, 0.45, 0.2),
            specular=vec3(0.0, 0.0, 0.0),
            shininess=1.0,
            alpha=0.5,
        )
        capsule_material = Material(
            ambient=vec3(0.1, 0.12, 0.16),
            diffuse=vec3(0.2, 0.35, 0.65),
            specular=vec3(0.0, 0.0, 0.0),
            shininess=1.0,
            alpha=1.0,
        )
        pyramid_material = Material(
            ambient=vec3(0.35, 0.30, 0.12),
            diffuse=vec3(1.0, 0.95, 0.45),
            specular=vec3(1.6, 1.6, 1.6),
            shininess=128.0,
            alpha=1.0,
        )

        # Place objects in a circle (radius=3.5, Y=1.2)
        radius = 3.5
        y_pos = 1.2
        
        # 6 objects at 60-degree intervals
        angles = [0, 60, 120, 180, 240, 300]
        positions = []
        for angle in angles:
            rad = np.radians(angle)
            x = radius * np.cos(rad)
            z = radius * np.sin(rad)
            positions.append([x, y_pos, z])
        
        cube = Object3D(create_cube(1.2), cube_material, position=positions[0], rotation=[0.0, 25.0, 0.0])
        sphere = Object3D(create_sphere(0.9, stacks=12, slices=16), shiny_material, position=positions[1], rotation=[0.0, 0.0, 0.0])
        pyramid = Object3D(create_pyramid(1.4, 1.2), pyramid_material, position=positions[2], rotation=[0.0, -30.0, 0.0])
        capsule1 = Object3D(create_capsule(0.5, 1.4, segments=12, rings=3), capsule_material, position=positions[4], rotation=[0.0, -30.0, 0.0])

        self.objects = [cube, sphere, pyramid, capsule1]

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
        input_state["up"] = keys[pygame.K_SPACE]
        input_state["down"] = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        # Mouse look
        mouse_x, mouse_y = pygame.mouse.get_rel()
        # Mouse sensitivity scaling
        mouse_sensitivity = 0.03
        input_state["turn_x"] = mouse_x * mouse_sensitivity
        input_state["turn_y"] = -mouse_y * mouse_sensitivity

        self.camera.update(dt, input_state)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.event.set_grab(False)
                    pygame.mouse.set_visible(True)
                    self.running = False
                if event.key == pygame.K_1:
                    self.ambient_light.enabled = not self.ambient_light.enabled
                if event.key == pygame.K_2:
                    self.point_light.enabled = not self.point_light.enabled
                if event.key == pygame.K_3:
                    self.directional_light.enabled = not self.directional_light.enabled
                if event.key == pygame.K_h:
                    self.show_overlay = not self.show_overlay
                if event.key == pygame.K_l:
                    self.show_light_lines = not self.show_light_lines
                    self._overlay_dirty = True

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
        # Project shadows onto multiple planes: floor and the four walls
        hs = 6.0 if self.walls is None else float(self.walls.mesh.vertices[2][1]) if False else None
        # The room size used when creating walls in setup_scene is 12.0, so half-size is 6.0
        hs = 6.0

        # Define planes with normals (n), d and bounds so we can clip projected points to room extents
        planes = [
            { 'n': np.array([0.0, 1.0, 0.0], dtype=np.float32),  'd': 0.0, 'bounds': {'x':(-hs, hs), 'z':(-hs, hs)} },
            { 'n': np.array([0.0, 0.0, 1.0], dtype=np.float32),  'd': hs,  'bounds': {'x':(-hs, hs), 'y':(0.0, 6.0)} },
            { 'n': np.array([0.0, 0.0, -1.0], dtype=np.float32), 'd': hs,  'bounds': {'x':(-hs, hs), 'y':(0.0, 6.0)} },
            { 'n': np.array([1.0, 0.0, 0.0], dtype=np.float32),  'd': hs,  'bounds': {'z':(-hs, hs), 'y':(0.0, 6.0)} },
            { 'n': np.array([-1.0, 0.0, 0.0], dtype=np.float32), 'd': hs,  'bounds': {'z':(-hs, hs), 'y':(0.0, 6.0)} },
        ]

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-2.0, -2.0)
        glDepthMask(False)
        glDisable(GL_CULL_FACE)

        for obj in self.objects:
            # Render shadows only for the directional light to avoid multiple overlapping shadows
            if self.directional_light.enabled:
                dir_info = {'direction': np.array(self.directional_light.direction, dtype=np.float32)}
                obj.draw_shadow_on_planes(dir_info, planes, alpha=0.45)

        glEnable(GL_CULL_FACE)
        glDepthMask(True)
        glDisable(GL_POLYGON_OFFSET_FILL)
        glDisable(GL_BLEND)

    def draw_scene(self):
        self.walls.draw(self.lights, self.camera.position)
        self.draw_shadows()
        self.draw_light_markers()

        # Draw opaque objects first
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        for obj in self.objects:
            if obj.material.alpha >= 1.0:
                obj.draw(self.lights, self.camera.position)

        # Draw transparent objects last with blend mode
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_CULL_FACE)
        glDepthMask(False)
        for obj in self.objects:
            if obj.material.alpha < 1.0:
                obj.draw(self.lights, self.camera.position)
        glDepthMask(True)

        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glDisable(GL_BLEND)

        self.draw_floor()

        if self.show_overlay:
            self.draw_overlay()

    def _build_overlay(self):
        lines = [
            f"FPS:     {self.fps}",
            "",
            "Lights",
            "-------------------------------------------------",
            f"Ambient : {'ON' if self.ambient_light.enabled else 'OFF'}",
            f"Point   : {'ON' if self.point_light.enabled else 'OFF'} pos = {self.point_light.position}",
            f"Dir     : {'ON' if self.directional_light.enabled else 'OFF'} dir = {self.directional_light.direction}",
            f"Toggle  : 1 = Ambient  2 = Point  3 = Directional",
            f"Display : H = Panel    L = Light Lines",
        ]
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
        self._overlay_surface = (panel, panel_width, panel_height)

    def draw_overlay(self):
        self._overlay_frame += 1
        if self._overlay_dirty or self._overlay_frame >= 15:
            self._build_overlay()
            self._overlay_dirty = False
            self._overlay_frame = 0

        panel, panel_width, panel_height = self._overlay_surface

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        text_data = pygame.image.tostring(panel, "RGBA", True)
        glWindowPos2f(16, self.height - panel_height - 12)
        glDrawPixels(panel_width, panel_height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        glDisable(GL_BLEND)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)

    def draw_light_markers(self):
        if not self.show_light_lines:
            return
        
        marker_color = (1.0, 0.55, 0.1, 1.0)

        if self.point_light.enabled:
            self.draw_marker(self.point_light.position, marker_color)

        if self.directional_light.enabled:
            self.draw_directional_light_lines(self.directional_light.direction, marker_color)

    def draw_directional_light_lines(self, direction, color):
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        d = np.array(direction, dtype=np.float32)
        d = d / np.linalg.norm(d)

        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(d, ref))) > 0.9:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = np.cross(d, ref)
        right /= np.linalg.norm(right)
        up2 = np.cross(right, d)
        up2 /= np.linalg.norm(up2)

        source_plane = -d * 7.0
        line_len = 17.0

        # Build a wide ray field instead of a single ray bundle.
        # This gives the impression of light passing through the whole space.
        field_radius_right = 14.0
        field_radius_up = 11.0
        steps_right = 13
        steps_up = 11

        glLineWidth(1.1)
        glBegin(GL_LINES)
        for i in range(-steps_right, steps_right + 1):
            for j in range(-steps_up, steps_up + 1):
                xr = (i / steps_right) * field_radius_right
                yu = (j / steps_up) * field_radius_up

                start = source_plane + right * xr + up2 * yu

                # Slightly vary the beam direction so it feels like a soft sheet.
                bend = (np.sin(i * 0.7) + np.cos(j * 0.6)) * 0.02
                end = start + d * line_len + right * bend * 2.0 + up2 * bend

                edge_fade = 1.0 - min(1.0, (abs(xr) / field_radius_right) ** 1.2 + (abs(yu) / field_radius_up) ** 1.2) * 0.55
                alpha = 0.03 + 0.12 * edge_fade

                glColor4f(1.0, 1.0, 1.0, alpha)
                glVertex3f(float(start[0]), float(start[1]), float(start[2]))
                glVertex3f(float(end[0]), float(end[1]), float(end[2]))
        glEnd()

        # Add a second softer pass to thicken the field without making it look like a single line.
        glLineWidth(2.0)
        glBegin(GL_LINES)
        for i in range(-steps_right, steps_right + 1, 2):
            for j in range(-steps_up, steps_up + 1, 2):
                xr = (i / steps_right) * (field_radius_right * 0.92)
                yu = (j / steps_up) * (field_radius_up * 0.92)
                start = source_plane + right * xr + up2 * yu
                end = start + d * (line_len * 0.98)
                edge_fade = 1.0 - min(1.0, (abs(xr) / field_radius_right) ** 1.15 + (abs(yu) / field_radius_up) ** 1.15) * 0.65
                alpha = 0.015 + 0.07 * edge_fade
                glColor4f(1.0, 1.0, 1.0, alpha)
                glVertex3f(float(start[0]), float(start[1]), float(start[2]))
                glVertex3f(float(end[0]), float(end[1]), float(end[2]))
        glEnd()

        glLineWidth(1.0)
        
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

    def draw_marker(self, position, color):
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
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
        
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

    def run(self):
        self.initialize()
        clock = pygame.time.Clock()

        while self.running:
            dt = clock.tick(60) / 1000.0
            new_fps = int(clock.get_fps())
            if new_fps != self.fps:
                self.fps = new_fps
                self._overlay_dirty = True

            prev_states = (self.ambient_light.enabled, self.point_light.enabled, self.directional_light.enabled)
            self.handle_events()
            self.handle_input(dt)
            if (self.ambient_light.enabled, self.point_light.enabled, self.directional_light.enabled) != prev_states:
                self._overlay_dirty = True

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.set_camera()
            self.draw_scene()
            pygame.display.flip()

        pygame.quit()
