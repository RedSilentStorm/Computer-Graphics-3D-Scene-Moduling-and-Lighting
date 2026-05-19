import numpy as np

from utils import vec3, normalize, look_at


class Camera:
    def __init__(self, position=None, yaw=-90.0, pitch=-15.0):
        self.position = vec3(0.0, 2.5, 8.0) if position is None else np.array(position, dtype=np.float32)
        self.yaw = yaw
        self.pitch = pitch
        self.move_speed = 4.0
        self.turn_speed = 60.0

    def get_forward(self):
        rad_yaw = np.radians(self.yaw)
        rad_pitch = np.radians(self.pitch)
        x = np.cos(rad_yaw) * np.cos(rad_pitch)
        y = np.sin(rad_pitch)
        z = np.sin(rad_yaw) * np.cos(rad_pitch)
        return normalize(vec3(x, y, z))

    def get_right(self):
        forward = self.get_forward()
        return normalize(np.cross(forward, vec3(0.0, 1.0, 0.0)))

    def update(self, dt, input_state):
        if input_state["forward"]:
            self.position += self.get_forward() * self.move_speed * dt
        if input_state["backward"]:
            self.position -= self.get_forward() * self.move_speed * dt
        if input_state["left"]:
            self.position -= self.get_right() * self.move_speed * dt
        if input_state["right"]:
            self.position += self.get_right() * self.move_speed * dt
        if input_state["up"]:
            self.position[1] += self.move_speed * dt
        if input_state["down"]:
            self.position[1] -= self.move_speed * dt

        self.yaw += input_state["turn_x"] * self.turn_speed * dt
        self.pitch += input_state["turn_y"] * self.turn_speed * dt
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def get_view_matrix(self):
        target = self.position + self.get_forward()
        return look_at(self.position, target, vec3(0.0, 1.0, 0.0))
