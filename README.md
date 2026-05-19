# PyOpenGL Phong Scene

A small 3D scene built from scratch with PyOpenGL and Pygame. The scene contains multiple objects, a closed room, Phong lighting, a point-light planar shadow, and a simple floor reflection.

## Features
- Phong reflection model (ambient + diffuse + specular)
- Ambient, point, and directional lights (toggleable)
- Cube, sphere, pyramid, octahedron
- Room with floor, walls, and ceiling
- Planar shadow projection
- Simple floor reflection effect
- Camera movement

## Requirements
- Python 3.9+
- PyOpenGL
- PyOpenGL_accelerate (optional but recommended)
- Pygame
- NumPy

## Install
```
pip install PyOpenGL PyOpenGL_accelerate pygame numpy
```

## Run
```
python main.py
```

## Controls
- Move: W/A/S/D or arrow keys
- Vertical: Q/E
- Look: I/J/K/L
- Toggle lights: 1 (ambient), 2 (point), 3 (directional)
- Exit: Esc

## File Structure
- main.py: entry point
- renderer.py: render loop and scene setup
- object3d.py: mesh/object base class and draw logic
- shapes.py: procedural meshes
- lighting.py: materials, lights, and Phong math
- camera.py: camera movement
- utils.py: math helpers

## Notes
- Lighting is computed per-vertex on the CPU for clarity.
- Shadows are planar projections from the point light onto the floor.
