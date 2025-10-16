# Matmul Arithmetic Intensity Animation

This Manim project visualizes one step of matrix multiplication in order to
introduce the arithmetic intensity calculation for a matmul kernel.

## Prerequisites

- Python 3.10+ (matching the `manim` release requirements)
- Recommended virtual environment located at `/Users/gabrielferns/venvs/aider`
- Manim dependencies (`ffmpeg`, `cairo`, `pango`, etc.) installed on the system

## Setup

Install Python packages using the provided interpreter:

```bash
/Users/gabrielferns/venvs/aider/bin/python -m pip install -r requirements.txt
```

## Rendering

Render the scene to a preview video:

```bash
/Users/gabrielferns/venvs/aider/bin/python -m manim matmul_scene.py MatmulArithmeticIntensityScene -p -ql
```

- `-p` opens the preview window when possible.
- `-ql` renders quickly at low resolution to speed up iteration. Replace with
  `-qh` for a high-quality render when ready.

The output video is saved under `media/videos/matmul_scene`.
