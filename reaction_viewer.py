from __future__ import annotations

import argparse
import asyncio
import base64
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pyvista as pv
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import html, vuetify3


REQUIRED_COLUMNS = ("particle_id", "is_ionized", "time", "pos_x", "pos_y", "pos_z")

PARTICLE_STYLES = {
    0: {"label": "Neutral argon", "color": "#ff1e1e", "size": 5.25},
    1: {"label": "Ionized argon", "color": "#fff300", "size": 7.5},
    2: {"label": "Electron", "color": "#0007ff", "size": 3.0},
    3: {"label": "Metastable argon", "color": "#39f353", "size": 6.0},
}

VIEWER_TOGGLE_DEFAULTS = {
    "outline_visibility": False,
    "grid_visibility": True,
    "axis_visibility": True,
    "parallel_projection": True,
    "use_server_rendering": True,
}

# Lower value means zoom in more for parallel projection.
DEFAULT_PARALLEL_SCALE_MULTIPLIER = 0.75
GLOBAL_BOUNDING_BOX_PADDING_RATIO = 0.02
MIN_PLAY_INTERVAL_MS = 20
MAX_PLAY_INTERVAL_MS = 400
LOGO_FILE = Path(__file__).resolve().parent / "assets" / "logo.png"


@dataclass(frozen=True)
class SimulationData:
    particle_ids: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    particle_types: np.ndarray


@dataclass
class ParticleLayer:
    particle_type: int
    style: dict
    current_mesh: pv.PolyData
    current_rgba: np.ndarray


@dataclass
class TrailLayer:
    trail_mesh: pv.PolyData
    trail_rgba: np.ndarray


@dataclass
class ViewerState:
    data: SimulationData
    plotter: pv.Plotter
    layers: Dict[int, ParticleLayer]
    trail: TrailLayer


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def _load_logo_data_uri(path: Path) -> str:
    if not path.exists():
        return ""

    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:{mime};base64,{encoded}"


def load_simulation_data(csv_path: str | Path) -> SimulationData:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        rows = []
        for row in reader:
            rows.append(
                (
                    int(row["particle_id"]),
                    int(row["is_ionized"]),
                    float(row["time"]),
                    float(row["pos_x"]),
                    float(row["pos_y"]),
                    float(row["pos_z"]),
                )
            )

    if not rows:
        raise ValueError(f"No data rows found in {csv_path}")

    data = np.array(
        rows,
        dtype=[
            ("particle_id", np.int64),
            ("is_ionized", np.int64),
            ("time", np.float64),
            ("pos_x", np.float64),
            ("pos_y", np.float64),
            ("pos_z", np.float64),
        ],
    )

    particle_ids = np.sort(np.unique(data["particle_id"]))
    times = np.sort(np.unique(data["time"]))

    positions = np.empty((len(times), len(particle_ids), 3), dtype=float)
    particle_types = np.empty((len(times), len(particle_ids)), dtype=int)

    for time_index, time_value in enumerate(times):
        frame = data[data["time"] == time_value]
        if frame.size != len(particle_ids):
            raise ValueError(f"Missing particle data at time {time_value}")

        order = np.argsort(frame["particle_id"])
        frame = frame[order]
        if not np.array_equal(frame["particle_id"], particle_ids):
            raise ValueError(f"Particle ordering mismatch at time {time_value}")

        positions[time_index] = np.column_stack((frame["pos_x"], frame["pos_y"], frame["pos_z"]))
        particle_types[time_index] = frame["is_ionized"]

    return SimulationData(
        particle_ids=particle_ids,
        times=times,
        positions=positions,
        particle_types=particle_types,
    )


def _build_trail_mesh(positions: np.ndarray, rgba: np.ndarray) -> pv.PolyData:
    n_times, n_particles, _ = positions.shape
    flat_points = positions.reshape(-1, 3)

    lines = np.empty(n_particles * (n_times + 1), dtype=np.int64)
    cursor = 0
    for particle_index in range(n_particles):
        start = particle_index
        lines[cursor] = n_times
        lines[cursor + 1 : cursor + 1 + n_times] = start + np.arange(n_times, dtype=np.int64) * n_particles
        cursor += n_times + 1

    mesh = pv.PolyData(flat_points)
    mesh.lines = lines
    mesh.point_data["rgba"] = rgba
    return mesh


def _solid_rgba(color: str, n_points: int, alpha: int = 205) -> np.ndarray:
    red, green, blue = _hex_to_rgb(color)
    rgba = np.empty((n_points, 4), dtype=np.uint8)
    rgba[:, 0] = red
    rgba[:, 1] = green
    rgba[:, 2] = blue
    rgba[:, 3] = alpha
    return rgba


def _rgba_for_type_series(type_series: np.ndarray, alpha: int = 205) -> np.ndarray:
    rgba = np.empty((type_series.size, 4), dtype=np.uint8)
    for particle_type, style in PARTICLE_STYLES.items():
        mask = type_series == particle_type
        if np.any(mask):
            rgba[mask, :3] = _hex_to_rgb(style["color"])
    rgba[:, 3] = alpha
    return rgba


def _configure_plotter(plotter: pv.Plotter, data: SimulationData) -> None:
    all_points = data.positions.reshape(-1, 3)
    bounds_min = all_points.min(axis=0)
    bounds_max = all_points.max(axis=0)
    center = (bounds_min + bounds_max) / 2.0
    span = np.maximum(bounds_max - bounds_min, 1e-9)
    max_span = float(np.max(span))
    camera_offset = np.array([max_span * 2.5, max_span * 2.0, max_span * 1.8])

    # Draw a persistent domain box that encloses every particle over all time.
    padding = np.maximum(span * GLOBAL_BOUNDING_BOX_PADDING_RATIO, max_span * 1e-3)
    domain_bounds = (
        bounds_min[0] - padding[0],
        bounds_max[0] + padding[0],
        bounds_min[1] - padding[1],
        bounds_max[1] + padding[1],
        bounds_min[2] - padding[2],
        bounds_max[2] + padding[2],
    )
    plotter.add_mesh(
        pv.Box(bounds=domain_bounds),
        style="wireframe",
        color="#0f172a",
        line_width=2.5,
        opacity=1.0,
        lighting=False,
        pickable=False,
        name="global_domain_bounds",
    )

    plotter.set_background("#f6f9fc", top="#dbeafe")
    plotter.add_axes(line_width=2)
    plotter.show_bounds(
        grid="back",
        location="outer",
        all_edges=True,
        color="#334155",
        font_size=10,
    )
    plotter.camera_position = [
        tuple(center + camera_offset),
        tuple(center),
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = max_span * DEFAULT_PARALLEL_SCALE_MULTIPLIER


def _build_layers(plotter: pv.Plotter, data: SimulationData) -> tuple[Dict[int, ParticleLayer], TrailLayer]:
    layers: Dict[int, ParticleLayer] = {}
    n_particles = len(data.particle_ids)

    for particle_type, style in PARTICLE_STYLES.items():
        initial_positions = data.positions[0]
        current_mesh = pv.PolyData(initial_positions.copy())
        current_rgba = _solid_rgba(style["color"], n_particles, alpha=0)
        current_mesh.point_data["rgba"] = current_rgba
        plotter.add_mesh(
            current_mesh,
            scalars="rgba",
            rgba=True,
            point_size=style["size"],
            render_points_as_spheres=True,
            lighting=True,
            name=f"current_{particle_type}",
        )

        layers[particle_type] = ParticleLayer(
            particle_type=particle_type,
            style=style,
            current_mesh=current_mesh,
            current_rgba=current_rgba,
        )

    trail_rgba = _rgba_for_type_series(data.particle_types.reshape(-1), alpha=205)
    trail_mesh = _build_trail_mesh(data.positions, trail_rgba)

    return layers, TrailLayer(trail_mesh=trail_mesh, trail_rgba=trail_rgba)


def _update_frame(viewer_state: ViewerState, frame_index: int) -> str:
    data = viewer_state.data
    frame_index = int(np.clip(frame_index, 0, len(data.times) - 1))
    frame_positions = data.positions[frame_index]
    frame_types = data.particle_types[frame_index]

    for layer in viewer_state.layers.values():
        layer.current_mesh.points = frame_positions
        layer.current_rgba[:, 3] = np.where(frame_types == layer.particle_type, 255, 0).astype(np.uint8)
        layer.current_mesh.point_data["rgba"] = layer.current_rgba

    visible = np.repeat(np.arange(data.times.size) <= frame_index, data.particle_ids.size)
    viewer_state.trail.trail_rgba[:, 3] = np.where(visible, 255, 0).astype(np.uint8)
    viewer_state.trail.trail_mesh.point_data["rgba"] = viewer_state.trail.trail_rgba

    current_time = data.times[frame_index]
    viewer_state.plotter.render()
    return f"Frame {frame_index + 1}/{len(data.times)} | Time {current_time:.6g} | Particles {len(data.particle_ids)}"


def build_app(csv_path: str | Path = "data.csv"):
    data = load_simulation_data(csv_path)
    logo_data_uri = _load_logo_data_uri(LOGO_FILE)

    plotter = pv.Plotter(off_screen=True)
    _configure_plotter(plotter, data)
    layers, trail = _build_layers(plotter, data)
    viewer_state = ViewerState(data=data, plotter=plotter, layers=layers, trail=trail)

    server = get_server(client_type="vue3")
    state = server.state
    ctrl = server.controller

    state.frame = 0
    state.frame_max = len(data.times) - 1
    state.playing = False
    state.play_interval_ms = 90
    state.status_text = _update_frame(viewer_state, 0)
    control = {"playing": False, "internal_frame_update": False}

    @ctrl.add_task("on_server_ready")
    async def play_loop(**_):
        while True:
            await asyncio.sleep(max(state.play_interval_ms, MIN_PLAY_INTERVAL_MS) / 1000.0)
            if not control["playing"]:
                continue

            next_frame = (state.frame + 1) % (state.frame_max + 1)
            control["internal_frame_update"] = True
            with state:
                state.frame = next_frame
            control["internal_frame_update"] = False

    @state.change("frame")
    def on_frame_change(frame, **_):
        if not control["internal_frame_update"] and control["playing"]:
            control["playing"] = False
            with state:
                state.playing = False
        state.status_text = _update_frame(viewer_state, int(frame))
        if hasattr(ctrl, "view_update"):
            ctrl.view_update()

    @state.change("play_interval_ms")
    def on_play_interval_change(play_interval_ms, **_):
        try:
            value = int(float(play_interval_ms))
        except (TypeError, ValueError):
            value = MIN_PLAY_INTERVAL_MS

        clamped = int(np.clip(value, MIN_PLAY_INTERVAL_MS, MAX_PLAY_INTERVAL_MS))
        if clamped != value or play_interval_ms != clamped:
            with state:
                state.play_interval_ms = clamped

    def on_toggle_play():
        next_playing = not control["playing"]
        control["playing"] = next_playing
        with state:
            state.playing = next_playing
        state.flush()

    def on_reset():
        control["playing"] = False
        with state:
            state.playing = False
            state.frame = 0
        state.flush()

    with SinglePageLayout(server, full_height=True) as layout:
        layout.title.hide()
        layout.icon.hide()

        with layout.toolbar:
            if logo_data_uri:
                html.Img(
                    src=logo_data_uri,
                    style="height: 80px; width: auto;margin-left: 10px; margin-right: 10px; object-fit: contain;",
                )
            html.Span("Plasma Reaction Viewer", style="font-size: 1.1rem; font-weight: 600; margin-right: 14px;")

            vuetify3.VBtn(
                icon=("playing ? 'mdi-pause' : 'mdi-play'",),
                click=on_toggle_play,
                density="comfortable",
                variant="tonal",
                color="primary",
            )
            vuetify3.VBtn(
                "Reset",
                click=on_reset,
                density="comfortable",
                variant="outlined",
            )
            vuetify3.VSlider(
                v_model=("frame", 0),
                min=0,
                max=("frame_max", state.frame_max),
                step=1,
                hide_details=True,
                density="compact",
                style="max-width: 420px; margin-left: 16px;",
            )
            '''
            #bug related with playback speed causes crashing
            #and can't figure out how to reproduce
            vuetify3.VTextField(
                v_model=("play_interval_ms", 90),
                type="number",
                min=MIN_PLAY_INTERVAL_MS,
                max=MAX_PLAY_INTERVAL_MS,
                step=10,
                hide_details=True,
                density="compact",
                variant="outlined",
                label="ms/frame",
                style="max-width: 140px; margin-left: 8px;",
            )
            '''
            vuetify3.VSpacer()
            html.Span("{{ status_text }}", style="font-weight: 600;")

        with layout.content:
            with vuetify3.VContainer(fluid=True, classes="pa-2 fill-height"):
                with vuetify3.VRow(classes="fill-height"):
                    with vuetify3.VCol(cols="12", md="3", classes="pr-md-2"):
                        with vuetify3.VCard(variant="tonal", classes="pa-3", style="height: 100%;"):
                            html.H3("Particle Types")
                            for particle_type, style in PARTICLE_STYLES.items():
                                html.Div(
                                    f"\u25a0 {style['label']}",
                                    style=f"color: {style['color']}; margin: 6px 0; font-weight: 600;",
                                )
                            html.Hr()
                            html.P(
                                "Camera controls: drag to rotate, right-drag or wheel to zoom, middle-drag to pan.",
                                style="font-size: 0.9rem;",)
                            html.Hr()
                            with html.Div(style="line-height: 1.75; margin-top: 8px; font-size: 1.0rem;"):
                                html.Span(
                                    "This is a demonstration of the particle reactions happening in the quartz tube of the plasma chamber. "
                                    "Where electrons"
                                    " emitted from the tungsten pin react with the propellant gas, argon, through stepwise ionization. "
                                    "In this process, each argon stage represents a different energy state. "
                                )
                                html.Span("Neutral argon", style=f"color: {PARTICLE_STYLES[0]['color']}; font-weight: 700;")
                                html.Span(
                                    " is the ground-state atom before significant electron impact. When energetic "
                                )
                                html.Span("electrons", style=f"color: {PARTICLE_STYLES[2]['color']}; font-weight: 700;")
                                html.Span(" collide with it, some atoms are promoted to ")
                                html.Span("metastable argon", style=f"color: {PARTICLE_STYLES[3]['color']}; font-weight: 700;")
                                html.Span(
                                    ", a long-lived excited state that stores energy and is easier to ionize than ground-state argon. "
                                    "With additional collisions, argon loses an electron and becomes "
                                )
                                html.Span("ionized argon", style=f"color: {PARTICLE_STYLES[1]['color']}; font-weight: 700;")
                                html.Span(".")
                            
                    with vuetify3.VCol(cols="12", md="9", classes="pl-md-2"):
                        with vuetify3.VCard(variant="flat", classes="fill-height"):
                            view = plotter_ui(plotter)
                            ctrl.view_update = view.update

                            state[f"{plotter._id_name}_outline_visibility"] = VIEWER_TOGGLE_DEFAULTS[
                                "outline_visibility"
                            ]
                            state[f"{plotter._id_name}_grid_visibility"] = VIEWER_TOGGLE_DEFAULTS[
                                "grid_visibility"
                            ]
                            state[f"{plotter._id_name}_axis_visibility"] = VIEWER_TOGGLE_DEFAULTS[
                                "axis_visibility"
                            ]
                            state[f"{plotter._id_name}_parallel_projection"] = VIEWER_TOGGLE_DEFAULTS[
                                "parallel_projection"
                            ]
                            state[f"{plotter._id_name}_use_server_rendering"] = VIEWER_TOGGLE_DEFAULTS[
                                "use_server_rendering"
                            ]

    return server


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Trame plasma reaction viewer.")
    parser.add_argument("--csv", default="data.csv", help="Path to simulation CSV file")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", default=8080, type=int, help="Port to serve the app")
    parser.add_argument("--open-browser", action="store_true", help="Open browser on start")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    server = build_app(args.csv)
    server.start(host=args.host, port=args.port, open_browser=args.open_browser)


if __name__ == "__main__":
    main()
