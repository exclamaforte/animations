import math
import numpy as np
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Literal, Union

from manim import VGroup, Mobject, VMobject, Animation, AnimationGroup
from manim import Scene, config
from manim import BLUE, BLUE_C, GRAY_B, GRAY_D, BLACK, WHITE, YELLOW_B
from manim import Indicate, Flash, Arrow, Line, Square, Rectangle, RoundedRectangle
from manim import Text, MathTex
from manim import LEFT, RIGHT, UP, DOWN
from manim import FadeOut, Transform, Succession, interpolate_color


class LabeledRoundedRect(VGroup):
    def __init__(
        self,
        label_text: str | Mobject,
        color=BLUE,
        *,
        corner_radius: float = 0.25,
        width: float | None = None,
        height: float | None = None,
        padding: tuple[float, float] = (0.6, 0.35),
        text_cls=MathTex,  # or Tex
        text_kwargs: dict | None = None,
        rect_kwargs: dict | None = None,
    ):
        text_kwargs = text_kwargs or {}
        rect_kwargs = rect_kwargs or {}

        # Build/accept the label mobject
        if isinstance(label_text, Mobject):
            label = label_text
        else:
            label = text_cls(label_text, **text_kwargs)

        # Size rectangle to label + padding
        pad_x, pad_y = padding
        label_based_width = label.width + 2 * pad_x
        label_based_height = label.height + 2 * pad_y
        target_width = (
            max(label_based_width, width) if width is not None else label_based_width
        )
        target_height = (
            max(label_based_height, height)
            if height is not None
            else label_based_height
        )
        rect = RoundedRectangle(
            corner_radius=corner_radius,
            width=target_width,
            height=target_height,
            **rect_kwargs,
        )
        # Default styling (can be overridden via rect_kwargs)
        rect.set_stroke(
            color,
            width=4 if "stroke_width" not in rect_kwargs else rect.get_stroke_width(),
        )
        rect.set_fill(
            color,
            opacity=(
                0.15 if "fill_opacity" not in rect_kwargs else rect.get_fill_opacity()
            ),
        )

        # Center things and expose handles
        label.move_to(rect.get_center())
        self.rect = rect
        self.label = label
        super().__init__(rect, label)

    def set_label(self, new_text: str | Mobject):
        """Replace the label and keep it centered & sized (recomputes rect)."""
        # Keep current style/padding
        color = self.rect.get_stroke_color()
        stroke_w = self.rect.get_stroke_width()
        fill_col = self.rect.get_fill_color()
        fill_op = self.rect.get_fill_opacity()
        corner_radius = self.rect.corner_radius

        # Attempt to recover padding from current geometry
        pad_x = (self.rect.width - self.label.width) / 2
        pad_y = (self.rect.height - self.label.height) / 2

        # Build new label
        label = new_text if isinstance(new_text, Mobject) else Text(new_text)
        rect = (
            RoundedRectangle(
                corner_radius=corner_radius,
                width=label.width + 2 * pad_x,
                height=label.height + 2 * pad_y,
            )
            .set_stroke(color, width=stroke_w)
            .set_fill(fill_col, opacity=fill_op)
        )

        label.move_to(rect.get_center())
        self.clear()
        self.rect, self.label = rect, label
        self.add(rect, label)


class LabeledRoundedRectDag(VGroup):
    """Layered DAG composed of `LabeledRoundedRect` nodes connected by arrows."""

    def __init__(
        self,
        layers: Sequence[Sequence[str]],
        edges: Sequence[tuple[str, str]],
        *,
        node_specs: Mapping[str, Any] | None = None,
        default_color=BLUE,
        text_cls=MathTex,
        default_text_kwargs: Mapping[str, Any] | None = None,
        default_rect_kwargs: Mapping[str, Any] | None = None,
        padding: tuple[float, float] = (0.6, 0.35),
        width: float | None = None,
        height: float | None = None,
        node_buff: float = 1.5,
        layer_buff: float = 2.0,
        arrow_buff: float = 0.35,
        arrow_kwargs: Mapping[str, Any] | None = None,
    ):
        super().__init__()

        if not layers:
            raise ValueError("Provide at least one layer for the DAG.")

        node_specs = node_specs or {}
        arrow_kwargs = dict(arrow_kwargs or {})
        default_text_kwargs = dict(default_text_kwargs or {})
        default_rect_kwargs = dict(default_rect_kwargs or {})

        node_levels: dict[str, int] = {}
        for layer_index, layer in enumerate(layers):
            for node_name in layer:
                if node_name in node_levels:
                    msg = f"Duplicate node '{node_name}' detected across layers."
                    raise ValueError(msg)
                node_levels[node_name] = layer_index

        nodes_by_layer: list[VGroup] = []
        self.node_map: dict[str, LabeledRoundedRect] = {}

        for layer in layers:
            layer_nodes: list[LabeledRoundedRect] = []
            for node_name in layer:
                spec = node_specs.get(node_name)
                if isinstance(spec, dict):
                    label_text = spec.get("label", node_name)
                    color = spec.get("color", default_color)
                    node_width = spec.get("width", width)
                    node_height = spec.get("height", height)
                    padding_override = spec.get("padding", padding)
                    corner_radius = spec.get("corner_radius", 0.25)
                    node_text_cls = spec.get("text_cls", text_cls)
                    text_kwargs = dict(default_text_kwargs)
                    text_kwargs.update(spec.get("text_kwargs", {}))
                    rect_kwargs = dict(default_rect_kwargs)
                    rect_kwargs.update(spec.get("rect_kwargs", {}))
                else:
                    label_text = spec if spec is not None else node_name
                    color = default_color
                    node_width = width
                    node_height = height
                    padding_override = padding
                    corner_radius = 0.25
                    node_text_cls = text_cls
                    text_kwargs = dict(default_text_kwargs)
                    rect_kwargs = dict(default_rect_kwargs)

                node = LabeledRoundedRect(
                    label_text=label_text,
                    color=color,
                    corner_radius=corner_radius,
                    width=node_width,
                    height=node_height,
                    padding=padding_override,
                    text_cls=node_text_cls,
                    text_kwargs=text_kwargs,
                    rect_kwargs=rect_kwargs,
                )
                self.node_map[node_name] = node
                layer_nodes.append(node)

            layer_group = VGroup(*layer_nodes).arrange(RIGHT, buff=node_buff)
            nodes_by_layer.append(layer_group)

        self.layer_groups = VGroup(*nodes_by_layer).arrange(
            DOWN, buff=layer_buff, aligned_edge=LEFT
        )
        self.add(self.layer_groups)

        arrows = VGroup()
        for source, target in edges:
            if source not in self.node_map or target not in self.node_map:
                missing = source if source not in self.node_map else target
                msg = f"Edge references unknown node '{missing}'."
                raise ValueError(msg)
            if node_levels[source] >= node_levels[target]:
                msg = f"Edge ({source!r} -> {target!r}) violates DAG layering."
                raise ValueError(msg)

            start_node = self.node_map[source]
            end_node = self.node_map[target]
            arrow = Arrow(
                start_node.get_center(),
                end_node.get_center(),
                buff=arrow_buff,
                **arrow_kwargs,
            )
            arrows.add(arrow)

        self.edges = arrows
        self.add(self.edges)


@dataclass
class SlotVisual:
    """Holds references to a slot that can display a colored value."""

    box: VMobject
    group: VGroup
    side: float
    label: Mobject | None = None
    value_square: VMobject | None = None
    color: str | None = None

    def center(self) -> np.ndarray:
        return self.box.get_center()


InstructionType = Literal[
    "global_to_shared",
    "load_global",
    "load_shared",
    "op",
    "store_shared",
    "store_global",
    "narrate",
]


@dataclass(slots=True)
class SMInstruction:
    """Structured instruction payload consumed by :class:`StreamingMultiprocessor`."""

    type: InstructionType
    voiceover: str = ""
    display: str | None = None
    thread: int | None = None
    global_address: int | None = None
    shared_index: int | None = None
    dest: int | None = None
    src: int | None = None
    src_a: int | None = None
    src_b: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "type": self.type,
            "voiceover": self.voiceover,
            "display": self.display,
            "thread": self.thread,
            "global_address": self.global_address,
            "shared_index": self.shared_index,
            "dest": self.dest,
            "src": self.src,
            "src_a": self.src_a,
            "src_b": self.src_b,
        }
        return {key: value for key, value in payload.items() if value is not None}


InstructionLike = Union[Mapping[str, Any], SMInstruction]


@dataclass(slots=True)
class PreparedInstruction:
    """Container for per-thread animation assets."""

    tokens: tuple[VMobject, ...] = ()
    pre_animation: Animation | None = None
    post_animation: Animation | None = None
    after_pre: Callable[[], None] | None = None


class StreamingMultiprocessor(VGroup):
    """Visualization helper for a simplified CUDA Streaming Multiprocessor.

    Parameters
    ----------
    num_threads:
        Number of thread lanes to display.
    num_registers_per_thread:
        Number of register slots per thread.
    shared_mem_size:
        Number of addressable shared-memory slots.
    global_mem_size:
        Number of addressable global-memory slots.
    global_color_map:
        Sequence of color names/values indexed by global-memory address.

    Instruction Format
    ------------------
    The :meth:`execute` method consumes a sequence of "SIMT steps". Each step is
    an iterable whose length must equal ``num_threads`` and that supplies either
    an instruction or ``None`` for every thread lane. Instructions may be
    provided as :class:`SMInstruction` instances or plain ``dict`` objects with
    the same keys. The ``thread`` key is optional—if omitted, the lane index of
    the tuple position is used automatically. If multiple threads include a
    ``voiceover`` string within the same step, the strings are concatenated and
    spoken via
    ``scene.voiceover(...)`` when available (matching the narration style used
    elsewhere in :mod:`matmul_scene`), while the on-screen caption updates in
    sync with the audio.

    Supported instruction ``type`` values (keys in parentheses are required):

    ``"global_to_shared"`` (``global_address``, ``shared_index``)
        Copy a value from global memory into a shared-memory slot.
    ``"load_global"`` (``dest``, ``global_address``)
        Load a global-memory value into a thread register.
    ``"load_shared"`` (``dest``, ``shared_index``)
        Load a shared-memory value into a thread register.
    ``"op"`` (``src_a``, ``src_b``, ``dest``)
        Combine two registers and write the blended result into ``dest``.
    ``"store_shared"`` (``src``, ``shared_index``)
        Write a register value into shared memory.
    ``"store_global"`` (``src``, ``global_address``)
        Write a register value back to global memory.
    ``"narrate"`` (no additional keys required)
        Update the narration without changing memory/register state.
    """

    def __init__(
        self,
        num_threads: int,
        num_registers_per_thread: int,
        shared_mem_size: int,
        global_mem_size: int,
        global_color_map: Sequence[str],
        *,
        thread_labels: Sequence[str] | None = None,
        show_counters: bool = False,
    ):
        super().__init__()

        if num_threads <= 0:
            msg = "num_threads must be positive."
            raise ValueError(msg)
        if num_registers_per_thread <= 0:
            msg = "num_registers_per_thread must be positive."
            raise ValueError(msg)
        if shared_mem_size < 0:
            msg = "shared_mem_size cannot be negative."
            raise ValueError(msg)
        if global_mem_size <= 0:
            msg = "global_mem_size must be positive."
            raise ValueError(msg)
        if len(global_color_map) != global_mem_size:
            msg = (
                "global_color_map length must match global_mem_size "
                f"({len(global_color_map)} != {global_mem_size})."
            )
            raise ValueError(msg)

        self.num_threads = num_threads
        self.num_registers_per_thread = num_registers_per_thread
        self.shared_mem_size = shared_mem_size
        self.global_mem_size = global_mem_size
        self.thread_labels = list(thread_labels or [])
        if self.thread_labels and len(self.thread_labels) != num_threads:
            msg = "thread_labels length must equal num_threads."
            raise ValueError(msg)

        self.global_color_map = global_color_map
        self._initial_global_colors = global_color_map.copy()
        self._instruction_font_size = 18
        self.global_slots: list[SlotVisual] = []
        self.shared_slots: list[SlotVisual] = []
        self.register_slots: list[list[SlotVisual]] = [[] for _ in range(num_threads)]
        self.thread_instruction_boxes: list[RoundedRectangle] = []
        self.thread_instruction_texts: list[MathTex] = []
        self.thread_headers: list[Text] = []
        self.show_counters = show_counters
        self._counter_specs: tuple[tuple[str, str], ...] = (
            ("ops", "FLOPs"),
            ("global", "Global Accesses"),
            ("shared", "Shared Accesses"),
        )
        self._counter_label_map: dict[str, str] = {
            key: label for key, label in self._counter_specs
        }
        self.thread_counter_values: list[dict[str, int]] = [
            {key: 0 for key, _ in self._counter_specs} for _ in range(num_threads)
        ]
        self.thread_counter_texts: list[dict[str, Text]] = [
            {} for _ in range(num_threads)
        ]
        self.thread_counter_boxes: list[RoundedRectangle | None] = []
        self.thread_counter_groups: list[VGroup | None] = []
        self._counter_font_size = 16

        self._build_sections()
        self._populate_global_memory()
        self._populate_shared_memory()
        self._populate_threads()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear shared memory and register values."""
        for slot in self.shared_slots:
            self._clear_slot(slot)
        for thread_slots in self.register_slots:
            for slot in thread_slots:
                self._clear_slot(slot)
        for text in self.thread_instruction_texts:
            reset_tex = self._build_instruction_tex(
                "Idle", font_size=self._instruction_font_size
            )
            reset_tex.move_to(text)
            text.become(reset_tex)
            text.display_text = "Idle"
        self.global_color_map = self._initial_global_colors
        for slot, color in zip(self.global_slots, self.global_color_map):
            self._set_slot_value(slot, color)
        for thread_idx, counters in enumerate(self.thread_counter_values):
            for key in counters:
                counters[key] = 0
                self._set_counter_value(thread_idx, key, 0, animate=False)

    def execute(
        self,
        scene: Scene,
        steps: Sequence[Sequence[InstructionLike | None]],
    ) -> None:
        """Animate the provided instruction sequence.

        Each element in ``steps`` represents a single SIMT time step and must be an
        iterable with length equal to ``num_threads``. Provide an instruction (or
        ``None``) for every thread position. Instructions within the same tuple are
        animated in parallel.
        """

        for step_index, raw_step in enumerate(steps):
            if len(raw_step) != self.num_threads:
                msg = (
                    f"Step {step_index} expected {self.num_threads} thread entries, "
                    f"received {len(raw_step)}."
                )
                raise ValueError(msg)

            voiceovers: list[str] = []
            voiceover_present = False
            pre_anims: list[Animation] = []
            post_anims: list[Animation] = []
            tokens_to_add: list[VMobject] = []
            after_pre_callbacks: list[Callable[[], None]] = []

            for thread_idx, inst_like in enumerate(raw_step):
                if inst_like is None:
                    transform, highlight = self._prepare_thread_instruction_display(
                        thread_idx,
                        "Idle",
                        highlight=False,
                    )
                    if transform:
                        pre_anims.append(transform)
                    if highlight:
                        post_anims.append(highlight)
                    continue

                instruction = (
                    inst_like.to_payload()
                    if isinstance(inst_like, SMInstruction)
                    else dict(inst_like)
                )
                instruction.setdefault("thread", thread_idx)
                if instruction["thread"] != thread_idx:
                    msg = (
                        f"Instruction thread mismatch: expected {thread_idx}, "
                        f"received {instruction['thread']}."
                    )
                    raise ValueError(msg)

                step_type = instruction.get("type")
                if step_type is None:
                    raise ValueError("Instruction missing 'type' key.")

                voice_text = instruction.get("voiceover")
                if voice_text is not None:
                    voiceover_present = True
                    voiceovers.append(voice_text)

                if step_type == "narrate":
                    display_text = instruction.get("display")
                    if display_text:
                        transform, highlight = self._prepare_thread_instruction_display(
                            thread_idx,
                            display_text,
                            highlight=False,
                        )
                        if transform:
                            pre_anims.append(transform)
                        if highlight:
                            post_anims.append(highlight)
                    continue

                display_text = instruction.get("display")
                if display_text is None:
                    display_text = self._default_display_text(instruction)

                transform, highlight = self._prepare_thread_instruction_display(
                    thread_idx,
                    display_text,
                    highlight=True,
                )
                if transform:
                    pre_anims.append(transform)
                if highlight:
                    post_anims.append(highlight)

                prepared = self._prepare_instruction(instruction)

                if prepared.tokens:
                    tokens_to_add.extend(prepared.tokens)
                if prepared.pre_animation:
                    pre_anims.append(prepared.pre_animation)
                if prepared.post_animation:
                    post_anims.append(prepared.post_animation)
                if prepared.after_pre:
                    after_pre_callbacks.append(prepared.after_pre)

                counter_anim = self._accumulate_counters(thread_idx, instruction)
                if counter_anim:
                    post_anims.append(counter_anim)

            voiceover_text = " ".join(voiceovers).strip() if voiceover_present else ""

            def run_step() -> None:
                if tokens_to_add:
                    scene.add(*tokens_to_add)

                if pre_anims:
                    scene.play(AnimationGroup(*pre_anims, lag_ratio=0.0))

                for callback in after_pre_callbacks:
                    callback()

                if post_anims:
                    scene.play(AnimationGroup(*post_anims, lag_ratio=0.0))

                scene.wait(0.2)

            if voiceover_present and hasattr(scene, "voiceover"):
                with scene.voiceover(text=voiceover_text):
                    run_step()
            else:
                run_step()

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------
    def _build_sections(self) -> None:
        frame_w = config.frame_width
        frame_h = config.frame_height
        margin = 0.4

        global_height = min(frame_h * 0.14, 2.2)
        upper_height = frame_h - 2 * margin - global_height
        if upper_height <= 1.5:
            upper_height = frame_h * 0.55
            global_height = frame_h - 2 * margin - upper_height

        upper_top = frame_h / 2 - margin
        upper_bottom = upper_top - upper_height
        upper_center_y = (upper_top + upper_bottom) / 2

        global_width = frame_w - 2 * margin
        global_center_y = -frame_h / 2 + margin + global_height / 2

        upper_width = frame_w - 2 * margin
        shared_width = min(max(upper_width * 0.18, 2.0), upper_width * 0.35)
        threads_width = upper_width - shared_width
        if threads_width <= 1.2:
            threads_width = upper_width * 0.6
            shared_width = upper_width - threads_width

        threads_center_x = -frame_w / 2 + margin + threads_width / 2
        shared_center_x = -frame_w / 2 + margin + threads_width + shared_width / 2

        self.global_section = Rectangle(
            width=global_width,
            height=global_height,
            stroke_color=GRAY_B,
            stroke_width=3,
            stroke_opacity=0.45,
            fill_opacity=0,
        ).move_to([0, global_center_y, 0])

        self.shared_section = Rectangle(
            width=shared_width,
            height=upper_height,
            stroke_color=GRAY_B,
            stroke_width=3,
            stroke_opacity=0.45,
            fill_opacity=0,
        ).move_to([shared_center_x, upper_center_y, 0])

        self.thread_section = Rectangle(
            width=threads_width,
            height=upper_height,
            stroke_color=GRAY_B,
            stroke_width=3,
            stroke_opacity=0.45,
            fill_opacity=0,
        ).move_to([threads_center_x, upper_center_y, 0])

        self.add(self.global_section, self.shared_section, self.thread_section)

    def _populate_global_memory(self) -> None:
        if self.global_mem_size == 0:
            return

        slots: list[SlotVisual] = []
        base_side = 0.5
        for color in self.global_color_map:
            base_square = Square(side_length=base_side)
            base_square.set_stroke(GRAY_D, width=1.2, opacity=0.45)
            base_square.set_fill(BLACK, opacity=0.05)
            value_square = Square(side_length=base_side * 0.76)
            value_square.set_fill(color, opacity=0.9)
            value_square.set_stroke(color, width=0)
            value_square.move_to(base_square)
            slot_group = VGroup(base_square, value_square)
            slot = SlotVisual(
                box=base_square,
                group=slot_group,
                side=base_side,
                value_square=value_square,
                color=color,
            )
            slots.append(slot)

        grid = VGroup(*(slot.group for slot in slots))
        grid.arrange(RIGHT, buff=0.12, aligned_edge=DOWN)
        available_width = self.global_section.width - 0.6
        if grid.width > available_width:
            grid.scale_to_fit_width(available_width)
        grid.move_to(self.global_section.get_center())

        for slot in slots:
            slot.side = slot.box.width

        self.global_slots = slots
        self.global_grid = grid
        self.add(self.global_grid)

    def _populate_shared_memory(self) -> None:
        slots: list[SlotVisual] = []
        base_side = 0.5
        for idx in range(self.shared_mem_size):
            base_square = Square(side_length=base_side)
            base_square.set_stroke(GRAY_D, width=1.2, opacity=0.35)
            base_square.set_fill(BLACK, opacity=0.05)
            slot_group = VGroup(base_square)
            label = Text(f"s{idx}", font_size=18)
            label.scale_to_fit_height(base_square.height * 0.6)
            label.next_to(base_square, LEFT, buff=0.08)
            label.align_to(base_square, UP)
            slot_group.add(label)
            slot = SlotVisual(
                box=base_square,
                group=slot_group,
                side=base_side,
                label=label,
            )
            slots.append(slot)

        if slots:
            grid = VGroup(*(slot.group for slot in slots))
            grid.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
            available_height = self.shared_section.height - 0.6
            if grid.height > available_height:
                grid.scale_to_fit_height(available_height)
            grid.move_to(self.shared_section.get_center())
            grid.align_to(self.shared_section.get_top(), UP)
            grid.shift(DOWN * 0.4)

            for slot in slots:
                slot.side = slot.box.width

            self.shared_slots = slots
            self.shared_grid = grid
            self.add(self.shared_grid)

    def _populate_threads(self) -> None:
        threads_height = self.thread_section.height - 0.6
        thread_height = threads_height / self.num_threads
        thread_width = self.thread_section.width
        top_edge = self.thread_section.get_top()[1] - 0.3

        for thread_idx in range(self.num_threads):
            center_y = top_edge - thread_height / 2 - thread_idx * thread_height
            thread_rect = Rectangle(
                width=thread_width,
                height=thread_height,
                stroke_color=GRAY_D,
                stroke_opacity=0,
                stroke_width=0,
                fill_opacity=0,
            ).move_to([self.thread_section.get_center()[0], center_y, 0])
            self.add(thread_rect)

            if thread_idx > 0:
                boundary_y = thread_rect.get_top()[1]
                divider = Line(LEFT, RIGHT)
                divider.set_width(thread_rect.width * 0.95)
                divider.set_stroke(GRAY_D, width=1.0, opacity=0.3)
                divider.move_to([self.thread_section.get_center()[0], boundary_y, 0])
                self.add(divider)

            header_text = (
                self.thread_labels[thread_idx]
                if thread_idx < len(self.thread_labels)
                else f"Thread {thread_idx}"
            )
            instruction_box = RoundedRectangle(
                corner_radius=0.15,
                width=thread_rect.width - 0.8,
                height=min(0.5, thread_height * 0.22),
            )
            instruction_box.set_stroke(BLUE_C, width=1.6, opacity=0.45)
            instruction_box.set_fill(BLACK, opacity=0.1)
            instruction_box.next_to(thread_rect.get_top(), DOWN, buff=0.18)
            instruction_box.align_to(thread_rect.get_left(), LEFT)
            instruction_box.shift(RIGHT * 0.25)

            header = Text(header_text, font_size=16)
            header.set_opacity(0.9)
            max_header_height = instruction_box.height * 0.65
            if header.height > max_header_height:
                header.scale_to_fit_height(max_header_height)
            header.next_to(instruction_box.get_left(), RIGHT, buff=0.12)
            header.align_to(instruction_box, UP)
            header.shift(DOWN * 0.05)

            instruction_text = self._build_instruction_tex(
                "Idle", font_size=self._instruction_font_size
            )
            instruction_text.display_text = "Idle"
            max_instr_width = instruction_box.width - 0.3
            if instruction_text.width > max_instr_width:
                instruction_text.scale_to_fit_width(max_instr_width)
            instruction_text.move_to(instruction_box.get_center())
            instruction_text.shift(RIGHT * 0.5)

            register_area_height = (
                thread_rect.height
                - (instruction_box.get_top()[1] - instruction_box.get_bottom()[1])
                - (header.get_top()[1] - header.get_bottom()[1])
                - 0.7
            )
            register_count = self.num_registers_per_thread
            cols = register_count if register_count > 0 else 1
            rows = 1
            base_side = 0.5
            register_slots: list[SlotVisual] = []
            for reg_idx in range(register_count):
                base_square = Square(side_length=base_side)
                base_square.set_stroke(GRAY_D, width=1.1, opacity=0.35)
                base_square.set_fill(BLACK, opacity=0.05)
                slot_group = VGroup(base_square)
                label = Text(f"r{reg_idx}", font_size=20)
                label.scale_to_fit_height(base_square.height * 0.35)
                label.next_to(base_square, DOWN, buff=0.05)
                slot_group.add(label)
                slot = SlotVisual(
                    box=base_square,
                    group=slot_group,
                    side=base_side,
                    label=label,
                )
                register_slots.append(slot)

            reserved_for_counters = (
                min(thread_rect.width * 0.32, 1.8) if self.show_counters else 0.0
            )
            registers_group = VGroup(*(slot.group for slot in register_slots))
            registers_group.arrange_in_grid(
                rows=rows, cols=cols, buff=0.12, center=True
            )
            base_max_width = thread_rect.width - 0.8
            max_width = (
                max(base_max_width - reserved_for_counters, 1.2)
                if self.show_counters
                else max(base_max_width, 1.2)
            )
            max_height = max(register_area_height, 0.55)
            if registers_group.width > max_width:
                registers_group.scale_to_fit_width(max_width)
            if registers_group.height > max_height:
                registers_group.scale_to_fit_height(max_height)
            registers_group.next_to(instruction_box, DOWN, buff=0.25)
            registers_group.align_to(instruction_box, LEFT)

            for slot in register_slots:
                slot.side = slot.box.width

            counter_box: RoundedRectangle | None = None
            counter_group: VGroup | None = None
            if self.show_counters:
                counter_box_width = max(reserved_for_counters, 1.2)
                available_height = thread_rect.height - 0.3
                counter_box_height = max(
                    registers_group.height + 0.35,
                    instruction_box.height + 0.3,
                    1.0,
                )
                counter_box_height = min(counter_box_height, available_height)
                counter_box = RoundedRectangle(
                    corner_radius=0.12,
                    width=counter_box_width,
                    height=counter_box_height,
                )
                counter_box.set_stroke(GRAY_B, width=1.2, opacity=0.35)
                counter_box.set_fill(BLACK, opacity=0.08)
                counter_box.move_to(thread_rect.get_center())
                counter_box.align_to(thread_rect, RIGHT)
                counter_box.shift(LEFT * 0.25)
                counter_box.align_to(instruction_box, UP)

                counter_texts: dict[str, Text] = {}
                counter_text_list: list[Text] = []
                for key, label in self._counter_specs:
                    display_text = f"{label}: 0"
                    counter_text = Text(display_text, font_size=self._counter_font_size)
                    max_line_width = counter_box.width - 0.3
                    if counter_text.width > max_line_width:
                        counter_text.scale_to_fit_width(max_line_width)
                    counter_texts[key] = counter_text
                    counter_text.counter_value = 0  # type: ignore[attr-defined]
                    counter_text_list.append(counter_text)

                counter_group = VGroup(*counter_text_list)
                if counter_text_list:
                    counter_group.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
                    counter_group.move_to(counter_box.get_center())
                    counter_group.align_to(counter_box, LEFT)
                    counter_group.align_to(counter_box, UP)
                    counter_group.shift(DOWN * 0.18 + RIGHT * 0.12)

                self.thread_counter_texts[thread_idx] = counter_texts
            else:
                self.thread_counter_texts[thread_idx] = {}

            self.thread_counter_boxes.append(counter_box)
            self.thread_counter_groups.append(counter_group)

            self.register_slots[thread_idx] = register_slots
            self.thread_instruction_boxes.append(instruction_box)
            self.thread_instruction_texts.append(instruction_text)
            self.thread_headers.append(header)

            to_add: list[Mobject] = [
                header,
                instruction_box,
                instruction_text,
                registers_group,
            ]
            if self.show_counters and counter_box and counter_group:
                to_add.extend([counter_box, counter_group])
            self.add(*to_add)

    # ------------------------------------------------------------------
    # Instruction helpers
    # ------------------------------------------------------------------
    def _instr_global_to_shared(self, step: Mapping[str, Any]) -> PreparedInstruction:
        address = step.get("global_address")
        shared_index = step.get("shared_index")
        if address is None or shared_index is None:
            raise ValueError(
                "global_to_shared instruction requires global_address and shared_index."
            )
        if not (0 <= address < self.global_mem_size):
            msg = f"global_address {address} out of bounds."
            raise ValueError(msg)
        if not (0 <= shared_index < len(self.shared_slots)):
            msg = f"shared_index {shared_index} out of bounds."
            raise ValueError(msg)

        source = self.global_slots[address]
        target = self.shared_slots[shared_index]

        if source.value_square is None:
            return PreparedInstruction()

        token = source.value_square.copy()
        flash = self._flash_slot(source)
        move_anim = AnimationGroup(
            flash,
            Indicate(source.group, color=YELLOW_B, scale_factor=1.02, run_time=0.4),
            token.animate.move_to(target.center()),
            lag_ratio=0.0,
        )

        def after_move() -> None:
            self._set_slot_value(target, source.color or WHITE, arriving=token)

        highlight = Indicate(
            target.group, color=YELLOW_B, scale_factor=1.05, run_time=0.5
        )
        return PreparedInstruction(
            tokens=(token,),
            pre_animation=move_anim,
            post_animation=highlight,
            after_pre=after_move,
        )

    def _instr_load_global(self, step: Mapping[str, Any]) -> PreparedInstruction:
        thread = step.get("thread")
        if thread is None:
            raise ValueError("load_global instruction requires a thread index.")
        dest = step.get("dest")
        address = step.get("global_address")
        if dest is None or address is None:
            raise ValueError(
                "load_global instruction requires dest and global_address."
            )
        if not (0 <= address < self.global_mem_size):
            msg = f"global_address {address} out of bounds."
            raise ValueError(msg)

        source = self.global_slots[address]
        target = self._get_register_slot(thread, dest)
        if source.value_square is None:
            return PreparedInstruction()

        token = source.value_square.copy()
        flash = self._flash_slot(source)
        move_anim = AnimationGroup(
            flash,
            Indicate(source.group, color=YELLOW_B, scale_factor=1.02, run_time=0.4),
            token.animate.move_to(target.center()),
            lag_ratio=0.0,
        )

        def after_move() -> None:
            self._set_slot_value(target, source.color or WHITE, arriving=token)

        highlight = Indicate(
            target.group, color=YELLOW_B, scale_factor=1.05, run_time=0.5
        )
        return PreparedInstruction(
            tokens=(token,),
            pre_animation=move_anim,
            post_animation=highlight,
            after_pre=after_move,
        )

    def _instr_load_shared(self, step: Mapping[str, Any]) -> PreparedInstruction:
        thread = step.get("thread")
        if thread is None:
            raise ValueError("load_shared instruction requires a thread index.")
        dest = step.get("dest")
        shared_index = step.get("shared_index")
        if dest is None or shared_index is None:
            raise ValueError("load_shared instruction requires dest and shared_index.")
        if not (0 <= shared_index < len(self.shared_slots)):
            msg = f"shared_index {shared_index} out of bounds."
            raise ValueError(msg)

        source = self.shared_slots[shared_index]
        target = self._get_register_slot(thread, dest)
        if source.value_square is None:
            return PreparedInstruction()

        token = source.value_square.copy()
        flash = self._flash_slot(source)
        move_anim = AnimationGroup(
            flash,
            Indicate(source.group, color=YELLOW_B, scale_factor=1.03, run_time=0.4),
            token.animate.move_to(target.center()),
            lag_ratio=0.0,
        )

        def after_move() -> None:
            self._set_slot_value(target, source.color or WHITE, arriving=token)

        highlight = Indicate(
            target.group, color=YELLOW_B, scale_factor=1.05, run_time=0.5
        )
        return PreparedInstruction(
            tokens=(token,),
            pre_animation=move_anim,
            post_animation=highlight,
            after_pre=after_move,
        )

    def _instr_op(self, step: Mapping[str, Any]) -> PreparedInstruction:
        thread = step.get("thread")
        if thread is None:
            raise ValueError("op instruction requires a thread index.")
        src_a = step.get("src_a")
        src_b = step.get("src_b")
        dest = step.get("dest")
        if None in (src_a, src_b, dest):
            raise ValueError("op instruction requires src_a, src_b, and dest.")

        slot_a = self._get_register_slot(thread, src_a)
        slot_b = self._get_register_slot(thread, src_b)
        dest_slot = self._get_register_slot(thread, dest)
        if slot_a.value_square is None or slot_b.value_square is None:
            return PreparedInstruction()

        token_a = slot_a.value_square.copy()
        token_b = slot_b.value_square.copy()
        dest_center = dest_slot.center()
        offset = dest_slot.side * 0.35

        first_phase = AnimationGroup(
            token_a.animate.move_to(dest_center + LEFT * offset),
            token_b.animate.move_to(dest_center + RIGHT * offset),
            lag_ratio=0.0,
        )
        second_phase = AnimationGroup(
            token_a.animate.move_to(dest_center),
            token_b.animate.move_to(dest_center),
            lag_ratio=0.0,
        )
        pre_anim = Succession(first_phase, second_phase)

        color_a = slot_a.color or WHITE
        color_b = slot_b.color or WHITE
        blended = interpolate_color(color_a, color_b, 0.5)

        def after_move() -> None:
            self._set_slot_value(dest_slot, blended, arriving=token_a)

        post_anim = AnimationGroup(
            FadeOut(token_b, scale=0.3),
            Indicate(dest_slot.group, color=YELLOW_B, scale_factor=1.05, run_time=0.5),
            lag_ratio=0.0,
        )
        return PreparedInstruction(
            tokens=(token_a, token_b),
            pre_animation=pre_anim,
            post_animation=post_anim,
            after_pre=after_move,
        )

    def _instr_store_shared(self, step: Mapping[str, Any]) -> PreparedInstruction:
        thread = step.get("thread")
        if thread is None:
            raise ValueError("store_shared instruction requires a thread index.")
        src = step.get("src")
        shared_index = step.get("shared_index")
        if src is None or shared_index is None:
            raise ValueError("store_shared instruction requires src and shared_index.")
        if not (0 <= shared_index < len(self.shared_slots)):
            msg = f"shared_index {shared_index} out of bounds."
            raise ValueError(msg)

        source = self._get_register_slot(thread, src)
        target = self.shared_slots[shared_index]
        if source.value_square is None:
            return PreparedInstruction()

        token = source.value_square.copy()
        move_anim = token.animate.move_to(target.center())

        def after_move() -> None:
            self._set_slot_value(target, source.color or WHITE, arriving=token)

        highlight = Indicate(
            target.group, color=YELLOW_B, scale_factor=1.05, run_time=0.5
        )
        return PreparedInstruction(
            tokens=(token,),
            pre_animation=move_anim,
            post_animation=highlight,
            after_pre=after_move,
        )

    def _instr_store_global(self, step: Mapping[str, Any]) -> PreparedInstruction:
        thread = step.get("thread")
        if thread is None:
            raise ValueError("store_global instruction requires a thread index.")
        src = step.get("src")
        address = step.get("global_address")
        if src is None or address is None:
            raise ValueError(
                "store_global instruction requires src and global_address."
            )
        if not (0 <= address < self.global_mem_size):
            msg = f"global_address {address} out of bounds."
            raise ValueError(msg)

        source = self._get_register_slot(thread, src)
        target = self.global_slots[address]
        if source.value_square is None:
            return PreparedInstruction()

        token = source.value_square.copy()
        move_anim = token.animate.move_to(target.center())

        def after_move() -> None:
            self._set_slot_value(target, source.color or WHITE, arriving=token)
            self.global_color_map[address] = target.color or WHITE

        highlight = Indicate(
            target.group, color=YELLOW_B, scale_factor=1.05, run_time=0.5
        )
        return PreparedInstruction(
            tokens=(token,),
            pre_animation=move_anim,
            post_animation=highlight,
            after_pre=after_move,
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _get_register_slot(self, thread: int, reg: int) -> SlotVisual:
        if not (0 <= reg < self.num_registers_per_thread):
            msg = f"register index {reg} out of bounds."
            raise ValueError(msg)
        return self.register_slots[thread][reg]

    def _set_slot_value(
        self,
        slot: SlotVisual,
        color: str,
        *,
        arriving: VMobject | None = None,
    ) -> None:
        normalized = color
        if slot.value_square is not None and slot.value_square is not arriving:
            slot.group.remove(slot.value_square)
        if arriving is None:
            value = Square(side_length=slot.side * 0.75)
            value.set_fill(normalized, opacity=0.9)
            value.set_stroke(normalized, width=0)
            value.move_to(slot.center())
        else:
            value = arriving
            value.set_fill(normalized, opacity=0.9)
            value.set_stroke(normalized, width=0)
            value.move_to(slot.center())

        slot.group.add(value)
        slot.value_square = value
        slot.color = normalized

    def _clear_slot(self, slot: SlotVisual) -> None:
        if slot.value_square is not None:
            slot.group.remove(slot.value_square)
            slot.value_square = None
        slot.color = None

    def _flash_slot(self, slot: SlotVisual) -> Animation:
        return Flash(
            slot.group,
            color=YELLOW_B,
            line_length=slot.side * 0.6 if slot.side else 0.3,
            num_lines=10,
            run_time=2.0,
        )

    def _accumulate_counters(
        self,
        thread: int,
        instruction: Mapping[str, Any],
    ) -> Animation | None:
        deltas = self._counter_deltas(instruction)
        if not deltas:
            return None
        animations: list[Animation] = []
        for key, delta in deltas.items():
            anim = self._increment_counter(thread, key, delta)
            if anim:
                animations.append(anim)
        if not animations:
            return None
        if len(animations) == 1:
            return animations[0]
        return AnimationGroup(*animations, lag_ratio=0.0)

    def _counter_deltas(self, instruction: Mapping[str, Any]) -> dict[str, int]:
        step_type = instruction.get("type")
        if step_type is None:
            return {}
        deltas: dict[str, int] = {}
        if step_type == "op":
            flops_raw = instruction.get("flops", 2)
            try:
                flops = int(flops_raw)
            except (TypeError, ValueError):
                flops = 2
            if flops:
                deltas["ops"] = deltas.get("ops", 0) + flops
        if step_type in {"load_global", "store_global", "global_to_shared"}:
            deltas["global"] = deltas.get("global", 0) + 1
        if step_type in {"load_shared", "store_shared"}:
            deltas["shared"] = deltas.get("shared", 0) + 1
        return deltas

    def _increment_counter(
        self,
        thread: int,
        key: str,
        delta: int,
    ) -> Animation | None:
        if delta == 0 or not (0 <= thread < len(self.thread_counter_values)):
            return None
        current = self.thread_counter_values[thread].get(key, 0)
        new_value = current + delta
        self.thread_counter_values[thread][key] = new_value
        return self._set_counter_value(thread, key, new_value, animate=True)

    def _set_counter_value(
        self,
        thread: int,
        key: str,
        value: int,
        *,
        animate: bool,
    ) -> Animation | None:
        if not (0 <= thread < len(self.thread_counter_texts)):
            return None
        text_obj = self.thread_counter_texts[thread].get(key)
        if text_obj is None:
            return None
        box = self.thread_counter_boxes[thread]
        label = self._counter_label_map.get(key, key.title())
        new_text = Text(
            f"{label}: {value}",
            font_size=self._counter_font_size,
        )
        if box is not None:
            max_line_width = box.width - 0.3
            if new_text.width > max_line_width:
                new_text.scale_to_fit_width(max_line_width)
        new_text.move_to(text_obj.get_center())
        text_obj.counter_value = value  # type: ignore[attr-defined]
        if animate:
            return Transform(text_obj, new_text)
        text_obj.become(new_text)
        return None

    def _build_instruction_tex(
        self, display_text: str, *, font_size: float | None = None
    ) -> MathTex:
        """Render instruction captions with consistent MathTex styling."""
        font_size = font_size or self._instruction_font_size
        latex_tokens: list[str] = []
        buffer: list[str] = []
        symbol_map = {"→": r"\rightarrow", "⊕": r"\oplus"}

        for char in display_text:
            replacement = symbol_map.get(char)
            if replacement is not None:
                if buffer:
                    latex_tokens.append(self._wrap_instruction_text("".join(buffer)))
                    buffer.clear()
                latex_tokens.append(replacement)
                continue
            buffer.append(char)

        if buffer:
            latex_tokens.append(self._wrap_instruction_text("".join(buffer)))
        if not latex_tokens:
            latex_tokens.append(self._wrap_instruction_text(display_text))

        math_tex = MathTex(*latex_tokens, font_size=font_size)
        math_tex.display_text = display_text
        return math_tex

    @staticmethod
    def _wrap_instruction_text(text: str) -> str:
        sanitized = StreamingMultiprocessor._sanitize_text_for_tex(text)
        return rf"\text{{{sanitized}}}"

    @staticmethod
    def _sanitize_text_for_tex(text: str) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "{": r"\{",
            "}": r"\}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(char, char) for char in text)

    def _prepare_thread_instruction_display(
        self,
        thread: int,
        display_text: str,
        *,
        highlight: bool,
    ) -> tuple[Animation | None, Animation | None]:
        if not (0 <= thread < self.num_threads):
            msg = f"thread index {thread} out of bounds."
            raise ValueError(msg)

        box = self.thread_instruction_boxes[thread]
        text_obj = self.thread_instruction_texts[thread]
        current_text = getattr(text_obj, "display_text", "")
        transform: Animation | None = None
        if display_text != current_text:
            target_text = self._build_instruction_tex(
                display_text, font_size=self._instruction_font_size
            )
            max_width = box.width - 0.3
            if target_text.width > max_width:
                target_text.scale_to_fit_width(max_width)
            target_text.move_to(box.get_center())
            transform = Transform(text_obj, target_text)
            text_obj.display_text = display_text

        highlight_anim = (
            Indicate(box, color=YELLOW_B, scale_factor=1.02, run_time=0.4)
            if highlight
            else None
        )
        return transform, highlight_anim

    def _prepare_instruction(self, step: Mapping[str, Any]) -> PreparedInstruction:
        step_type = step.get("type")
        if step_type is None:
            raise ValueError("Instruction missing 'type' key.")
        if step_type == "global_to_shared":
            return self._instr_global_to_shared(step)
        if step_type == "load_global":
            return self._instr_load_global(step)
        if step_type == "load_shared":
            return self._instr_load_shared(step)
        if step_type == "op":
            return self._instr_op(step)
        if step_type == "store_shared":
            return self._instr_store_shared(step)
        if step_type == "store_global":
            return self._instr_store_global(step)
        if step_type == "narrate":
            return PreparedInstruction()
        msg = f"Unsupported instruction type '{step_type}'."
        raise ValueError(msg)

    def _default_display_text(self, step: Mapping[str, Any]) -> str:
        step_type = step["type"]
        if step_type == "global_to_shared":
            return (
                f"Stage global[{step.get('global_address')}] → "
                f"s{step.get('shared_index')}"
            )
        if step_type == "load_global":
            return f"Load global[{step.get('global_address')}] → r{step.get('dest')}"
        if step_type == "load_shared":
            return f"Load shared s{step.get('shared_index')} → r{step.get('dest')}"
        if step_type == "op":
            return f"r{step.get('src_a')} ⊕ r{step.get('src_b')} → r{step.get('dest')}"
        if step_type == "store_shared":
            return f"Store r{step.get('src')} → s{step.get('shared_index')}"
        if step_type == "store_global":
            return f"Store r{step.get('src')} → global[{step.get('global_address')}]"
        if step_type == "narrate":
            return "Narrate"
        return step_type

    def get_counter_snapshot(self) -> list[dict[str, int]]:
        """Return a shallow copy of per-thread counter totals."""
        return [dict(counters) for counters in self.thread_counter_values]

    def get_total_counters(self) -> dict[str, int]:
        """Return totals across all threads for each tracked metric."""
        totals = {key: 0 for key, _ in self._counter_specs}
        for counters in self.thread_counter_values:
            for key in totals:
                totals[key] += counters.get(key, 0)
        return totals
