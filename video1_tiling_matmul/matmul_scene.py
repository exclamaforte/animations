"""Animated walkthrough of matrix multiply arithmetic intensity concepts."""

import math
import sys
from pathlib import Path
from typing import Tuple

# Ensure project root is importable when Manim loads this module directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
from manim_types.my_manim_types import (
    LabeledRoundedRect,
    StreamingMultiprocessor,
    SMInstruction,
)
from tiled_matmul_visualization import (
    tiled_matmul_visualization,
    simple_matmul_visualization,
)

from manim_themes.manim_theme import apply_theme
from manim import (
    Axes,
    Arrow,
    BackgroundRectangle,
    BLUE_A,
    BLUE_B,
    BLUE_C,
    BLUE_D,
    Create,
    DashedLine,
    Dot,
    DOWN,
    FadeIn,
    FadeOut,
    GOLD_D,
    GREEN_A,
    GREEN_B,
    GREEN_C,
    GREEN_D,
    Group,
    LEFT,
    MathTex,
    Matrix,
    ORANGE,
    PURPLE_A,
    PURPLE_B,
    PURPLE_C,
    PURPLE_D,
    RIGHT,
    RoundedRectangle,
    SurroundingRectangle,
    Table,
    TEAL_A,
    TEAL_B,
    TEAL_C,
    TEAL_D,
    Tex,
    UP,
    UR,
    VGroup,
    Write,
    YELLOW_A,
    YELLOW_B,
    YELLOW_C,
    YELLOW_D,
)


def build_matrices() -> Tuple[Matrix, Matrix, Matrix, VGroup]:
    """Create and lay out A, B, C matrices with labels for reuse."""
    a_matrix = Matrix(
        [
            ["a_{11}", "a_{12}", "a_{13}"],
            ["a_{21}", "a_{22}", "a_{23}"],
            ["a_{31}", "a_{32}", "a_{33}"],
        ],
    )
    b_matrix = Matrix(
        [
            ["b_{11}", "b_{12}", "b_{13}"],
            ["b_{21}", "b_{22}", "b_{23}"],
            ["b_{31}", "b_{32}", "b_{33}"],
        ],
    )
    c_matrix = Matrix(
        [
            ["c_{11}", "c_{12}", "c_{13}"],
            ["c_{21}", "c_{22}", "c_{23}"],
            ["c_{31}", "c_{32}", "c_{33}"],
        ],
    )

    a_matrix.scale(0.6)
    b_matrix.scale(0.6)
    c_matrix.scale(0.6).to_edge(UP, buff=1.5)

    a_matrix.next_to(c_matrix, LEFT, buff=0.5).align_to(c_matrix, UP)
    b_matrix.next_to(c_matrix, UP, buff=0.5).align_to(c_matrix, RIGHT)

    labels = VGroup(
        Tex("A").next_to(a_matrix, LEFT),
        Tex("B").next_to(b_matrix, UP),
        Tex("C").next_to(c_matrix, RIGHT),
    )

    VGroup(a_matrix, b_matrix, c_matrix, labels).shift(DOWN * 2.5)
    return a_matrix, b_matrix, c_matrix, labels


def get_h100_specs(example_dim: int) -> list[dict]:
    return [
        {
            "precision": "FP32",
            "compute_peak": 67.0,
            "bandwidth": 3.35,
            "color": "#4C78A8",
            "element_size": 4,
        },
        {
            "precision": "FP16",
            "compute_peak": 989.5,
            "bandwidth": 3.35,
            "color": "#F58518",
            "element_size": 2,
        },
    ]


def series_intro(scene: VoiceoverScene) -> None:
    """Set the stage for the arithmetic intensity walkthrough."""

    title = Tex("Optimizing Machine Learning Models").scale(1.5)
    overview_voiceover = """
    Welcome to my series on Kernel Optimization for Machine Learning. 
    In this series, my goal is to outline a system for optimizing GPU kernels in the context of taking a naive implementation of a model and optimizing it to run as fast as possible on modern hardware.

    We will start off on single GPU kernels, but eventually we will expand to multi-GPU and distributed systems.
    """
    with scene.voiceover(text=overview_voiceover):
        scene.play(Write(title))
        scene.wait(1.0)
    scene.play(FadeOut(title))


def kernel_optimization_intro(scene: VoiceoverScene) -> None:

    intro_voiceover_1 = """
    The first step in optimizing a kernel is estimating its maximum achievable performance, assuming full hardware utilization. At full-utilization, there are broadly three possible bottlenecks we could encounter: compute, memory bandwidth, and kernel execution latency.
    """
    # Visualization: Memory vs Compute bottleneck
    width = 1.5
    memory_box = LabeledRoundedRect(
        label_text=r"\textbf{Memory-bound}",
        corner_radius=0.2,
        width=width,
        height=1.1,
        color="teal",
        text_kwargs={"font_size": 30},
    )
    compute_box = LabeledRoundedRect(
        label_text=r"\textbf{Compute-bound}",
        corner_radius=0.2,
        width=width,
        height=1.1,
        color="orange",
        text_kwargs={"font_size": 30},
    )
    latency_box = LabeledRoundedRect(
        label_text=r"\textbf{Latency-bound}",
        corner_radius=0.2,
        width=width,
        height=1.1,
        color="red",
        text_kwargs={"font_size": 30},
    )

    mem_vs_compute_group = Group(memory_box, compute_box, latency_box)
    # arrange in a row with some space between
    mem_vs_compute_group.arrange(RIGHT, buff=1.0)

    with scene.voiceover(text=intro_voiceover_1):
        scene.play(FadeIn(memory_box), FadeIn(compute_box), FadeIn(latency_box))
        scene.wait(0.5)

    with scene.voiceover(
        text="Because we generally work with large problems, and it's possible to hide latency using things like CUDA Graphs, we can ignore latency and focus on compute and memory bandwidth."
    ):
        # set latency box to lower opacity and gray color
        scene.play(
            latency_box.animate.set_fill(color="gray", opacity=0.2).set_stroke(
                color="gray", width=1
            )
        )
        scene.wait(0.5)
    scene.play(FadeOut(mem_vs_compute_group))

    intensity = MathTex(
        r"\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Moved}}",
    ).scale(1.2)
    intro_voiceover_2 = """
    For a given kernel, we can find the ratio of floating-point operations to bytes moved. This is known as the arithmetic intensity.
    Arithmetic intensity tells us how many floating-point operations we perform per byte of memory read or written.
    High arithmetic intensity means lots of useful work per data fetched.
    Low arithmetic intensity means the GPU is starved, waiting on memory.
    """
    with scene.voiceover(text=intro_voiceover_2):
        scene.play(Write(intensity))
    scene.play(FadeOut(intensity))


def roofline_model(scene: VoiceoverScene) -> None:
    """Introduce the roofline model and walk through ridge calculations."""

    h100_specs = get_h100_specs(example_dim=128)
    for spec in h100_specs:
        spec["ridge_intensity"] = spec["compute_peak"] / spec["bandwidth"]

    table_data = [
        ["Precision", "Bandwidth (TB/s)", "Peak (TFLOP/s)", "Ratio (FLOPs/byte)"],
        *[
            [
                spec["precision"],
                f"{spec['bandwidth']:.2f}",
                f"{spec['compute_peak']:.1f}",
                f"{spec['ridge_intensity']:.1f}",
            ]
            for spec in h100_specs
        ],
    ]
    spec_table = (
        Table(
            table_data,
            include_outer_lines=True,
        )
        .scale(0.55)
        .to_edge(LEFT, buff=0.8)
        .shift(UP * 0.6)
    )
    header_row = spec_table.get_rows()[0]
    header_row.set_fill("#2a2a2a", opacity=1.0)
    header_row.set_stroke(width=0.0)
    header_row.set_color("#f5f5f5")

    max_ratio = max(spec["ridge_intensity"] for spec in h100_specs) * 1.1
    max_compute = max(spec["compute_peak"] for spec in h100_specs) * 1.1
    bandwidth = h100_specs[0]["bandwidth"]

    axes = Axes(
        x_range=[0, max_ratio, max_ratio / 6],
        y_range=[0, max_compute, max_compute / 10],
        x_length=5.6,
        y_length=3.6,
        axis_config={
            "include_tip": True,
            "include_numbers": False,
            "include_ticks": False,
            "stroke_width": 2.5,
        },
    )
    axes_group = VGroup(
        axes,
        axes.get_axis_labels(
            Tex("Arithmetic Intensity (FLOPs/byte)").scale(0.55),
            Tex("Performance (TFLOP/s)").scale(0.55),
        ),
    ).to_edge(RIGHT, buff=0.9)

    memory_line = axes.plot(
        lambda x: bandwidth * x,
        x_range=[0, max_ratio],
        use_smoothing=False,
        color="#bbbbbb",
    )

    plotted_specs = []
    for spec in h100_specs:
        ratio = spec["ridge_intensity"]
        compute_peak = spec["compute_peak"]
        horizontal = DashedLine(
            axes.c2p(0, compute_peak),
            axes.c2p(ratio, compute_peak),
            color=spec["color"],
        )
        vertical = DashedLine(
            axes.c2p(ratio, 0),
            axes.c2p(ratio, compute_peak),
            color=spec["color"],
            dash_length=0.2,
        ).set_stroke(width=2.5)
        ridge_point = Dot(axes.c2p(ratio, compute_peak), color=spec["color"])
        ratio_label = (
            MathTex(rf"{ratio:.1f}")
            .scale(0.45)
            .set_color(spec["color"])
            .next_to(axes.c2p(ratio, 0), DOWN, buff=0.12)
        )
        plotted_specs.append(
            {
                "spec": spec,
                "horizontal": horizontal,
                "vertical": vertical,
                "ridge_point": ridge_point,
                "ratio_label": ratio_label,
            }
        )

    fp32_spec = next(
        spec for spec in plotted_specs if spec["spec"]["precision"] == "FP32"
    )

    table_voiceover = """
    Let's anchor on the H100 specs. FP32 and FP16 share the same high-bandwidth memory, so their bandwidth column matches. The peak compute throughput jumps from sixty seven to nine hundred eighty nine teraFLOP per second, and dividing by bandwidth exposes the ridge intensity for each precision.
    """
    with scene.voiceover(text=table_voiceover):
        scene.play(Create(spec_table))

    axes_voiceover = """
    On the right we will keep a simplified roofline. No tick marks, just intensity along the base and performance up the side so the relationships stay front and center.
    """
    with scene.voiceover(text=axes_voiceover):
        scene.play(Create(axes_group))
        scene.play(Create(memory_line))

    ridge_voiceover = """
    The memory slope starts at the origin and rises with three point three five terabytes per second. Each compute roof appears as a horizontal line, and where it meets the slope we drop a dotted guide down to the x-axis. That intensity is the crossover between memory-bound and compute-bound execution for that precision.
    """
    with scene.voiceover(text=ridge_voiceover):
        for entry in plotted_specs:
            scene.play(Create(entry["horizontal"]))
            scene.play(FadeIn(entry["ridge_point"]))
            scene.play(Create(entry["vertical"]))
            scene.play(Write(entry["ratio_label"]))

    ratios_voiceover = """
    Those ratios, about twenty FLOPs per byte for FP32 and almost two hundred ninety five for FP16, match the last column of our table. They are the ridge points that separate the two regimes.
    """
    with scene.voiceover(text=ratios_voiceover):
        scene.wait(0.5)

    square_formula = (
        MathTex(r"\text{AI}_{\text{square}} = \frac{K}{3 \cdot \text{sizeof(element)}}")
        .scale(0.7)
        .to_edge(LEFT, buff=0.8)
    )
    intensity_target = (
        MathTex(r"\frac{K}{3 \cdot \text{sizeof(element)}} = 20")
        .scale(0.7)
        .next_to(square_formula, DOWN, aligned_edge=LEFT, buff=0.4)
    )
    solution_group = (
        VGroup(
            MathTex(r"K_{\text{FP32}} = 20 \cdot 3 \cdot 4 = 240").scale(0.65),
            MathTex(r"K_{\text{FP16}} = 20 \cdot 3 \cdot 2 = 120").scale(0.65),
        )
        .arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        .next_to(intensity_target, DOWN, aligned_edge=LEFT, buff=0.35)
    )

    formula_voiceover = """
    We can relate that ridge back to matrix size. For square GEMMs the arithmetic intensity is K divided by three times the element size. Setting that equal to twenty FLOPs per byte yields K equals two hundred forty for FP32 and one hundred twenty for FP16. Larger tiles creep up and to the right, smaller ones drift left toward the memory wall.
    """
    with scene.voiceover(text=formula_voiceover):
        scene.play(FadeOut(spec_table))
        scene.play(Write(square_formula))
        scene.play(Write(intensity_target))
        scene.play(Write(solution_group))

    achieved_perf_text = (
        Tex("Achieved FLOPs = FLOPs $\\div$ runtime")
        .scale(0.6)
        .next_to(solution_group, DOWN, aligned_edge=LEFT, buff=0.5)
    )
    achieved_example = (
        MathTex(
            r"\frac{2 \cdot 240^3}{5.2 \times 10^{-7}\ \text{s}} \approx 53.6\ \text{TFLOP/s}"
        )
        .scale(0.6)
        .next_to(achieved_perf_text, DOWN, aligned_edge=LEFT, buff=0.3)
    )
    achieved_perf = 0.8 * fp32_spec["spec"]["compute_peak"]
    achieved_dot = Dot(
        axes.c2p(fp32_spec["spec"]["ridge_intensity"], achieved_perf), color="#ffd447"
    )
    achieved_label = (
        Tex(r"$\approx 80\%$ of FP32 roof")
        .scale(0.5)
        .next_to(achieved_dot, UR, buff=0.1)
    )

    achieved_voiceover = """
    Measured performance lands by dividing the total FLOPs by the kernel runtime. This example matmul clocks fifty three point six teraFLOP per second, about eighty percent of the FP32 roof. We place it on the ridge because its tile size matches that crossover intensity.
    """
    with scene.voiceover(text=achieved_voiceover):
        scene.play(Write(achieved_perf_text))
        scene.play(Write(achieved_example))
        scene.play(FadeIn(achieved_dot), FadeIn(achieved_label))

    next_steps_header = Tex("After You Find the Regime").scale(0.6)
    next_steps_todo = Tex("TODO: Fill in follow-up actions for each regime.").scale(
        0.55
    )
    next_steps_group = (
        VGroup(next_steps_header, next_steps_todo)
        .arrange(DOWN, buff=0.2)
        .to_edge(DOWN, buff=0.8)
    )

    todo_voiceover = """
    Once we know which side we landed on, the next step is to outline the playbook. I'll drop in a reminder here to fill that in.
    """
    with scene.voiceover(text=todo_voiceover):
        scene.play(FadeIn(next_steps_group))

    scene.wait(0.5)


def matmul_element_focus(scene: VoiceoverScene) -> None:
    """Focus on how a single C entry is formed."""

    a_matrix, b_matrix, c_matrix, labels = build_matrices()
    formula = MathTex(
        r"C_{ij}", r"= \sum_{k=1}^{K}", r"A_{ik}", r"\cdot", r"B_{kj}"
    ).to_corner(UR, buff=0.6)

    intro_voiceover = """
    Matrices A and B multiply to form matrix C. 
    We focus on a single entry in C and follow the data it depends on.
    """
    with scene.voiceover(text=intro_voiceover):
        scene.play(FadeIn(a_matrix), FadeIn(b_matrix))
        scene.play(FadeIn(labels[0]), FadeIn(labels[1]))
        scene.play(FadeIn(c_matrix), FadeIn(labels[2]))
        scene.play(Write(formula))

    a_term = formula.get_part_by_tex("A_{ik}")
    b_term = formula.get_part_by_tex("B_{kj}")
    c_term = formula.get_part_by_tex("C_{ij}")

    row_index = 1
    column_index = 1

    a_row = a_matrix.get_rows()[row_index]
    b_column = b_matrix.get_columns()[column_index]

    row_highlight = SurroundingRectangle(a_row, buff=0.15, color="red")
    column_highlight = SurroundingRectangle(b_column, buff=0.15, color="teal")

    a_term_highlight = BackgroundRectangle(
        a_term, buff=0.05, color="red", fill_opacity=0.3, stroke_width=0
    )
    b_term_highlight = BackgroundRectangle(
        b_term, buff=0.05, color="teal", fill_opacity=0.3, stroke_width=0
    )
    c_term_highlight = BackgroundRectangle(
        c_term, buff=0.05, color="yellow", fill_opacity=0.3, stroke_width=0
    )

    a_term_highlight.set_z_index(-1)
    b_term_highlight.set_z_index(-1)
    c_term_highlight.set_z_index(-1)
    formula.set_z_index(1)

    row_voiceover = """
    First, take the row from A that aligns with our output position and match it to the A term in the summation.
    """
    with scene.voiceover(text=row_voiceover):
        scene.play(Create(row_highlight), FadeIn(a_term_highlight))

    column_voiceover = """
    Then gather the column from B and link it with the B term in the formula.
    """
    with scene.voiceover(text=column_voiceover):
        scene.play(Create(column_highlight), FadeIn(b_term_highlight))

    target_entry = c_matrix.get_entries()[row_index * 3 + column_index]
    entry_highlight = SurroundingRectangle(target_entry, buff=0.18, color="yellow")

    combine_voiceover = """
    These pairs of elements multiply, sum across K, and land in the target element of C.
    """
    with scene.voiceover(text=combine_voiceover):
        scene.play(Create(entry_highlight), FadeIn(c_term_highlight))
        scene.wait(0.5)
    scene.play(
        FadeOut(a_term_highlight),
        FadeOut(b_term_highlight),
        FadeOut(c_term_highlight),
        FadeOut(row_highlight),
        FadeOut(column_highlight),
        FadeOut(entry_highlight),
        FadeOut(formula),
        FadeOut(a_matrix),
        FadeOut(b_matrix),
        FadeOut(c_matrix),
        FadeOut(labels),
    )


def matmul_arithmetic_intensity(scene: VoiceoverScene) -> None:
    """Walk through the arithmetic intensity calculation step by step."""

    a_matrix, b_matrix, c_matrix, labels = build_matrices()
    formula = MathTex(
        r"C_{ij}", r"= \sum_{k=1}^{K}", r"A_{ik}", r"\cdot", r"B_{kj}"
    ).to_corner(UR, buff=0.6)

    ops_detail = Tex(r"$1$ multiply $+$ $1$ add $= 2$ FLOPs").scale(0.8)
    ops_total = MathTex(r"\text{Ops} = 2 \cdot M \cdot N \cdot K").scale(0.8)

    memory_row = Tex(r"$M \cdot K$ elements from $A$").scale(0.8)
    memory_col = Tex(r"$N \cdot K$ elements from $B$").scale(0.8)
    memory_write = Tex(r"$M \cdot N$ elements written to $C$").scale(0.8)
    memory_elements = MathTex(r"\text{Elements touched} = (M K + N K + M N)").scale(0.8)
    memory_total = MathTex(
        r"\text{Bytes moved} = (M K + N K + M N) \cdot \text{sizeof(element)}"
    ).scale(0.8)

    ai_header = Tex("Arithmetic intensity").scale(0.8)
    ai_formula = MathTex(
        r"\text{AI} = \frac{2 M N K}{2 (M K + N K + M N) \cdot \text{sizeof(element)}}"
    ).scale(0.8)

    conclusion_header = Tex("For square matrices ($M=N=K$)").scale(0.8)
    conclusion_text = MathTex(
        r"\text{AI} \approx \frac{K}{3 \cdot \text{sizeof(element)}}"
    ).scale(0.8)
    # highlight conclusion formula

    analysis_group = VGroup(
        ops_detail,
        ops_total,
        memory_row,
        memory_col,
        memory_write,
        memory_elements,
        memory_total,
        ai_header,
        ai_formula,
        conclusion_header,
        conclusion_text,
    ).arrange(DOWN, aligned_edge=LEFT)
    analysis_group.scale(0.9).to_edge(LEFT, buff=0.5).to_edge(UP, buff=0.8)

    with scene.voiceover(
        text=(
            "Let's quantify the work for that dot product. "
            "Each term involves a multiply followed by an add, so every pair gives us two floating point operations."
        )
    ):
        scene.play(Write(formula))
        scene.play(FadeIn(ops_detail))

    ops_voiceover = """
    For an output matrix of size M by N, we repeat that across every column of B and every row of A, which contributes a factor of K in the inner loop. 
    That yields two times M times N times K floating point operations in total.
    """
    with scene.voiceover(text=ops_voiceover):
        scene.play(FadeIn(ops_total))

    memory_voiceover = """
    Each output still pulls a row from A and a column from B.
    Across the entire product we load every element of A and B, and we write every element of C back out to memory.
    """
    with scene.voiceover(text=memory_voiceover):
        scene.play(FadeIn(memory_row))

    memory_total_voiceover = """
    Counting both the reads and the writes gives M K plus N K plus M N elements.
    Multiplying by the element size converts that volume into bytes moved.
    """
    with scene.voiceover(text=memory_total_voiceover):
        scene.play(FadeIn(memory_elements))
        scene.play(FadeIn(memory_total))

    ai_voiceover = """
    Arithmetic intensity divides the floating point work by the bytes moved. 
    Plugging in those expressions yields two M N K over M K plus N K plus M N, all scaled by the element size.
    """
    with scene.voiceover(text=ai_voiceover):
        scene.play(FadeIn(ai_header))
        scene.play(Write(ai_formula))

    conclusion_voiceover = """
    For square matrices the ratio simplifies to K divided by three times the element size.
    With K equal to one hundred twenty eight that works out to roughly ten point seven FLOPs per byte in FP32 and twenty one point three in FP16.
    """
    ai_highlight = SurroundingRectangle(
        ai_formula.get_part_by_tex(
            r"\frac{2 M N K}{2 (M K + N K + M N) \cdot \text{sizeof(element)}}"
        ),
        buff=0.1,
        color="yellow",
    )
    with scene.voiceover(text=conclusion_voiceover):
        scene.play(FadeIn(conclusion_header), FadeIn(conclusion_text))
        scene.play(Create(ai_highlight))
        scene.wait(1.0)

    # fade everything out
    scene.play(
        FadeOut(analysis_group),
        FadeOut(formula),
        FadeOut(ai_highlight),
    )

    scene.wait(1.0)


class Video(VoiceoverScene):
    def setup(self):
        # Set the background color to a light beige
        theme = "Andromeda"
        # theme = "Ubuntu"
        # theme = "SleepyHollow"
        apply_theme(manim_scene=self, theme_name=theme, light_theme=True)
        self.set_speech_service(GTTSService())

    def construct(self) -> None:
        series_intro(self)

        kernel_optimization_intro(self)

        matmul_element_focus(self)

        matmul_arithmetic_intensity(self)

        roofline_model(self)

        simple_matmul_visualization(self)

        # TODO tiled matmul element focus
        # TODO for the matmul arithmetic intensity, justify the bandwidth calculation by saying there's efficient reads with tiling algorithm.
        # TODO there's an errant 2 in the denominator
        # TODO the fucking voiceover for the roofline has a ton of garbage
        # TODO the roofline model doesn't clear at the end.
        # TODO tiled matmul arithmetic intensity
        # TODO make the transition into the matmul visualization smoother
        # TODO "Second value" instead of "second row"
        # TODO "so for each 2x2 output tile, we have performed 4 reads"
        # TODO reanalyize the roofline of tiling vs normal.

        tiled_matmul_visualization(self)

        self.wait(1.0)
