"""Matmul visualization scene for arithmetic intensity explanation."""

from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
from manim import (
    Matrix,
    MathTex,
    Tex,
    VGroup,
    SurroundingRectangle,
    BackgroundRectangle,
    FadeIn,
    FadeOut,
    Write,
    Create,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    UR,
)


class MatmulArithmeticIntensityScene(VoiceoverScene):
    """Visualize how a single entry of C is produced from matrices A and B."""

    def title(self) -> None:
        title = Tex("Optimizing Machine Learning Models").scale(1.5)
        v1 = """
        Welcome to my series on Kernel Optimization for Machine Learning. 
        In this series, my goal is to outline a system for optimizing GPU kernels in the context of taking a naive implementation of a model and optimizing it to run as fast as possible on modern hardware.

        We will start off on single GPU kernels, but eventually we will expand to multi-GPU and distributed systems.
        """
        with self.voiceover(text=v1):
            self.play(Write(title))
            self.wait(1.0)
        self.play(FadeOut(title))

    def intro(self) -> None:
        title = MathTex(
            r"\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Moved}}",
        ).scale(1.2)

        v1 = """
        The first step to understanding a kernel is to understand its arithmetic intensity, which is the ratio of floating point operations to bytes moved. 
        In this video, we will explore the arithmetic intensity of matrix multiplication.
        """

        with self.voiceover(text=v1) as tracker:
            self.play(Write(title))
        self.play(FadeOut(title))

    def construct(self) -> None:
        self.set_speech_service(GTTSService())

        self.title()
        self.intro()

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

        formula = MathTex(
            r"C_{ij}", r"= \sum_{k=1}^{K}", r"A_{ik}", r"\cdot", r"B_{kj}"
        ).to_corner(UR, buff=0.6)

        with self.voiceover(
            text=(
                "Matrices A and B multiply to form matrix C. "
                "We focus on a single entry in C."
            )
        ):
            self.play(FadeIn(a_matrix), FadeIn(b_matrix))
            self.play(FadeIn(labels[0]), FadeIn(labels[1]))
            self.play(FadeIn(c_matrix), FadeIn(labels[2]))
            self.play(Write(formula))

        a_term = formula.get_part_by_tex("A_{ik}")
        b_term = formula.get_part_by_tex("B_{kj}")
        c_term = formula.get_part_by_tex("C_{ij}")

        row_index = 1  # zero-based row index for A to highlight
        column_index = 1  # zero-based column index for B to highlight

        a_row = a_matrix.get_rows()[row_index]
        b_column = b_matrix.get_columns()[column_index]

        row_highlight = SurroundingRectangle(a_row, buff=0.15, color="#FF6B6B")
        column_highlight = SurroundingRectangle(b_column, buff=0.15, color="#4ECDC4")

        a_term_highlight = BackgroundRectangle(
            a_term, buff=0.05, color="#FF6B6B", fill_opacity=0.3, stroke_width=0
        )
        b_term_highlight = BackgroundRectangle(
            b_term, buff=0.05, color="#4ECDC4", fill_opacity=0.3, stroke_width=0
        )
        c_term_highlight = BackgroundRectangle(
            c_term, buff=0.05, color="#FFD93D", fill_opacity=0.3, stroke_width=0
        )

        a_term_highlight.set_z_index(-1)
        b_term_highlight.set_z_index(-1)
        c_term_highlight.set_z_index(-1)
        formula.set_z_index(1)

        with self.voiceover(
            text="Highlight the row from A and the matching term in the formula."
        ):
            self.play(
                Create(row_highlight),
                FadeIn(a_term_highlight),
            )

        with self.voiceover(
            text="Now highlight the column from B and its term in the formula."
        ):
            self.play(
                Create(column_highlight),
                FadeIn(b_term_highlight),
            )

        target_entry = c_matrix.get_entries()[row_index * 3 + column_index]
        entry_highlight = SurroundingRectangle(target_entry, buff=0.18, color="#FFD93D")

        with self.voiceover(text="These combine to compute the selected entry in C."):
            self.play(Create(entry_highlight), FadeIn(c_term_highlight))
            self.wait(0.5)

        self.wait(2.0)

        # self.play(
        #     FadeOut(row_highlight),
        #     FadeOut(column_highlight),
        #     FadeOut(a_term_highlight),
        #     FadeOut(b_term_highlight),
        #     FadeOut(c_term_highlight),
        #     FadeOut(entry_highlight),
        #     FadeOut(formula),
        #     FadeOut(labels),
        #     FadeOut(a_matrix),
        #     FadeOut(b_matrix),
        #     FadeOut(c_matrix),
        # )
