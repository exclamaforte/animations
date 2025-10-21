from manim_voiceover import VoiceoverScene
from manim_types.my_manim_types import StreamingMultiprocessor, SMInstruction
from manim import (
    FadeIn,
    FadeOut,
    Text,
    Table,
    DOWN,
    UP,
    BLUE_A,
    BLUE_B,
    BLUE_C,
    BLUE_D,
    PURPLE_A,
    PURPLE_B,
    PURPLE_C,
    PURPLE_D,
    YELLOW_A,
    YELLOW_B,
    YELLOW_C,
    YELLOW_D,
)


COLOR_BLOCKS = (
    (BLUE_A, BLUE_B, BLUE_C, BLUE_D),
    (YELLOW_A, YELLOW_B, YELLOW_C, YELLOW_D),
    (PURPLE_A, PURPLE_B, PURPLE_C, PURPLE_D),
)
DEFAULT_GLOBAL_PALETTE = [
    color for block in COLOR_BLOCKS for color in block * 4
]


def tiled_matmul_visualization(scene: VoiceoverScene) -> StreamingMultiprocessor:
    """Visualize a tiled 4x4 matrix multiply with 2x2 tiles."""
    palette = DEFAULT_GLOBAL_PALETTE.copy()

    sm = StreamingMultiprocessor(
        num_threads=4,
        num_registers_per_thread=4,
        shared_mem_size=8,
        global_mem_size=len(palette),
        global_color_map=palette,
    )
    intro = (
        "Let's close by animating a four by four tiled matrix multiply with two by two "
        "blocks. Four threads cooperate to stage data in shared memory before computing "
        "their output tile."
    )
    with scene.voiceover(
        text=intro,
    ):
        scene.play(FadeIn(sm, lag_ratio=0.05, run_time=1.2))
    tile_size = 2
    matrix_width = 4
    num_tiles = matrix_width // tile_size
    block_row = block_col = 0
    thread_coords = ((0, 0), (0, 1), (1, 0), (1, 1))

    m_global_base = 0
    n_global_base = 16
    c_global_base = 32
    a_shared_base = 0
    b_shared_base = 4

    def per_thread(builder):
        return tuple(
            builder(lane, ty, tx) for lane, (ty, tx) in enumerate(thread_coords)
        )

    steps: list[tuple[SMInstruction | None, ...]] = []

    def add_narration(voice: str, display: str | None = None) -> None:
        steps.append(
            (
                SMInstruction(
                    type="narrate",
                    voiceover=voice,
                    display=display or voice,
                ),
                None,
                None,
                None,
            )
        )

    def add_sync(voice: str | None) -> None:
        steps.append(
            per_thread(
                lambda lane, ty, tx: SMInstruction(
                    type="narrate",
                    voiceover=voice if (lane == 0 and voice) else None,
                    display="__syncthreads()",
                )
            )
        )

    add_narration("Each thread zeros its accumulator register r2.", "r2 ← 0")

    def stage_a_voice(tile_iter: int) -> str:
        if tile_iter == 0:
            return "Threads cooperatively load the first tile of A into shared memory."
        return "They advance to the next tile of A along the k dimension."

    def stage_b_voice(tile_iter: int) -> str:
        if tile_iter == 0:
            return "They also fetch the matching tile of B."
        return "They fetch the next tile of B."

    def load_a_voice(tile_iter: int, k_index: int) -> str:
        if tile_iter == 0 and k_index == 0:
            return "Each thread reads its row element from the staged A tile."
        if k_index == 0:
            return "With the new tile resident, they refill r0 from A."
        return "They reuse A's tile to grab the second value."

    def load_b_voice(tile_iter: int, k_index: int) -> str:
        if tile_iter == 0 and k_index == 0:
            return "And the column value comes from B's tile."
        if k_index == 0:
            return "They pull the first column from the new B tile."
        return "Then they fetch the second column from B."

    def compute_voice(partial_index: int, total_partials: int) -> str:
        if partial_index == 1:
            return "Multiply accumulate the first partial product into r2."
        if partial_index == total_partials:
            return "Accumulate the final partial product before finishing."
        return "Accumulate the next partial product into r2."

    def post_tile_voice(tile_iter: int) -> str:
        if tile_iter == num_tiles - 1:
            return "One more barrier before committing results to C."
        return "Synchronize so the block can stage the next tiles."

    total_partials = num_tiles * tile_size
    partial_counter = 0

    for tile_iter in range(num_tiles):
        steps.append(
            per_thread(
                lambda lane, ty, tx, t=tile_iter: SMInstruction(
                    type="global_to_shared",
                    voiceover=stage_a_voice(t) if lane == 0 else None,
                    global_address=(
                        m_global_base
                        + (block_row * tile_size + ty) * matrix_width
                        + t * tile_size
                        + tx
                    ),
                    shared_index=a_shared_base + ty * tile_size + tx,
                    display=f"A[{block_row * tile_size + ty},{t * tile_size + tx}] → As[{ty},{tx}]",
                )
            )
        )
        steps.append(
            per_thread(
                lambda lane, ty, tx, t=tile_iter: SMInstruction(
                    type="global_to_shared",
                    voiceover=stage_b_voice(t) if lane == 0 else None,
                    global_address=(
                        n_global_base
                        + (t * tile_size + ty) * matrix_width
                        + block_col * tile_size
                        + tx
                    ),
                    shared_index=b_shared_base + ty * tile_size + tx,
                    display=f"B[{t * tile_size + ty},{block_col * tile_size + tx}] → Bs[{ty},{tx}]",
                )
            )
        )
        add_sync("A barrier makes the tile visible to every thread.")

        for k_index in range(tile_size):
            steps.append(
                per_thread(
                    lambda lane, ty, tx, t=tile_iter, k=k_index: SMInstruction(
                        type="load_shared",
                        voiceover=load_a_voice(t, k) if lane == 0 else None,
                        dest=0,
                        shared_index=a_shared_base + ty * tile_size + k,
                        display=f"As[{ty},{k}] → r0",
                    )
                )
            )
            steps.append(
                per_thread(
                    lambda lane, ty, tx, t=tile_iter, k=k_index: SMInstruction(
                        type="load_shared",
                        voiceover=load_b_voice(t, k) if lane == 0 else None,
                        dest=1,
                        shared_index=b_shared_base + k * tile_size + tx,
                        display=f"Bs[{k},{tx}] → r1",
                    )
                )
            )
            partial_counter += 1
            steps.append(
                per_thread(
                    lambda lane, ty, tx, count=partial_counter: SMInstruction(
                        type="op",
                        voiceover=(
                            compute_voice(count, total_partials) if lane == 0 else None
                        ),
                        src_a=0,
                        src_b=1,
                        dest=2,
                        display="r0 × r1 ⊕ r2 → r2",
                    )
                )
            )
        add_sync(post_tile_voice(tile_iter))

    steps.append(
        per_thread(
            lambda lane, ty, tx: SMInstruction(
                type="store_global",
                voiceover=(
                    "Finally, each thread commits its result to C."
                    if lane == 0
                    else None
                ),
                src=2,
                global_address=(
                    c_global_base
                    + (block_row * tile_size + ty) * matrix_width
                    + block_col * tile_size
                    + tx
                ),
                display=f"r2 → C[{block_row * tile_size + ty},{block_col * tile_size + tx}]",
            )
        )
    )
    add_narration("Our two by two tile of C is now complete.")

    sm.execute(scene, steps)
    scene.wait(1.0)
    scene.play(FadeOut(sm, run_time=0.6))
    return sm


def simple_matmul_visualization(scene: VoiceoverScene) -> StreamingMultiprocessor:
    """Visualize the naive global-memory matmul kernel."""
    palette = DEFAULT_GLOBAL_PALETTE.copy()
    sm = StreamingMultiprocessor(
        num_threads=4,
        num_registers_per_thread=4,
        shared_mem_size=0,
        global_mem_size=len(palette),
        global_color_map=palette,
    )
    intro = (
        "For comparison, let's animate the naive version where every multiply "
        "loads directly from global memory."
    )
    with scene.voiceover(text=intro):
        scene.play(FadeIn(sm, lag_ratio=0.05, run_time=1.2))

    tile_size = 2
    matrix_width = 4
    block_row = block_col = 0
    thread_coords = ((0, 0), (0, 1), (1, 0), (1, 1))

    m_global_base = 0
    n_global_base = 16
    c_global_base = 32

    def per_thread(builder):
        return tuple(
            builder(lane, ty, tx) for lane, (ty, tx) in enumerate(thread_coords)
        )

    steps: list[tuple[SMInstruction | None, ...]] = []

    def add_narration(voice: str, display: str | None = None) -> None:
        steps.append(
            (
                SMInstruction(
                    type="narrate",
                    voiceover=voice,
                    display=display or voice,
                ),
                None,
                None,
                None,
            )
        )

    add_narration("Each thread zeros its accumulator register r2.", "r2 ← 0")

    def load_a_voice(k_index: int) -> str:
        if k_index == 0:
            return "Every iteration begins with a global load from A."
        return "Another trip to global memory fetches the next A element."

    def load_b_voice(k_index: int) -> str:
        if k_index == 0:
            return "They also fetch B's element from global memory each time."
        return "And another global load for B follows immediately."

    def compute_voice(partial_index: int, total_partials: int) -> str:
        if partial_index == 1:
            return "Multiply accumulate the first product into r2."
        if partial_index == total_partials:
            return "Complete the final multiply add."
        return "Accumulate the next product from global memory."

    total_partials = matrix_width
    partial_counter = 0

    for k_index in range(matrix_width):
        steps.append(
            per_thread(
                lambda lane, ty, tx, k=k_index: SMInstruction(
                    type="load_global",
                    voiceover=load_a_voice(k) if lane == 0 else None,
                    dest=0,
                    global_address=(
                        m_global_base
                        + (block_row * tile_size + ty) * matrix_width
                        + k
                    ),
                    display=(
                        f"A[{block_row * tile_size + ty},{k}] → r0"
                    ),
                )
            )
        )
        steps.append(
            per_thread(
                lambda lane, ty, tx, k=k_index: SMInstruction(
                    type="load_global",
                    voiceover=load_b_voice(k) if lane == 0 else None,
                    dest=1,
                    global_address=(
                        n_global_base
                        + k * matrix_width
                        + (block_col * tile_size + tx)
                    ),
                    display=(
                        f"B[{k},{block_col * tile_size + tx}] → r1"
                    ),
                )
            )
        )
        partial_counter += 1
        steps.append(
            per_thread(
                lambda lane, ty, tx, count=partial_counter: SMInstruction(
                    type="op",
                    voiceover=(
                        compute_voice(count, total_partials) if lane == 0 else None
                    ),
                    src_a=0,
                    src_b=1,
                    dest=2,
                    display="r0 × r1 ⊕ r2 → r2",
                )
            )
        )

    steps.append(
        per_thread(
            lambda lane, ty, tx: SMInstruction(
                type="store_global",
                voiceover=(
                    "After all that global traffic, each thread writes C back out."
                    if lane == 0
                    else None
                ),
                src=2,
                global_address=(
                    c_global_base
                    + (block_row * tile_size + ty) * matrix_width
                    + (block_col * tile_size + tx)
                ),
                display=(
                    f"r2 → C[{block_row * tile_size + ty},"
                    f"{block_col * tile_size + tx}]"
                ),
            )
        )
    )
    add_narration("That's the naive kernel—almost every instruction touches HBM.")

    sm.execute(scene, steps)
    scene.wait(1.0)
    scene.play(FadeOut(sm, run_time=0.6))
    return sm


def compare_matmul_counters(
    scene: VoiceoverScene,
    naive_sm: StreamingMultiprocessor,
    tiled_sm: StreamingMultiprocessor,
    *,
    shared_latency_cycles: int = 25,
    global_latency_cycles: int = 500,
) -> None:
    """Summarize counters from both visualizations and estimate memory latency."""

    def format_int(value: int) -> str:
        return f"{value:,}"

    def estimate_memory_cycles(totals: dict[str, int]) -> int:
        global_cycles = totals.get("global", 0) * global_latency_cycles
        shared_cycles = totals.get("shared", 0) * shared_latency_cycles
        return global_cycles + shared_cycles

    naive_totals = naive_sm.get_total_counters()
    tiled_totals = tiled_sm.get_total_counters()
    naive_cycles = estimate_memory_cycles(naive_totals)
    tiled_cycles = estimate_memory_cycles(tiled_totals)
    cycle_delta = max(naive_cycles - tiled_cycles, 0)
    cycle_ratio = (
        naive_cycles / tiled_cycles if tiled_cycles > 0 else float("inf")
    )

    table_data = [
        [
            "Naive",
            format_int(naive_totals.get("ops", 0)),
            format_int(naive_totals.get("global", 0)),
            format_int(naive_totals.get("shared", 0)),
            f"≈ {format_int(naive_cycles)}",
        ],
        [
            "Tiled",
            format_int(tiled_totals.get("ops", 0)),
            format_int(tiled_totals.get("global", 0)),
            format_int(tiled_totals.get("shared", 0)),
            f"≈ {format_int(tiled_cycles)}",
        ],
    ]
    headers = [
        "Kernel",
        "FLOPs",
        "Global Accesses",
        "Shared Accesses",
        "Est. Memory Cycles",
    ]
    table = Table(
        table_data,
        col_labels=[Text(head, font_size=28) for head in headers],
        include_outer_lines=True,
        element_to_mobject=lambda value: Text(str(value), font_size=26),
    )
    table.scale(0.8)
    table.to_edge(UP, buff=1.0)

    ratio_display = "∞" if tiled_cycles == 0 else f"{cycle_ratio:,.1f}"
    ratio_text = Text(
        f"Naive memory wait ≈ {ratio_display}× tiled "
        f"(Δ ≈ {format_int(cycle_delta)} cycles)",
        font_size=30,
    )
    ratio_text.next_to(table, DOWN, buff=0.6)

    latency_text = Text(
        f"Assuming shared ≈ {shared_latency_cycles} cycles and HBM ≈ "
        f"{global_latency_cycles} cycles of latency on H100.",
        font_size=24,
    )
    latency_text.next_to(ratio_text, DOWN, buff=0.35)

    comparison_voice = (
        "Using the counters we just gathered, the naive kernel issues "
        f"{naive_totals.get('global', 0)} global memory touches, "
        "versus "
        f"{tiled_totals.get('global', 0)} for the tiled version. "
        "With global accesses costing roughly "
        f"{global_latency_cycles} cycles compared to about "
        f"{shared_latency_cycles} for shared memory, "
        "the naive variant spends much longer waiting on HBM."
    )
    with scene.voiceover(text=comparison_voice):
        scene.play(FadeIn(table, lag_ratio=0.1))
        scene.play(FadeIn(ratio_text), FadeIn(latency_text))

    scene.wait(1.0)
