import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_architecture():
    # Setup figure
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Modern Color Palette
    BG_COLOR = "#F4F6F9"
    BOX_BORDER = "#4A5568"
    TEXT_MAIN = "#2D3748"
    
    PALETTE = {
        "input": {"bg": "#EBF8FF", "border": "#3182CE", "text": "#2B6CB0"},
        "llm": {"bg": "#FAF5FF", "border": "#805AD5", "text": "#553C9A"},
        "ts": {"bg": "#E6FFFA", "border": "#319795", "text": "#234E52"},
        "output": {"bg": "#F0FFF4", "border": "#38A169", "text": "#276749"},
        "arrow": "#718096"
    }

    # Helper function to draw rounded boxes
    def draw_box(x, y, w, h, config, title, items):
        # Background block
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=1.5",
            facecolor=config["bg"],
            edgecolor=config["border"],
            linewidth=2.0
        )
        ax.add_patch(rect)
        
        # Header text
        ax.text(
            x + w/2, y + h - 2.5, title,
            fontsize=12, fontweight='bold', color=config["text"],
            ha='center', va='center'
        )
        
        # Divider line
        ax.plot([x + 1, x + w - 1], [y + h - 5, y + h - 5], color=config["border"], linewidth=0.8, alpha=0.5)
        
        # Content lines
        for idx, item in enumerate(items):
            ax.text(
                x + 2, y + h - 9 - (idx * 3.5), item,
                fontsize=9.5, color=TEXT_MAIN,
                ha='left', va='center'
            )

    # 1. Draw Input Box (Left)
    draw_box(
        x=5, y=25, w=22, h=50,
        config=PALETTE["input"],
        title="1. Multi-Center EHR Input",
        items=[
            "• EHR Data Sources:",
            "  - eICU Database",
            "  - MIMIC-IV Database",
            "• Patient Trajectory (Days 1-3):",
            "  - Demographic (Age, Gender)",
            "  - Mean Arterial Pressure (MAP)",
            "  - Serum Creatinine (SCr)",
            "  - Vanco/Zosyn Administrations"
        ]
    )

    # 2. Draw Gemma-4 12B Sentinel Box (Center Top)
    draw_box(
        x=38, y=52, w=24, h=35,
        config=PALETTE["llm"],
        title="2. Sentinel LLM (Gemma-4 12B)",
        items=[
            "• SFT Fine-Tuned Clinical Sentinel",
            "• Parses incoming clinical features",
            "• Formulates structured prompt",
            "• Generates TOOL_CALL payload",
            "  with 3-day SCr history"
        ]
    )

    # 3. Draw TimesFM 2.5 Forecaster Box (Center Bottom)
    draw_box(
        x=38, y=13, w=24, h=30,
        config=PALETTE["ts"],
        title="3. TimesFM 2.5 Forecaster",
        items=[
            "• Receives 3-day laboratory input",
            "• Projects time-series dynamics",
            "• Forecasts Day 4 & Day 5 SCr",
            "  (No MAP forecast, only SCr)"
        ]
    )

    # 4. Draw Output Box (Right)
    draw_box(
        x=73, y=25, w=22, h=50,
        config=PALETTE["output"],
        title="4. Risk & Rationale Output",
        items=[
            "• Clinical Synthesis Note:",
            "  - Summary of risk factors",
            "  - Predicted SCr trajectory",
            "• KDIGO Guideline Reasoning",
            "• Final AKI Staging Label:",
            "  - [AKI_STAGE_1+] or",
            "  - [NORMAL]"
        ]
    )

    # --- Draw Arrows & Connections ---
    
    # Arrow 1: Input to LLM
    ax.annotate(
        "", xy=(36, 68), xytext=(28, 55),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["arrow"], lw=2.5, mutation_scale=15)
    )
    ax.text(29, 65, "DP Parameter\nCohort", fontsize=9, color=TEXT_MAIN, ha='center', va='center')

    # Arrow 2: LLM Tool Call to TimesFM (Downward)
    ax.annotate(
        "", xy=(46, 45), xytext=(46, 51),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["ts"]["border"], lw=2, mutation_scale=15, linestyle='--')
    )
    ax.text(41.5, 48, "[TOOL_CALL]\nSCr History", fontsize=8.5, color=PALETTE["ts"]["text"], ha='center', va='center')

    # Arrow 3: TimesFM Return to LLM (Upward)
    ax.annotate(
        "", xy=(54, 51), xytext=(54, 45),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["llm"]["border"], lw=2, mutation_scale=15)
    )
    ax.text(59, 48, "[FORECAST]\nDay 4-5 SCr", fontsize=8.5, color=PALETTE["llm"]["text"], ha='center', va='center')

    # Arrow 4: LLM to Output
    ax.annotate(
        "", xy=(72, 50), xytext=(63.5, 68),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["arrow"], lw=2.5, mutation_scale=15)
    )
    ax.text(70, 61, "Generate\nReport", fontsize=9, color=TEXT_MAIN, ha='center', va='center')

    # Add a title at the top of the canvas
    plt.title("Agentic-TimesFM-AKI: System Architecture", fontsize=16, fontweight='bold', pad=20, color=TEXT_MAIN)

    # Make parent directories if they don't exist
    out_path = Path("Article/figures/system_architecture.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated System Architecture diagram at: {out_path.resolve()}")

if __name__ == "__main__":
    draw_architecture()
