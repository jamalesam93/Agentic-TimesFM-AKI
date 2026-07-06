import matplotlib.pyplot as plt
import os
import numpy as np

# Set academic styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
})

def main():
    # DP Epsilon Budget Allocation
    labels = [
        'Demographics & Comorbidities\n(15 Queries)', 
        'SCr Labs & Multiple Regression\n(7 Queries)', 
        'Drug Exposures & Outcomes\n(7 Queries)'
    ]
    sizes = [40, 35, 25]  # Percentages
    colors = ['#4285F4', '#EA4335', '#FBBC05']
    explode = (0.05, 0.05, 0.05)  # slight separation

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create donut chart
    wedges, texts, autotexts = ax.pie(
        sizes, 
        explode=explode, 
        labels=labels, 
        colors=colors, 
        autopct='%1.1f%%',
        shadow=False, 
        startangle=90,
        pctdistance=0.75,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    # Draw white circle in the middle
    centre_circle = plt.Circle((0,0),0.50,fc='white')
    fig.gca().add_artist(centre_circle)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')  
    plt.title('Differential Privacy Budget (ε) Allocation', fontsize=16, fontweight='bold', pad=20)
    
    # Add text in the center
    plt.text(0, 0, 'Total ε\n100%', ha='center', va='center', fontsize=14, fontweight='bold')

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(root_dir, "plots", "real_world")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "privacy_budget_allocation.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Privacy Budget Chart to {out_path}")

if __name__ == "__main__":
    main()
