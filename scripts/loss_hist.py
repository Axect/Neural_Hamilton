import polars as pl
import matplotlib.pyplot as plt
import scienceplots
import os
import beaupy
from rich.console import Console

console = Console()

def choose_projects_to_plot():
    project_names = []
    
    # List all folders in figs
    for d in os.listdir("figs"):
        if os.path.isdir(os.path.join("figs", d)):
            project_names.append(os.path.basename(d))

    console.print("Choose projects to draw histogram")
    selected_projects = beaupy.select_multiple(
        project_names
    )

    selected_projects = [os.path.join("figs", d) for d in selected_projects] #pyright:ignore

    return selected_projects


# Example usage
if __name__ == "__main__":
    selected = choose_projects_to_plot()
    print("Selected projects:", selected)
