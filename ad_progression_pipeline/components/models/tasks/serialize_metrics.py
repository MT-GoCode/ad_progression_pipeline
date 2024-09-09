import base64
from io import BytesIO

import matplotlib.axes


def save_plot_to_base64(ax: matplotlib.axes._axes.Axes) -> str:
    buffer = BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    return img_base64


def serialize_metrics(data: dict, file_path: str) -> None:
    """Takes a dictionary and creates a markdown file."""
    with open(file_path, "w") as f:
        for key, value in data.items():
            # Write header
            f.write(f"## {key}\n\n")

            if isinstance(value, (str, float, int)):
                f.write(f"{value}\n\n")
            elif isinstance(value, matplotlib.axes._axes.Axes):  # noqa: SLF001
                img_base64 = save_plot_to_base64(value)
                f.write(f"![{key}](data:image/png;base64,{img_base64})\n\n")
            else:
                f.write("Unsupported type\n\n")
