"""Ejecuta en Colab solo las celdas que *muestran* resultados de un notebook y guarda sus salidas.

Las celdas que entrenan o sobrescriben resultados (entrenar, --export-g, --candidates, --select, métricas) se
saltan: sus resultados ya están en Drive. La celda de drive.mount se reemplaza por su versión sin el montaje
(Drive ya tiene que estar montado en la VM de Colab).

    python scripts/run_notebook_views.py notebooks/02_entrenamiento_y_experimentos.ipynb salida.ipynb
"""
import sys

import nbformat
from nbclient import NotebookClient

SKIP = ("entrenar('configs", "src.train", "datetime", "--export-g", "--candidates", "--select",
        "--run \"", "--gallery \"")


def main(src, dst):
    nb = nbformat.read(src, as_version=4)
    client = NotebookClient(nb, timeout=1800, kernel_name="python3", allow_errors=False)
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            if "drive.mount" in cell.source:
                lines = [l for l in cell.source.splitlines() if "google.colab" not in l and "drive.mount" not in l]
                original = cell.source
                setup = nbformat.v4.new_code_cell("\n".join(lines))
                client.execute_cell(setup, i)
                # execute_cell deja la celda ejecutada en nb.cells[i]: devolverle su código original
                nb.cells[i] = setup
                setup.source = original
                if "id" in cell:
                    setup.id = cell.id
                print(f"[{i}] setup sin drive.mount")
                continue
            if any(k in cell.source for k in SKIP):
                print(f"[{i}] saltada: {cell.source.splitlines()[0][:70]}")
                continue
            client.execute_cell(cell, i)
            print(f"[{i}] ejecutada ({len(cell.outputs)} salidas)")
    nbformat.write(nb, dst)
    print("Escrito", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
