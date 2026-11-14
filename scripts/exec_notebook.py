import os
import sys
import types

# 1. Provide safe dummy pyarrow to avoid Windows WDAC policy block
dummy_pa = types.ModuleType('pyarrow')
class _DummyType: pass
dummy_pa.__version__ = '15.0.0'
dummy_pa.Table = _DummyType
dummy_pa.RecordBatch = _DummyType
dummy_pa.Array = _DummyType
dummy_pa.ChunkedArray = _DummyType
dummy_pa.compute = types.ModuleType('pyarrow.compute')
sys.modules['pyarrow'] = dummy_pa
sys.modules['pyarrow.compute'] = dummy_pa.compute

# 2. Fix matplotlib config dir to a writable temp path
temp_mpl = os.path.join(os.environ.get('TEMP', os.getcwd()), 'mpl_cache')
os.makedirs(temp_mpl, exist_ok=True)
os.environ['MPLCONFIGDIR'] = temp_mpl

import io
import json
import base64
import contextlib
import traceback
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

def execute_notebook(nb_path, output_path=None, cell_indices=None):
    if output_path is None:
        output_path = nb_path
        
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    global_env = {
        "__name__": "__main__",
        "__file__": os.path.abspath(nb_path),
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    
    # Pre-populate device and common paths if needed
    init_code = """
import os, sys
sys.path.insert(0, os.getcwd())
import glob, time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import build_model
from src.explainability import GradCAM
"""
    exec(init_code, global_env)
    
    cells = nb.get("cells", [])
    print(f"[*] Executing notebook: {nb_path} ({len(cells)} cells)")
    
    execution_count = 1
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
            
        if cell_indices is not None and i not in cell_indices:
            print(f"  [-] Skipping cell {i} (not in specified indices)")
            continue
            
        code = "".join(cell.get("source", ""))
        # Filter out bash commands like !ls, !pip or %matplotlib while preserving indentation
        clean_lines = []
        for line in code.splitlines(True):
            sline = line.strip()
            if sline.startswith("!") or sline.startswith("%"):
                indent = len(line) - len(line.lstrip())
                clean_lines.append(" " * indent + "pass\n")
            else:
                clean_lines.append(line)
        clean_code = "".join(clean_lines)
        
        if not clean_code.strip():
            continue
            
        print(f"  [+] Executing Cell {i}...")
        cell_outputs = []
        
        def custom_show(*args, **kwargs):
            for fignum in plt.get_fignums():
                fig = plt.figure(fignum)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("utf-8")
                cell_outputs.append({
                    "output_type": "display_data",
                    "data": {
                        "image/png": b64,
                        "text/plain": ["<Figure size ... with ... Axes>"]
                    },
                    "metadata": {}
                })
            plt.close("all")
            
        global_env["plt"].show = custom_show
        
        f_stdout = io.StringIO()
        f_stderr = io.StringIO()
        
        t0 = time.time()
        with contextlib.redirect_stdout(f_stdout), contextlib.redirect_stderr(f_stderr):
            try:
                exec(clean_code, global_env)
            except Exception as e:
                f_stderr.write(traceback.format_exc())
                print(f"    [!] Error in cell {i}: {e}")
                
        # Flush any remaining plots if show wasn't called
        custom_show()
        
        out_text = f_stdout.getvalue()
        err_text = f_stderr.getvalue()
        
        if out_text:
            cell_outputs.insert(0, {
                "output_type": "stream",
                "name": "stdout",
                "text": out_text.splitlines(keepends=True)
            })
        if err_text:
            cell_outputs.append({
                "output_type": "stream",
                "name": "stderr",
                "text": err_text.splitlines(keepends=True)
            })
            
        cell["outputs"] = cell_outputs
        cell["execution_count"] = execution_count
        execution_count += 1
        print(f"    Done in {time.time() - t0:.2f}s | Outputs: {len(cell_outputs)}")
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print(f"[✓] Finished and saved notebook to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        execute_notebook(sys.argv[1])
