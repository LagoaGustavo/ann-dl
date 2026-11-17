"""Roda os scripts da entrega 2 em ordem e grava a saída completa em saida.txt (UTF-8).

    python docs/projects/classification/code/roda_tudo.py

Todo número citado no relatório sai deste arquivo.
"""
import platform
import subprocess
import sys
from pathlib import Path

import numpy
import pandas
import sklearn
import torch

BASE = Path(__file__).resolve().parent
SCRIPTS = ["s0_partida.py", "s1_baselines.py", "s23_rede_treino.py",
           "s4_regularizacao.py", "s56_teste_erros.py"]

with open(BASE / "saida.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write(f"### saída de todos os scripts | Python {platform.python_version()} | "
            f"torch {torch.__version__} | scikit-learn {sklearn.__version__} | "
            f"numpy {numpy.__version__} | pandas {pandas.__version__}\n")
    for s in SCRIPTS:
        f.write(f"\n######## {s}\n")
        f.flush()
        r = subprocess.run([sys.executable, str(BASE / s)], cwd=BASE, capture_output=True,
                           text=True, encoding="utf-8", env={**__import__("os").environ,
                                                             "PYTHONUTF8": "1"})
        f.write(r.stdout)
        if r.returncode != 0:
            f.write(r.stderr)
        f.write(f"\n[{s} {'OK' if r.returncode == 0 else 'FALHOU'}]\n")
        print(f"{s}: {'OK' if r.returncode == 0 else 'FALHOU'}", flush=True)
        if r.returncode != 0:
            sys.exit(1)
