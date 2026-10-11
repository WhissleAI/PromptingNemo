"""Print "<name> <python>" for every system in steer/paper/baselines_ext.py SYSTEMS (for run_speed.sh).
Reads the registry statically (ast) so this works from any env; falls back to importing it."""
import ast
import sys
from pathlib import Path

SRC = Path(__file__).with_name("baselines_ext.py")


def py_of(env):
    env = env or ""
    return env if env.endswith("python") else f"/home/karan/anaconda3/envs/{env}/bin/python"


def main():
    out = {}
    try:
        tree = ast.parse(SRC.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == "SYSTEMS" for t in tgts) and isinstance(node.value, ast.Dict):
                    for k, v in zip(node.value.keys, node.value.values):
                        if isinstance(k, ast.Constant) and isinstance(v, ast.Dict):
                            env = None
                            for kk, vv in zip(v.keys, v.values):
                                if isinstance(kk, ast.Constant) and kk.value == "env" and isinstance(vv, ast.Constant):
                                    env = vv.value
                            out[k.value] = env
    except Exception as e:  # noqa: BLE001
        print("# ast failed:", e, file=sys.stderr)
    if not out:
        try:
            sys.path.insert(0, str(SRC.parents[2]))
            from steer.paper import baselines_ext as be
            out = {n: e.get("env") for n, e in be.SYSTEMS.items()}
        except Exception as e:  # noqa: BLE001
            print("# import failed:", e, file=sys.stderr)
    for n, env in out.items():
        print(n, py_of(env))


if __name__ == "__main__":
    main()
