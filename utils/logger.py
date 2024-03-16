from termcolor import colored

def _log(prefix, *args):
    prefix = colored(f"[{prefix}]", "light_magenta")
    out = " ".join([str(a).strip(" ") for a in args])
    if out[0] != "\n":
        out = f"{prefix} {out}"
    out = out.replace("\n", f"\n{prefix} ")
    print(out)                