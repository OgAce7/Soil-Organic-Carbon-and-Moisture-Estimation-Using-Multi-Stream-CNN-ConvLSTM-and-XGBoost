"""
apply_log_target.py -- add an optional "train on log(SOC)" mode to the project.

Run once from the project root:
    python scripts/apply_log_target.py

What it changes (each file is backed up first as <file>.bak_log):
  config.py : adds   log_target = True      (set to False to go back to normal training)
  utils.py  : generate_xy() returns log1p(SOC) when log_target is True;
              adds inv_target() which converts values back to g/kg
  train.py  : the train/test scores are computed after converting back to g/kg,
              so R2 / RMSE / MAE stay comparable with earlier runs (loss is still on the log scale)
  pred.py   : same back-conversion before the test scores

Safe to run twice (it skips anything already patched). Your other edits are not touched.
"""
import os
import shutil
import sys

problems = []


def patch(path, edits, append=None):
    if not os.path.exists(path):
        problems.append("{} not found (run from the project root)".format(path))
        return
    s = open(path, encoding="utf-8").read()
    if "log_target" in s or "inv_target" in s:
        print("{}: already patched, skipped".format(path))
        return
    new = s
    for anchor, repl, count in edits:
        n = new.count(anchor)
        if n != count:
            problems.append("{}: expected {} match(es) of {!r} but found {}".format(path, count, anchor[:60], n))
            return
        new = new.replace(anchor, repl)
    if append:
        new = new.rstrip("\n") + "\n" + append
    shutil.copy(path, path + ".bak_log")
    open(path, "w", encoding="utf-8").write(new)
    print("{}: patched (backup: {}.bak_log)".format(path, path))


patch("config.py", [(
    "rand_seed = 314\n",
    "rand_seed = 314\nlog_target = True   # True: train on log1p(SOC); scores are still reported in g/kg\n", 1)])

patch("utils.py", [(
    "    y = np.array(df_samples[cfg.target_var_name])\n",
    "    y = np.array(df_samples[cfg.target_var_name])\n"
    "    if getattr(cfg, 'log_target', False):\n"
    "        y = np.log1p(y)\n", 1)],
    append=(
        "\n\ndef inv_target(v):\n"
        "    \"\"\"convert model-scale values back to g/kg (identity if log_target is False)\"\"\"\n"
        "    v = np.asarray(v, dtype=float).ravel()\n"
        "    if getattr(cfg, 'log_target', False):\n"
        "        return np.maximum(np.expm1(v), 0.0)\n"
        "    return v\n"))

BACK = ("        y_input_list = utils.inv_target(y_input_list)\n"
        "        y_pred_list = utils.inv_target(y_pred_list)\n")
patch("train.py", [(
    "        rmse = np.sqrt(metrics.mean_squared_error(y_input_list, y_pred_list))\n",
    BACK + "        rmse = np.sqrt(metrics.mean_squared_error(y_input_list, y_pred_list))\n", 2)])

patch("pred.py", [(
    "    rmse = np.sqrt(metrics.mean_squared_error(y_true_list, y_pred_list))\n",
    "    y_true_list = utils.inv_target(y_true_list)\n"
    "    y_pred_list = utils.inv_target(y_pred_list)\n"
    "    rmse = np.sqrt(metrics.mean_squared_error(y_true_list, y_pred_list))\n", 1)])

if problems:
    print("\nPROBLEMS (nothing was changed for these files):")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("\nDone. Set log_target = False in config.py to switch back.")