import subprocess
import math
import os
import matplotlib.pyplot as plt

TRAIN = "heart_train.tsv"
TEST = "heart_test.tsv"
TREE = "decision_tree.py"

def get_num_attributes(tsv_path):
    with open(tsv_path, "r") as f:
        header = f.readline().strip().split("\t")
    # label is last column
    return len(header) - 1

def run_one_depth(d):
    train_out   = f"heart_{d}_train.txt"
    test_out    = f"heart_{d}_test.txt"
    metrics_out = f"heart_{d}_metrics.txt"
    print_out   = f"heart_{d}_print.txt"

    cmd = [
        "python", TREE, TRAIN, TEST, str(d),
        train_out, test_out, metrics_out, print_out
    ]
    subprocess.run(cmd, check=True)

    # parse metrics
    with open(metrics_out, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # expect:
    # error(train): <float>
    # error(test): <float>
    tr = float(lines[0].split(":")[1].strip())
    te = float(lines[1].split(":")[1].strip())
    return tr, te

def main():
    m = get_num_attributes(TRAIN)
    depths = list(range(0, m + 1))
    train_errs = []
    test_errs = []

    for d in depths:
        tr, te = run_one_depth(d)
        train_errs.append(tr)
        test_errs.append(te)

    # Save CSV for your report/LaTeX table if needed
    with open("heart_error_vs_depth.csv", "w") as f:
        f.write("depth,train_error,test_error\n")
        for d, tr, te in zip(depths, train_errs, test_errs):
            f.write(f"{d},{tr},{te}\n")

    # Plot (single figure, both lines)
        # Plot (single figure, both lines, distinct colors and styles)
        # Plot (blue vs red, clear labels)
    plt.figure(figsize=(7,5))
    plt.plot(depths, train_errs, marker="o", color="blue", linestyle="-", linewidth=2, markersize=6, label="Training Error")
    plt.plot(depths, test_errs, marker="s", color="red", linestyle="--", linewidth=2, markersize=6, label="Testing Error")

    plt.xlabel("Max Depth", fontsize=12)
    plt.ylabel("Error", fontsize=12)
    plt.title("Heart Dataset: Error vs. Tree Depth", fontsize=14, pad=12)
    plt.legend(frameon=True, fontsize=11)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig("heart_error_vs_depth.png", dpi=200)
    print("Saved: heart_error_vs_depth.png and heart_error_vs_depth.csv")
if __name__ == "__main__":
    main()