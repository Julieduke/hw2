import sys
import math

def compute_entropy(labels):
    total = len(labels)
    counts = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy
    

def compute_error(labels):
    total = len(labels)
    counts = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    majority = max(counts.values())
    return 1 - majority / total

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    #read file
    labels = []
    with open(input_file, "r") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            row = line.strip().split("\t")
            labels.append(row[-1]) 

    # entropy and error
    entropy = compute_entropy(labels)
    error = compute_error(labels)

    with open(output_file, "w") as f:
        f.write(f"entropy: {entropy}\n")
        f.write(f"error: {error}\n")
