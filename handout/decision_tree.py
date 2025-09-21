import argparse
import math

class Node:
    """
    Binary decision tree node for 0/1 features and labels.
    Stores training-label counts for pretty printing.
    """
    def __init__(self, depth=0):
        self.left = None           # child when feature == '0'
        self.right = None          # child when feature == '1'
        self.attr = None           # attribute used for split at this node
        self.vote = None           # majority label at this node ('0' or '1')
        self.depth = depth         # root = 0
        self.n0 = 0                # # of '0' labels among training rows at this node
        self.n1 = 0                # # of '1' labels among training rows at this node

    def is_leaf(self):
        return self.attr is None

def read_tsv(path):
    with open(path, "r") as f:
        header = f.readline().strip().split("\t")
        rows = []
        for line in f:
            s = line.rstrip("\n")
            if not s:
                continue
            parts = s.split("\t")
            rows.append(dict(zip(header, parts)))
    return header, rows

def count_labels(labels):
    c0 = 0
    for y in labels:
        if y == "0":
            c0 += 1
    c1 = len(labels) - c0
    return c0, c1

def majority_label(labels):
    c0, c1 = count_labels(labels)
    return "1" if c1 >= c0 else "0"

# entropy and information

def entropy_of_labels(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    c0, c1 = count_labels(labels)
    H = 0.0
    if c0:
        p = c0 / n
        H -= p * math.log2(p)
    if c1:
        p = c1 / n
        H -= p * math.log2(p)
    return H

def information_gain(rows, attr, label_name):
    """I(Y;X)=H(Y) - P(X=0)H(Y|X=0) - P(X=1)H(Y|X=1). Weighted by empirical proportions."""
    n = len(rows)
    if n == 0:
        return 0.0
    labels_parent = [r[label_name] for r in rows]
    H_parent = entropy_of_labels(labels_parent)

    left = [r for r in rows if r[attr] == "0"]
    right = [r for r in rows if r[attr] == "1"]

    wH = 0.0
    if left:
        w = len(left) / n
        wH += w * entropy_of_labels([r[label_name] for r in left])
    if right:
        w = len(right) / n
        wH += w * entropy_of_labels([r[label_name] for r in right])

    return H_parent - wH

# Train model

def build_tree(rows, feature_names, label_name, max_depth, depth=0):
    node = Node(depth=depth)
    labels = [r[label_name] for r in rows]

    node.n0, node.n1 = count_labels(labels)
    node.vote = majority_label(labels) if labels else "0"

    if depth >= max_depth:
        return node
    if node.n0 == 0 or node.n1 == 0:
        return node
    if not feature_names:
        return node

    best_attr = None
    best_gain = -1.0
    for a in feature_names: 
        ig = information_gain(rows, a, label_name)
        if ig > best_gain + 1e-12:
            best_gain = ig
            best_attr = a

    if best_attr is None or best_gain <= 0.0:
        return node

    left_rows = [r for r in rows if r[best_attr] == "0"]
    right_rows = [r for r in rows if r[best_attr] == "1"]
    if len(left_rows) == 0 or len(right_rows) == 0:
        return node

    node.attr = best_attr
    remaining = [a for a in feature_names if a != best_attr]

    node.left = build_tree(left_rows, remaining, label_name, max_depth, depth + 1)
    node.right = build_tree(right_rows, remaining, label_name, max_depth, depth + 1)
    return node

def predict_one(node, row):
    cur = node
    while not cur.is_leaf():
        if row[cur.attr] == "0":
            cur = cur.left
        else:
            cur = cur.right
    return cur.vote

def predict_all(node, rows):
    return [predict_one(node, r) for r in rows]

def error_rate(preds, rows, label_name):
    total = len(rows)
    wrong = 0
    for p, r in zip(preds, rows):
        if p != r[label_name]:
            wrong += 1
    return wrong / total if total > 0 else 0.0

# print

def print_tree(root, file):
    file.write(f"[{root.n0} 0/{root.n1} 1]\n")

    def recurse(n):
        if n.is_leaf():
            return
        # Left=0
        file.write(f"{'| ' * n.depth}{n.attr} = 0: [{n.left.n0} 0/{n.left.n1} 1]\n")
        recurse(n.left)
        # Right=1
        file.write(f"{'| ' * n.depth}{n.attr} = 1: [{n.right.n0} 0/{n.right.n1} 1]\n")
        recurse(n.right)

    recurse(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help="path to training input .tsv file")
    parser.add_argument("test_input", type=str, help="path to the test input .tsv file")
    parser.add_argument("max_depth", type=int, help="maximum depth to which the tree should be built")
    parser.add_argument("train_out", type=str, help="path to output .txt for predictions on training data")
    parser.add_argument("test_out", type=str, help="path to output .txt for predictions on test data")
    parser.add_argument("metrics_out", type=str, help="path to output .txt for train/test error")
    parser.add_argument("print_out", type=str, help="path to output .txt for printed tree")
    args = parser.parse_args()

    header, train_rows = read_tsv(args.train_input)
    _, test_rows = read_tsv(args.test_input)

    label_name = header[-1]
    feature_names = header[:-1] 

    # build tree
    tree = build_tree(train_rows, feature_names, label_name, max_depth=args.max_depth, depth=0)

    # predictions
    train_preds = predict_all(tree, train_rows)
    test_preds = predict_all(tree, test_rows)

    with open(args.train_out, "w") as f:
        f.write("\n".join(train_preds))
        f.write("\n")
    with open(args.test_out, "w") as f:
        f.write("\n".join(test_preds))
        f.write("\n")

    # metrics
    train_err = error_rate(train_preds, train_rows, label_name)
    test_err = error_rate(test_preds, test_rows, label_name)
    with open(args.metrics_out, "w") as f:
        f.write(f"error(train): {train_err}\n")
        f.write(f"error(test): {test_err}\n")

    # printed tree
    with open(args.print_out, "w") as f:
        print_tree(tree, f)
