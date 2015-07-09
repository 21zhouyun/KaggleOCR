import numpy as np
import pandas as pd

def get_misclassified_index(pred, train_y):
    return pred != train_y

def get_classified_index(pred, train_y):
    return pred == train_y

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": range(1,len(preds)+1), "Label": preds}).to_csv(fname, index=False, header=True)

def write_label_pattern_pairs(labels, patterns, outf):
    o = open(outf, "w")

    o.write("label," + ",".join("pixel%d" % x for x in range(0, 28 * 28)) + "\n")
    for i in range(patterns.shape[0]):
        o.write(str(labels[i]) + "," + ",".join(str(pix) for pix in patterns[i])+"\n")
    o.close()