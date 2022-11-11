import os
import logging

from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

def do_plot(fn):
    headers = ['date','values']
    df = pd.read_csv(fn, names=headers)
    df['date'] = df['date'].map(lambda x: datetime.strptime(str(x), '%Y-%m-%dT%H:%M:%S'))
    print(df.head())

    plt.figure(figsize=(12, 4))
    x = df['date']
    y = df['values']
    plt.scatter(x, y, edgecolor="k", color="lightblue")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='rule file')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not args.file or not os.path.exists(args.file):
        logging.error('Error: Missing rule file')
    else:
        do_plot(args.file)
