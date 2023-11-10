import csv

import pandas as pd


def select_first_20000_rows():
    selected_data = []
    count = 0

    csv_file = 'WholeVdata2.csv'
    output_file = 'selected_data20000.csv'

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            selected_data.append(row)
            count += 1
            if count == 20000:
                break

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(selected_data)

def split():
    original_data = pd.read_csv("interactive_formation_improve.csv")

    total_samples = len(original_data)
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    train_samples = int(total_samples * train_ratio)
    val_samples = int(total_samples * val_ratio)

    train_data = original_data[:train_samples]
    val_data = original_data[train_samples:(train_samples + val_samples)]
    test_data = original_data[(train_samples + val_samples):]

    train_data.to_csv("train_dataset.csv", index=False)
    val_data.to_csv("val_dataset.csv", index=False)
    test_data.to_csv("test_dataset.csv", index=False)



if __name__ == '__main__':
    # select_first_20000_rows()

    split()