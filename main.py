import argparse
import pandas as pd

def convert_elpv_labels_csv(inFilePath: str, outFilePath: str):
    """
    Read the elpv labels.csv file. Then converts and saves it as a proper csv file, since the original 
    labels.csv is improperly formatted
    
    inFilePath: 
        filepath of the input labels.csv file, which is in the format provided by the original elpv dataset
    
    outFilePath:
        filepath to write the output csv
    """

    f = open(inFilePath)
    lines = [line.split() for line in f.readlines()]

    df = pd.DataFrame(lines, columns=['path', 'probability', 'type'])
    df['type'] = df['type'].apply(lambda x: 0 if x == "mono" else 1)
    
    # df1 = df[['path', 'probability']]
    df.to_csv(outFilePath, index=False)


def main():
    print("Hello World")

if __name__ == "__main__":
    main()