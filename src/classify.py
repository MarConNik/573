import argparse


def classify(records, model=None):
    for record in records:
        print(record.strip('\n'))


if __name__ == '__main__':
    # Parser arguments:
    parser = argparse.ArgumentParser(description='classify records in a test set')
    parser.add_argument('--model-file', type=argparse.FileType('rb'),
                        help='path to saved model to use in classification')
    parser.add_argument('--test-data', type=argparse.FileType('r'),
                        help='path to testing data file')
    parser.add_argument('--output-file', type=argparse.FileType('w'),
                        help='path to output file')

    args = parser.parse_args()
    classify(args.test_data)
