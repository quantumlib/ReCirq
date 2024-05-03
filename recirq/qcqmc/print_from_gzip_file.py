import argparse

from qc_afqmc.newtilities import load_data_from_gzip_file


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("file", type=str, default=None, help="The .gzip file to load and print.")
    parser.add_argument(
        "-d",
        "--data",
        dest="print_data_too",
        default=False,
        action='store_true',
        help="Prints the data as well as the params.",
    )

    args = parser.parse_args()

    if args.file is None:
        raise ValueError("Need to provide a .gzip file to open")

    results = load_data_from_gzip_file(args.file, do_print=False)

    if args.print_data_too:
        print(results)
    else:
        print("Printing params only, use -d flag to print data too.\n")
        print("params=")
        print(results.params)


if __name__ == "__main__":
    main()
