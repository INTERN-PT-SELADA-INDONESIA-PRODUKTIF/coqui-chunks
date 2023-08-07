from helper_stt import upload_file
import argparse

def main():
    parser = argparse.ArgumentParser(description='Upload  audio file.')
    parser.add_argument('PATH', type=str, help='Path audio file')

    args = parser.parse_args()
    PATH = args.PATH

    upload_file(PATH)

if __name__ == "__main__":
    main()
