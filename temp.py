import time
import argparse

parser = argparse.ArgumentParser(description="ZSL")
parser.add_argument("--p", default=0, type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    time.sleep(5)
    if args.p <= 0:
        raise Exception
    print(args.p)
