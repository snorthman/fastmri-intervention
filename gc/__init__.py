import gcapi
import os, sys, argparse, traceback, pathlib, collections

parser = argparse.ArgumentParser()
parser.add_argument('--api', type=str, required=True)
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--slug', type=str, required=True)


def upload(args):
    client = gcapi.Client(token=args.api)

    files = {}
    # Loop through files in the specified directory and add their names to the list
    for root, directories, filenames in os.walk(args.input):
        for direc in directories:
            temp = []
            for file in os.listdir(os.path.join(root, direc)):
                temp.append(os.path.join(root, direc, file))
            files[direc] = [pathlib.Path(t) for t in temp]
    print("Found", len(files.keys()), "cases for upload")

    sessions = []
    for i, key in enumerate(files.keys(), 1):
        try:
            temp_session = client.upload_cases(files=files[key], reader_study=args.slug)
            sessions.append(temp_session)
            print(files[key])
            print("Uploaded session", i, "of", len(files.keys()))
        except Exception as e:
            print(f"files[key]={files[key]}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    args = parser.parse_args()
    upload(args)

