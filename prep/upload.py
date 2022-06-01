import os, traceback
from pathlib import Path

import gcapi


def upload_data(input: Path, slug: str, api: str):
    client = gcapi.Client(token=api)

    files = {}
    # Loop through files in the specified directory and add their names to the list
    for root, directories, filenames in os.walk(input):
        for direc in directories:
            temp = []
            for file in os.listdir(os.path.join(root, direc)):
                temp.append(os.path.join(root, direc, file))
            files[direc] = [Path(t) for t in temp]
    print("Found", len(files.keys()), "cases for upload")

    sessions = []
    for i, key in enumerate(files.keys(), 1):
        try:
            temp_session = client.upload_cases(files=files[key], reader_study=slug)
            sessions.append(temp_session)
            print("Uploaded session", i, "of", len(files.keys()))
        except Exception:
            traceback.print_exc()