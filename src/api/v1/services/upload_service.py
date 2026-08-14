import shutil
from pathlib import Path
from fastapi import HTTPException, status
from src.ingestion.ingestion import run_ingestion

DATA_FOLDER = Path("data")


def call_file_ingestion(file):
    """
    Upload file received for ingestion to the data folder.
    Checks if a file exists in a data folder and then call the file ingestion process.
    """

    if not DATA_FOLDER.exists():
        DATA_FOLDER.mkdir(parents=True, exist_ok=True)
        print("Data folder created successfully.")

    file_path = DATA_FOLDER / file.filename
    print(f"File Path = {file_path}")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            print("File uploaded successfully.")
            print("File ingestion started...")
            run_ingestion(str(file_path))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Service: An error occurred during file upload/ingestion process: {str(e)}",
        )
