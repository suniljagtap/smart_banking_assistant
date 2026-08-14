from fastapi import APIRouter, HTTPException, status, UploadFile, File
from src.api.v1.services.upload_service import call_file_ingestion

router = APIRouter(prefix="/api/v1/admin")


@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_file(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file was uploaded or provided in the request.",
        )

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF allowed.",
        )

    try:
        call_file_ingestion(file=file)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Route: An error occurred while uploading the file: {str(e)}",
        )

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "status": "File upload and ingestion completed successfully.",
    }
