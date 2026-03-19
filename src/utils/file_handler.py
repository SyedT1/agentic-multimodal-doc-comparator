"""
File handling utilities for document upload and validation.
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple
import config


def validate_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate uploaded file.

    Args:
        file_path: Path to the file to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    file_path_obj = Path(file_path)

    # Check if file exists
    if not file_path_obj.exists():
        return False, "File does not exist"

    # Check file extension
    if file_path_obj.suffix.lower() not in config.ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"

    # Check file size
    file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
    if file_size_mb > config.MAX_FILE_SIZE_MB:
        return False, f"File too large ({file_size_mb:.1f}MB). Max: {config.MAX_FILE_SIZE_MB}MB"

    return True, ""


def save_uploaded_file(uploaded_file, destination_dir: Path = None) -> str:
    """
    Save an uploaded Streamlit file to disk.

    Args:
        uploaded_file: Streamlit UploadedFile object
        destination_dir: Directory to save the file (default: config.UPLOAD_DIR)

    Returns:
        Path to saved file as string
    """
    if destination_dir is None:
        destination_dir = config.UPLOAD_DIR

    # Ensure destination directory exists
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Create file path
    file_path = destination_dir / uploaded_file.name

    # Write file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(file_path)


def cleanup_file(file_path: str) -> bool:
    """
    Delete a file from disk.

    Args:
        file_path: Path to file to delete

    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            file_path_obj.unlink()
            return True
        return False
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")
        return False


def cleanup_directory(dir_path: Path, keep_dir: bool = True) -> bool:
    """
    Clean up all files in a directory.

    Args:
        dir_path: Directory to clean
        keep_dir: If True, keep the directory but remove contents

    Returns:
        True if successful, False otherwise
    """
    try:
        if dir_path.exists():
            if keep_dir:
                # Remove all files but keep directory
                for item in dir_path.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            else:
                # Remove directory entirely
                shutil.rmtree(dir_path)
        return True
    except Exception as e:
        print(f"Error cleaning directory {dir_path}: {e}")
        return False


def get_file_type(file_path: str) -> str:
    """
    Get the file type from file extension.

    Args:
        file_path: Path to file

    Returns:
        File type as string ('pdf' or 'docx')
    """
    extension = Path(file_path).suffix.lower()
    if extension == ".pdf":
        return "pdf"
    elif extension in [".docx", ".doc"]:
        return "docx"
    else:
        return "unknown"
