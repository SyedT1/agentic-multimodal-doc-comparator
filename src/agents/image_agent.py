"""
Image agent for extracting and embedding images from documents.
Uses CLIP for vision embeddings.
"""
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path
from PIL import Image
import io

from agents.base_agent import BaseAgent
from models.document import ImageExtraction, RawDocument
import config

# Try to import CLIP dependencies
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠ CLIP not available. Install: pip install transformers torch")

# Try to import PDF image extraction libraries
try:
    import fitz  # PyMuPDF for image extraction
    PYMUPDF_AVAILABLE = True
except (ImportError, OSError):
    PYMUPDF_AVAILABLE = False
    print("⚠ PyMuPDF not available for image extraction")


class ImageAgent(BaseAgent):
    """Agent responsible for image extraction and embedding generation."""

    def __init__(self, config_dict: Dict[str, Any] = None):
        super().__init__(config_dict)

        if not CLIP_AVAILABLE:
            raise ImportError(
                "CLIP model not available. Install: pip install transformers torch"
            )

        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = config_dict.get("clip_model", "openai/clip-vit-base-patch32") if config_dict else "openai/clip-vit-base-patch32"

        print(f"Loading CLIP model: {model_name} on {self.device}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def get_agent_name(self) -> str:
        return "ImageAgent"

    async def process(self, raw_document: RawDocument) -> Tuple[List["ImageExtraction"], np.ndarray]:
        """
        Process raw document and extract images with embeddings.

        Args:
            raw_document: Raw document with file path information

        Returns:
            Tuple of (list of ImageExtraction objects, numpy array of embeddings)
        """
        # Extract images from document
        images = self.extract_images(raw_document)

        # Generate embeddings
        if images:
            embeddings = self.generate_embeddings(images)
        else:
            embeddings = np.array([])

        return images, embeddings

    def extract_images(self, raw_document: RawDocument) -> List["ImageExtraction"]:
        """
        Extract images from document.

        Args:
            raw_document: Raw document

        Returns:
            List of ImageExtraction objects
        """
        images = []

        # Get file path from raw_document metadata
        # Since we don't store it, we'll need to extract from pages metadata
        file_type = raw_document.file_type

        if file_type == "pdf":
            images = self._extract_images_from_pdf(raw_document)
        elif file_type == "docx":
            images = self._extract_images_from_docx(raw_document)

        return images

    def _extract_images_from_pdf(self, raw_document: RawDocument) -> List["ImageExtraction"]:
        """
        Extract images from PDF using PyMuPDF.

        Args:
            raw_document: Raw document

        Returns:
            List of ImageExtraction objects
        """
        if not PYMUPDF_AVAILABLE:
            print("⚠ PyMuPDF not available, skipping image extraction")
            return []

        images = []

        # We need the file path - for now, we'll need to modify the flow
        # For this implementation, we'll store it in metadata
        file_path = raw_document.metadata.get("file_path") if hasattr(raw_document, "metadata") else None

        if not file_path:
            print("⚠ File path not available in raw_document metadata")
            return []

        try:
            with fitz.open(file_path) as pdf_doc:
                for page_num, page in enumerate(pdf_doc, start=1):
                    # Get images on this page
                    image_list = page.get_images(full=True)

                    for img_idx, img_info in enumerate(image_list):
                        xref = img_info[0]

                        # Extract image
                        base_image = pdf_doc.extract_image(xref)
                        image_bytes = base_image["image"]

                        # Convert to PIL Image
                        pil_image = Image.open(io.BytesIO(image_bytes))

                        # Skip very small images (likely icons/decorations)
                        if pil_image.width < 50 or pil_image.height < 50:
                            continue

                        # Convert to RGB if needed
                        if pil_image.mode != "RGB":
                            pil_image = pil_image.convert("RGB")

                        # Create ImageExtraction object
                        from models.document import ImageExtraction

                        image_extraction = ImageExtraction(
                            image=pil_image,
                            page_number=page_num,
                            image_index=img_idx,
                            width=pil_image.width,
                            height=pil_image.height,
                            format=base_image.get("ext", "unknown")
                        )

                        images.append(image_extraction)

        except Exception as e:
            print(f"Error extracting images from PDF: {e}")

        return images

    def _extract_images_from_docx(self, raw_document: RawDocument) -> List["ImageExtraction"]:
        """
        Extract images from DOCX.

        Args:
            raw_document: Raw document

        Returns:
            List of ImageExtraction objects
        """
        from docx import Document
        from models.document import ImageExtraction

        images = []
        file_path = raw_document.metadata.get("file_path") if hasattr(raw_document, "metadata") else None

        if not file_path:
            print("⚠ File path not available in raw_document metadata")
            return []

        try:
            doc = Document(file_path)

            # Extract images from document relationships
            for rel_idx, rel in enumerate(doc.part.rels.values()):
                if "image" in rel.target_ref:
                    try:
                        image_bytes = rel.target_part.blob
                        pil_image = Image.open(io.BytesIO(image_bytes))

                        # Skip very small images
                        if pil_image.width < 50 or pil_image.height < 50:
                            continue

                        # Convert to RGB if needed
                        if pil_image.mode != "RGB":
                            pil_image = pil_image.convert("RGB")

                        image_extraction = ImageExtraction(
                            image=pil_image,
                            page_number=1,  # DOCX treated as single page
                            image_index=rel_idx,
                            width=pil_image.width,
                            height=pil_image.height,
                            format=rel.target_ref.split('.')[-1]
                        )

                        images.append(image_extraction)

                    except Exception as e:
                        print(f"Error processing image {rel_idx}: {e}")
                        continue

        except Exception as e:
            print(f"Error extracting images from DOCX: {e}")

        return images

    def generate_embeddings(self, images: List["ImageExtraction"]) -> np.ndarray:
        """
        Generate CLIP embeddings for images.

        Args:
            images: List of ImageExtraction objects

        Returns:
            Numpy array of embeddings (shape: num_images x embedding_dim)
        """
        if not images:
            return np.array([])

        embeddings_list = []

        with torch.no_grad():
            for img_extraction in images:
                try:
                    # Process image
                    inputs = self.processor(
                        images=img_extraction.image,
                        return_tensors="pt"
                    ).to(self.device)

                    # Get image features (embeddings)
                    image_features = self.model.get_image_features(**inputs)

                    # Normalize embeddings
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    # Convert to numpy
                    embedding = image_features.cpu().numpy().flatten()
                    embeddings_list.append(embedding)

                except Exception as e:
                    print(f"Error generating embedding for image: {e}")
                    # Add zero embedding as placeholder
                    embeddings_list.append(np.zeros(512))  # CLIP base has 512 dims

        if embeddings_list:
            return np.array(embeddings_list)
        else:
            return np.array([])
