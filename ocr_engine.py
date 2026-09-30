"""
OCR Engine untuk mengekstrak teks dan koordinat spasial dari file PDF KSM.
Mendukung multi-halaman untuk PDF teks digital (PyMuPDF) dan PDF scan/vektor (RapidOCR ONNX).
"""

import os
import logging
from typing import List, Optional
from dataclasses import dataclass, field

import fitz  # PyMuPDF
from PIL import Image

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPID_OCR_AVAILABLE = True
except ImportError:
    RAPID_OCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OCR_Engine")


@dataclass
class DocumentBox:
    """Kotak teks berkoordinat spasial untuk analisis tata letak (layout)."""
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    score: float = 1.0
    page_num: int = 0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class OCRResult:
    """Hasil ekstraksi teks dan struktur spasial dari PDF KSM."""
    raw_text: str
    page_count: int
    source_file: str
    extraction_method: str  # "pymupdf_words" | "rapidocr_onnx"
    boxes: List[DocumentBox] = field(default_factory=list)
    confidence: float = 1.0


class OCREngine:
    """
    Mesin OCR adaptif untuk KSM multi-halaman.
    - Cepat & presisi menggunakan PyMuPDF untuk digital PDF
    - Fallback otomatis ke RapidOCR (ONNX deep learning) untuk PDF scan/vektor
    """

    def __init__(self, tesseract_path: Optional[str] = None):
        self._rapid_engine = None

    def _get_rapid_engine(self):
        if self._rapid_engine is None:
            if not RAPID_OCR_AVAILABLE:
                raise RuntimeError("rapidocr-onnxruntime tidak terpasang!")
            self._rapid_engine = RapidOCR()
        return self._rapid_engine

    def extract_text(
        self,
        pdf_path: str,
        method: str = "auto",
        ocr_fallback: bool = True
    ) -> OCRResult:
        """
        Ekstraksi teks dan koordinat spasial dari seluruh halaman PDF.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File tidak ditemukan: {pdf_path}")

        logger.info(f"Memproses file: {os.path.basename(pdf_path)}")
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        if page_count == 0:
            doc.close()
            return OCRResult(raw_text="", page_count=0, source_file=pdf_path, extraction_method="empty")

        # Cek ketersediaan kata digital di seluruh halaman
        total_digital_words = sum(len(doc[p].get_text("words")) for p in range(page_count))

        # Mode PyMuPDF (Digital)
        if (method in ("auto", "pymupdf")) and total_digital_words >= 25:
            boxes = []
            full_text_pages = []
            y_offset = 0.0

            for p_num in range(page_count):
                page = doc[p_num]
                page_h = page.rect.height
                raw_words = page.get_text("words")
                for w in raw_words:
                    x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4].strip()
                    if text:
                        boxes.append(DocumentBox(
                            x0=x0,
                            y0=y0 + y_offset,
                            x1=x1,
                            y1=y1 + y_offset,
                            text=text,
                            score=1.0,
                            page_num=p_num
                        ))
                full_text_pages.append(page.get_text())
                y_offset += page_h + 100.0

            full_text = "\n".join(full_text_pages)
            doc.close()
            logger.info(f"Berhasil mengekstrak {len(boxes)} kata dari {page_count} halaman via PyMuPDF.")
            return OCRResult(
                raw_text=full_text,
                page_count=page_count,
                source_file=pdf_path,
                extraction_method="pymupdf_words",
                boxes=boxes,
                confidence=1.0
            )

        # Mode Fallback RapidOCR (Scan/Vektor)
        if (ocr_fallback and method == "auto") or method == "rapidocr":
            logger.info(f"PDF berformat scan/vektor ({page_count} halaman). Menjalankan RapidOCR ONNX...")
            boxes = []
            full_text_pages = []
            total_score = 0.0
            y_offset = 0.0
            engine = self._get_rapid_engine()

            for p_num in range(page_count):
                page = doc[p_num]
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                results, _ = engine(img_bytes)

                if results:
                    for item in results:
                        pts = item[0]
                        text = item[1].strip()
                        score = float(item[2])
                        x0 = min(p[0] for p in pts)
                        y0 = min(p[1] for p in pts)
                        x1 = max(p[0] for p in pts)
                        y1 = max(p[1] for p in pts)
                        boxes.append(DocumentBox(
                            x0=x0,
                            y0=y0 + y_offset,
                            x1=x1,
                            y1=y1 + y_offset,
                            text=text,
                            score=score,
                            page_num=p_num
                        ))
                        total_score += score
                y_offset += pix.height + 150.0

            doc.close()
            sorted_boxes = sorted(boxes, key=lambda b: (round(b.cy / 18), b.x0))
            full_text = " ".join(b.text for b in sorted_boxes)
            avg_score = (total_score / len(boxes)) if boxes else 0.0

            logger.info(f"RapidOCR selesai ({len(boxes)} elemen teks terdeteksi dari {page_count} halaman, akurasi: {avg_score:.2f}).")
            return OCRResult(
                raw_text=full_text,
                page_count=page_count,
                source_file=pdf_path,
                extraction_method="rapidocr_onnx",
                boxes=boxes,
                confidence=avg_score
            )

        doc.close()
        return OCRResult(raw_text="", page_count=page_count, source_file=pdf_path, extraction_method="none")