import base64
import io
import os
import re
import traceback

from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import TableFormerMode
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()


def _describe_image_with_openai(img_b64: str) -> str:
    """Call an OpenAI vision model to generate a rich, searchable description."""
    vision_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    vision_llm = ChatOpenAI(
    model=vision_model,
    api_key=os.getenv("OPENAI_API_KEY"),
    )
    msg = HumanMessage(
    content=[
        {
        "type": "text",
        "text": (
            "Describe this image in detail for document search indexing. "
            "Include chart titles, axis labels, legend entries, key data "
            "points, trends, numbers, and any visible text. Be specific. "
            "The description you generate is for a RAG bot."
            ),
        },
        {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        },
    ]
)

    try:
        response = vision_llm.invoke([msg])
        content = response.content
        if isinstance(content, list):
            return " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ).strip()
        return str(content).strip()
    except Exception:
        return ""

def _make_metadata(content_type: str, element_type: str, page_no: int |None, source_file: str, position: dict| None,
                   heading_stack: dict, img_b64:str|None = None) -> dict :
            return {
            "content_type": content_type,
            "element_type": element_type,
            "section": ">".join(heading_stack.values()) if heading_stack else None,
            "page_number": page_no,
            "source_file": source_file,
            "position": position,
            "image_base64": img_b64,
        }
def parse_document(file_path: str) -> list[dict]:
    """Parse a PDF into a flat list of contextualized, typed content chunks using Docling."""
    try: 
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
            generate_picture_images=True,
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        )
        pipeline_options.table_structure_options.mode = TableFormerMode.FAST

        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            },
        )

        result = converter.convert(file_path)
        doc = result.document

        parsed_chunks: list[dict] = []

    # Heading hierarchy stack: tracks [Level 1, Level 2, Level 3...]
        heading_stack: dict[int, str] = {}
        source_file = os.path.basename(file_path)

        for item in doc.iterate_items():
            if isinstance(item, tuple):
                node, level = item
            else:
                node = item
                level = getattr(node, "level", 1)

            label = str(getattr(node, "label", "")).lower()

            # Skip headers and footers
            if label in ("page_header", "page_footer"):
                continue

            prov = getattr(node, "prov", None)
            page_no = prov[0].page_no if prov else None
            position: dict | None = None
            if prov and hasattr(prov[0], "bbox") and prov[0].bbox is not None:
                b = prov[0].bbox
                position = {"l": b.l, "t": b.t, "r": b.r, "b": b.b}

            # Handle Section Headings & Hierarchy updates
            if "section_header" in label or label == "title":
                text = getattr(node, "text", "").strip()
                if text:
            # Update current level and prune deeper levels
                    heading_stack[level] = text
                    heading_stack = {k: v for k, v in heading_stack.items() if k <= level}

            # Build active context breadcrumb string (e.g. "Header 1 > Subheader 2")
                    breadcrumb = " > ".join(heading_stack.values())

                    parsed_chunks.append({
                        "content": f"[Section:{breadcrumb}]",
                        "content_type": "text",
                        "metadata": _make_metadata("text",label,page_no,source_file,position,heading_stack)
            })
                continue
            breadcrumb = " > ".join(heading_stack.values()) if heading_stack else ""
    

    

# ── Tables ────────────────────────────────────────────────────────────
            if "table" in label:
                    table_text = ""
                    if hasattr(node, "export_to_dataframe"):
                        try:
                            df = node.export_to_dataframe()
                            if df is not None and not df.empty:
                                rows_text: list[str] = []
                                headers = [str(c).strip() for c in df.columns]
                                for _, row in df.iterrows():
                                    pairs = [
                                        f"{h}: {str(v).strip()}"
                                        for h, v in zip(headers, row)
                                        if str(v).strip() not in ("", "nan", "None")
                                    ]
                                    if pairs:
                                        rows_text.append(" | ".join(pairs))
                                table_text = "\n".join(rows_text)
                        except Exception:
                            pass

                    if not table_text and hasattr(node, "export_to_html"):
                        try:
                            raw_html = node.export_to_html(doc)
                            table_text = re.sub(r"<[^>]+>", " ", raw_html or "")
                            table_text = re.sub(r"\s+", " ", table_text).strip()
                        except Exception:
                            pass

                    if not table_text:
                        table_text = getattr(node, "text", "")

                    if table_text and table_text.strip():
                        # Enrich chunk text with structural section breadcrumbs
                        contextualized_content = (
                            f"Section Context: {breadcrumb}\nTable Content:\n{table_text.strip()}"
                            if breadcrumb else table_text.strip()
                        )
                        parsed_chunks.append(
                        {
                            "content": contextualized_content,
                            "content_type": "table",
                            "metadata": _make_metadata("table",label,page_no,source_file,position,heading_stack),
                        }
                        )

    # ── Pictures / Charts / Figures ───────────────────────────────────────
            elif "picture" in label or "figure" in label or label == "chart":
                    img_b64 = None
                    caption = getattr(node, "text", "") or ""

                    try:
                        if hasattr(node, "get_image"):
                            pil_img = node.get_image(doc)
                            if pil_img:
                                buf = io.BytesIO()
                                pil_img.save(buf, format="PNG")
                                img_b64 = base64.b64encode(buf.getvalue()).decode()
                    except Exception:
                        pass

                    if img_b64:
                        description = _describe_image_with_openai(img_b64)
                        raw_content = description or caption.strip() or f"[Image on page {page_no}]"
                    else:
                        raw_content = caption.strip() or f"[Image on page {page_no}]"

                    contextualized_content = (
                        f"Section Context: {breadcrumb}\nImage Context:\n{raw_content}"
                        if breadcrumb else raw_content
                    )

                    parsed_chunks.append(
                    {
                    "content": contextualized_content,
                    "content_type": "image",
                    "metadata": _make_metadata("image",label,page_no,source_file,position,heading_stack,img_b64),
                    }
                    )

            # ── Regular Text & Paragraphs ─────────────────────────────────────────
            else:
                    text = getattr(node, "text", "")
                    if text and text.strip():
                        contextualized_content = (
                        f"[{breadcrumb}]\n{text.strip()}"
                        if breadcrumb else text.strip()
                    )
                    parsed_chunks.append(
                    {
                        "content": contextualized_content,
                        "content_type": "text",
                        "metadata": _make_metadata("text", label,page_no,source_file,position,heading_stack),
                    }
                    )

        return parsed_chunks
    except Exception as e:
            print("/n" + "="*50)
            print ("crash in store chunks:")
            traceback.print_exc()
            raise e 