#!/usr/bin/env python
import base64
from typing import List
from mistralai import Mistral, OCRImageObject
from mistralai.models import OCRResponse
from pathlib import Path
import mimetypes
from enum import Enum
import sys

class Modes(Enum):
    FULL = 0
    FULL_ALT = 1
    FULL_NO_DIR = 2
    FULL_NO_PAGES = 3
    TEXT = 4
    TEXT_NO_PAGES = 5

def get_mode_from_string(mode_str: str):
    for mode in Modes:
        if mode.name == mode_str.upper() or mode.value == mode_str:
            return mode
    raise ValueError(f"Unknown mode: {mode_str}")

def b64encode_document(document_path: Path):
    try:
        with open(document_path, "rb") as doc_file:
            return base64.b64encode(doc_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None
    except Exception as e:
        return None
    
def b64decode_document(base64_data: str, output_path: Path):
    if ',' in base64_data:
        _, base64_str = base64_data.split(',', 1)
    else:
        base64_str = base64_data
    try:
        image_data = base64.b64decode(base64_str)
    except (base64.binascii.Error, ValueError) as e:
        print(f"Error decoding base64 data: {e}", file=sys.stderr)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(image_data)

class Page:
    def __init__(self, index, markdown=None, images:List[OCRImageObject]=None):
        self.index = index
        self.markdown = markdown
        self.images = images if images is not None else []

    def write_markdown(self, output_path: Path, append: bool = False, insert = None):
        if self.markdown:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mode = 'a' if append else 'w'
            with open(output_path, mode) as md_file:
                if insert:
                    md_file.write(insert)
                md_file.write(self.markdown)
    
    def write_images(self, output_directory: Path):
        if not self.images:
            return
        
        for image in self.images:
            if image and image.image_base64:
                image_name = image.id
                image_path = output_directory / image_name
                b64decode_document(image.image_base64, image_path)

class MistralOCRDocument:
    def __init__(self, 
                 document_path: Path, 
                 api_key: str, 
                 include_images=True,
                 output_directory: Path = None,
                 generate_pages=True,
                 full_directory_name="full",
                 page_separator="\n",
                 page_directory_name="page_<index>",
                 page_text_name="<stem>.md",
                 json_ocr_response_path=None,
                 save_json=True,
                ):
        self.document_path = document_path
        self.api_key = api_key
        self.include_images = include_images
        self.generate_pages = generate_pages
        self.save_json = save_json
        self.full_directory_name = full_directory_name
        self.page_separator = page_separator
        self.page_directory_name = page_directory_name
        self.page_text_name = page_text_name
        self.json_ocr_response_path = json_ocr_response_path
        if output_directory is None:
            self.output_directory = self.get_input_path().parent / self.get_input_path().stem
        else:
            self.output_directory = output_directory

    def get_ocr_response(self, mimetype, base64_document):
        client = Mistral(api_key=self.api_key)
        if mimetype.startswith("image/"):
            document_type = "image_url"
        elif mimetype.startswith("application/pdf"):
            document_type = "document_url"
        else:
            raise ValueError(f"Unsupported MIME type: {mimetype}. Only image and PDF files are supported.")
        self.ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": document_type,
                document_type: f"data:{mimetype};base64,{base64_document}" 
            },
            include_image_base64=self.include_images
        )

    def process_document(self):
        if not self.document_path.exists():
            raise FileNotFoundError(f"The document {self.document_path} does not exist.")
        if not self.document_path.is_file():
            raise ValueError(f"The path {self.document_path} is not a valid file.")
        
        mimetype, _ = mimetypes.guess_type(self.document_path)
        if mimetype is None:
            raise ValueError(f"Could not determine the MIME type for {self.document_path}.")
        
        self.get_ocr_response(mimetype, b64encode_document(self.document_path))
        self.write_json()
        self.process_ocr_response()

    def process_json_response(self):
        if self.json_ocr_response_path is None or not self.json_ocr_response_path.exists():
            raise FileNotFoundError(f"The JSON OCR response {self.json_ocr_response_path} does not exist.")
        
        with open(self.json_ocr_response_path, "r") as json_file:
            self.ocr_response = OCRResponse.model_validate_json(json_file.read())
        self.write_json()
        self.process_ocr_response()

    def process(self):
        if self.json_ocr_response_path is not None:
            self.process_json_response()
        else:
            self.process_document()

    def get_input_path(self):
        if self.json_ocr_response_path is not None:
            return self.json_ocr_response_path
        return self.document_path
    
    def write_json(self):
        if self.save_json:
            output_path = (self.output_directory / self.get_input_path().stem).with_suffix(".json")
            self.output_directory.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as text_file:
                text_file.write(self.ocr_response.model_dump_json(indent=2))

    def process_ocr_response(self):
        response_pages = self.ocr_response.pages
        if not response_pages:
            print("No pages found in the OCR response.")
            return
        
        pages = []

        full_dir = self.output_directory / self.full_directory_name

        for r_page in response_pages:
            page = Page(
                index=r_page.index,
                markdown=r_page.markdown,
                images=r_page.images
            )
            if self.generate_pages:
                page_dir = self.output_directory / self.page_directory_name.replace("<index>", str(page.index))
                page.write_markdown((
                    page_dir / self.page_text_name.
                    replace("<stem>", self.get_input_path().stem).
                    replace("<index>", str(page.index))
                    ).with_suffix(".md"))
                if self.include_images:
                    page.write_images(page_dir)
            if self.include_images:
                page.write_images(full_dir)
            pages.append(page)
        for i, page in enumerate(sorted(pages, key=lambda p: p.index)):
            first = i == 0
            md_file = (full_dir / self.get_input_path().stem).with_suffix(".md")
            insert = self.page_separator if not first else None
            page.write_markdown(md_file, append=not first, insert=insert)

def construct_from_mode(
    document_path: Path,
    api_key: str,
    output_directory: Path = None,
    json_ocr_response_path: Path = None,
    page_separator: str = "\n",
    write_json: bool = True,
    mode: Modes = Modes.FULL
):
    kwargs = dict(
        document_path=document_path,
        api_key=api_key,
        output_directory=output_directory,
        json_ocr_response_path=json_ocr_response_path,
        page_separator=page_separator,
        save_json=write_json
    )
    match mode:
        case Modes.FULL:
            kwargs.update(
                include_images=True,
                generate_pages=True
            )
        case Modes.FULL_ALT:
            kwargs.update(
                include_images=True,
                generate_pages=True,
                full_directory_name="."
            )
        case Modes.FULL_NO_DIR:
            kwargs.update(
                include_images=True,
                generate_pages=True,
                full_directory_name=".",
                page_directory_name=".",
                page_text_name="<stem><index>.md"
            )
        case Modes.FULL_NO_PAGES:
            kwargs.update(
                include_images=True,
                generate_pages=False,
                full_directory_name="."
            )
        case Modes.TEXT:
            kwargs.update(
                include_images=False,
                generate_pages=True,
                full_directory_name=".",
                page_directory_name=".",
                page_text_name="<stem><index>.md"
            )
        case Modes.TEXT_NO_PAGES:
            kwargs.update(
                include_images=False,
                generate_pages=False,
                full_directory_name="."
            )
        case _:
            raise ValueError(f"Unknown mode: {mode}")
    return MistralOCRDocument(**kwargs)