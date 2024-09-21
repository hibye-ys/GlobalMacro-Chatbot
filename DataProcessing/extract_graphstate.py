from typing import List, Dict
import json
from langchain_core.documents import Document


def extract_documents_for_multidocstore(json_data: Dict) -> List[Document]:
    documents = []
    for page_num, page_data in json_data["page_elements"].items():
        elements = sorted(page_data["elements"], key=lambda x: x["id"])
        page_content = "\n".join([element["content"] for element in elements])
        page_metadata = json_data["page_metadata"][str(page_num)]
        metadata = {
            "id": page_num,
            "page": int(page_num),
            "images": (
                json_data["images"].get(page_num, "") if "images" in json_data else []
            ),
            "doc_id": f"{json_data['title']}_{page_num}",
            **page_metadata,
        }
        documents.append(Document(metadata=metadata, page_content=page_content))
    return documents


def extract_documents_for_multivectorstore(json_data: Dict) -> List[Document]:
    documents = []
    for page_num, summary in json_data["text_summary"].items():
        page_content = summary
        if "image_summary" in json_data and page_num in json_data["image_summary"]:
            page_content += "\n" + json_data["image_summary"][page_num]
        page_metadata = json_data["page_metadata"].get(str(page_num), "")
        metadata = {
            "page": int(page_num),
            "images": (
                json_data["images"].get(page_num, "") if "images" in json_data else []
            ),
            "doc_id": f"{json_data['title']}_{page_num}",
            **page_metadata,
        }
        documents.append(Document(metadata=metadata, page_content=page_content))
    return documents


def extract_documents_for_singlestore(json_data: Dict) -> list[Document]:

    documents = []

    for page_number, page_data in json_data["page_elements"].items():
        page_content = ""
        images = []

        for element in page_data["elements"]:
            if element["type"] == "text":
                page_content += f"<text>{element['content']}</text>\n"
            elif element["type"] == "image":
                image_id = element["id"]
                image_summary = json_data["image_summary"].get(
                    str(image_id), "이미지 설명 없음"
                )
                page_content += f"{image_summary}\n"
                images.append(element["content"])

        page_metadata = json_data["page_metadata"].get(page_number, {})
        page_metadata["page_number"] = int(page_number)

        doc = Document(
            page_content=page_content,
            metadata={
                **page_metadata,
                "images": images,
                "doc_id": f"{json_data['title']}_{page_number}",
            },
        )
        documents.append(doc)

    return documents
