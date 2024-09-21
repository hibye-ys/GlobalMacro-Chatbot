from langchain_openai import ChatOpenAI
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import SimpleJsonOutputParser
import yaml
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from utils import load_yaml


class Summarizer:
    def __init__(
        self,
        config,
        summary_type="reduce",
        use_cod=True,
        sm_chunker=False,
        test_count=0,
    ):
        self.sm_chunker = sm_chunker
        self.summary_type = summary_type
        self.filetype = config["settings"]["filetype"]
        self.test_count = test_count
        self.use_cod = use_cod
        self.base_path = config["settings"]["base_path"]
        self.edit_path = config["settings"]["edit_path"]
        self.category_id = config["settings"]["category_id"]
        self.raw_path = f"{self.base_path}/{self.category_id}/{self.filetype}"
        self.output_path = f"{self.edit_path}/{self.category_id}/{self.filetype}"

        self.llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini-2024-07-18")
        self.langsmith = f"{self.category_id}Summary"

        self.cod_prompt_template = self.load_yaml(
            "../prompts/summarization/chain_of_density.yaml"
        )["knowledge_graph_prompt2"]
        self.map_prompt_template = self.load_yaml(
            "../prompts/summarization/map_prompt.yaml"
        )["prompt"]
        self.reduce_prompt_template = self.load_yaml(
            "../prompts/summarization/reduce_prompt.yaml"
        )["prompt"]
        self.refine_prompt_template = self.load_yaml(
            "../prompts/summarization/refine_prompt.yaml"
        )["prompt"]

        self.cod_prompt = PromptTemplate.from_template(self.cod_prompt_template)
        self.map_prompt = PromptTemplate.from_template(self.map_prompt_template)
        self.reduce_prompt = PromptTemplate.from_template(self.reduce_prompt_template)
        self.refine_prompt = PromptTemplate.from_template(self.refine_prompt_template)

        self.map_chain = self.map_prompt | self.llm | StrOutputParser()
        self.reduce_chain = self.reduce_prompt | self.llm | StrOutputParser()
        self.refine_chain = self.refine_prompt | self.llm | StrOutputParser()

    def load_yaml(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def split_MDdocs(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            md_content = file.read()

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        split_MDdocs = splitter.split_text(md_content)

        num_tokens = self.llm.get_num_tokens(md_content)

        print(f"document의 token 수는 {num_tokens}개 입니다")
        print(f"document의 split 개수는 {len(split_MDdocs)}개 입니다")

        return split_MDdocs

    def split_TXTdocs(self, filename):
        loader = TextLoader(filename)
        doc = loader.load()
        full_text = doc[0].page_content

        if self.sm_chunker:
            splitter = SemanticChunker(
                OpenAIEmbeddings(),
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=70,
            )
            split_TXTdocs = splitter.create_documents([full_text])
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)

            split_TXTdocs = splitter.split_documents(doc)
        num_tokens = self.llm.get_num_tokens(full_text)

        print(f"document의 token 수는 {num_tokens}개 입니다")
        print(f"document의 split 개수는 {len(split_TXTdocs)}개 입니다")

        return split_TXTdocs

    def directory_exists(self, output_path):
        directory = os.path.dirname(output_path)

        if not os.path.exists(directory):
            os.makedirs(directory)

    def map_summaries_to_file(self, summary, title):
        path = f"{self.output_path}/{title}_MapSummary{self.test_count}.{self.filetype}"
        with open(path, "a", encoding="utf-8") as file:
            file.write(summary + "\n\n" + "-" * 50 + "\n\n")

    def chain_of_density(self, docs, title=None):
        cod_chain_inputs = {
            "content": lambda d: d.get("content"),
            "content_category": lambda d: d.get(
                "content_category", "Financial Article"
            ),
            "entity_range": lambda d: d.get("entity_range", "1-3"),
            "max_words": lambda d: int(d.get("max_words", 500)),
            "iterations": lambda d: int(d.get("iterations", 5)),
        }

        cod_chain = (
            cod_chain_inputs | self.cod_prompt | self.llm | SimpleJsonOutputParser()
        )

        cod_docs = []
        for content in docs:
            # 결과를 저장할 빈 리스트 초기화
            results: list[dict[str, str]] = []

            # cod_chain을 스트리밍 모드로 실행하고 부분적인 JSON 결과를 처리
            for partial_json in cod_chain.stream(
                {"content": content, "content_category": "Financial Article"}
            ):
                # 각 반복마다 results를 업데이트
                results = partial_json

                # 현재 결과를 같은 줄에 출력 (캐리지 리턴을 사용하여 이전 출력을 덮어씀)
                print(results, end="\r", flush=True)

            total_summaries = len(results)
            print("\n")

            # 각 요약을 순회하며 처리
            i = 1
            for cod in results:
                # 누락된 엔티티들을 추출하고 포맷팅
                added_entities = ", ".join(
                    [
                        ent.strip()
                        for ent in cod.get(
                            "missing_entities", 'ERR: "missing_entiies" key not found'
                        ).split(";")
                    ]
                )
                # 더 밀도 있는 요약 추출
                summary = cod.get("denser_summary", 'ERR: missing key "denser_summary"')

                i += 1
            if title is not None:
                self.map_summaries_to_file(summary, title)
            cod_docs.append(summary)

        if self.summary_type == "only_cod":
            return cod_docs

        elif self.summary_type == "refine":
            previous_summary = cod_docs[0]
            for current_summary in cod_docs[1:]:

                previous_summary = self.refine_chain.invoke(
                    {
                        "previous_summary": previous_summary,
                        "current_summary": current_summary,
                        "language": "Korean",
                    }
                )
            return previous_summary

        elif self.summary_type == "reduce":
            return self.reduce_chain.invoke(
                {"doc_summaries": cod_docs, "language": "Korean"}
            )

    def map_reduce_chain(self, docs):
        doc_summaries = [
            self.map_chain.invoke({"documents": doc, "language": "Korean"})
            for doc in docs
        ]
        return self.reduce_chain.invoke(
            {"doc_summaries": doc_summaries, "language": "Korean"}
        )

    def map_refine_chain(self, docs):
        input_doc = [
            {"documents": doc.page_content, "language": "Korean"} for doc in docs
        ]

        doc_summaries = self.map_chain.batch(input_doc)

        previous_summary = doc_summaries[0]

        for current_summary in doc_summaries[1:]:

            previous_summary = self.refine_chain.invoke(
                {
                    "previous_summary": previous_summary,
                    "current_summary": current_summary,
                    "language": "Korean",
                }
            )

        return previous_summary

    def run_chain(self):

        self.directory_exists(f"{self.output_path}/")

        raw_files = set(
            x.split(f".{self.filetype}")[0] for x in os.listdir(self.raw_path)
        )
        edit_files = set(
            x.split(f"_{self.summary_type}")[0] for x in os.listdir(self.output_path)
        )

        new_files = raw_files - edit_files
        print(new_files)
        if not new_files:
            print("업데이트할 파일이 없습니다")

        for file_name in list(new_files):
            raw_file_path = os.path.join(self.raw_path, file_name + f".{self.filetype}")

            if self.filetype == "md":
                splitted_docs = self.split_MDdocs(raw_file_path)
            elif self.filetype == "txt":
                splitted_docs = self.split_TXTdocs(raw_file_path)

            if self.use_cod:
                summary_result = self.chain_of_density(splitted_docs, file_name)
            elif self.summary_type == "refine" and not self.use_cod:
                summary_result = self.map_refine_chain(splitted_docs)
            elif self.summary_type == "reduce" and not self.use_cod:
                summary_result = self.map_reduce_chain(splitted_docs)

            output_file_path = f"{self.edit_path}/{self.category_id}/{self.filetype}/{file_name}_{self.summary_type}Summary{self.test_count}.{self.filetype}"

            with open(output_file_path, "w", encoding="utf-8") as file:
                file.write(summary_result)

            print(f"{raw_file_path} 요약이 완료되었습니다")


if __name__ == "__main__":
    config = load_yaml("../config/summarizer.yaml")
    summarizer = Summarizer(
        config, summary_type="refine", use_cod=True, sm_chunker=False, test_count=0
    )
    summarizer.run_chain()
