import yaml
from openai import OpenAI
import time
from dotenv import load_dotenv
from datetime import datetime
import os
from langchain_community.document_loaders import PyPDFLoader
import json

load_dotenv()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class BatchAPI:
    def __init__(self, input_path, output_path, prompts, model, task, filetype):
        self.input_path = input_path  # 한번에 처리해야할 문서들 디렉토리
        self.output_path = output_path  # 처리된 문서를 저장할 디렉토리
        self.prompts = prompts  # 반복 프로세스의 프롬프트
        self.model = model  # 사용할 모델
        self.client = OpenAI()
        self.input_list = [
            x
            for x in os.listdir(input_path)
            if not x.startswith(".") and not x.endswith(".jsonl")
        ]
        self.task = task
        self.jsonl_path = [
            x
            for x in os.listdir(input_path)
            if not x.startswith(".") and x.endswith(".jsonl")
        ]
        self.filetype = filetype

    def create_jsonl(self):
        with open(
            f"{self.input_path}/{datetime.now().strftime('%Y-%m-%d')}_{self.task}.jsonl",
            encoding="utf-8",
            mode="w",
        ) as file:
            for input in self.input_list:
                with open(
                    os.path.join(self.input_path, input),
                    "r",
                    encoding="utf-8",
                ) as f:
                    if self.filetype == "json":
                        text = json.load(f)["text"]
                    elif self.filetype == "pdf":
                        text = PyPDFLoader(f)
                    else:
                        text = f.read()

                document = {
                    "custom_id": f"{input}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": self.prompts},
                            {"role": "user", "content": text},
                        ],
                    },
                }

                file.write(json.dumps(document))
                file.write("\n")
        return print("jsonl 파일 생성 완료")

    def upload_batch(self):
        sorted_dates = sorted(
            self.jsonl_path,
            key=lambda date: datetime.strptime(date.split("_")[0], "%Y-%m-%d"),
            reverse=True,
        )
        batch_file = self.client.files.create(
            file=open(f"{self.input_path}/{sorted_dates[0]}", "rb"), purpose="batch"
        )

        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print("upload batch complete")
        print(f"batch_job: {batch_job}")
        return batch_job

    def save_result(self, result):
        with open(f"{self.input_path}/{self.task}_processed.jsonl", "wb") as file:
            file.write(result)
        print("save result complete")

    def get_batch_job(self):
        return self.client.batches.list().data

    def check_status(self, batch_job):
        batch_job = self.client.batches.retrieve(batch_job.id)
        if batch_job.status == "completed":
            if batch_job.output_file_id:
                result_file_id = batch_job.output_file_id
            elif batch_job.error_file_id:
                result_file_id = batch_job.error_file_id
            result = self.client.files.content(result_file_id).content
            self.save_result(result)

            print("process batch download complete")
        else:
            print(batch_job.status)
            print("process batch not completed")

    def load_result(self):
        with open(f"{self.input_path}/{self.task}_processed.jsonl", "r") as file:
            return file.readlines()

    def extract_data(self):
        results = self.load_result()
        for result in results:
            data = json.loads(result)
            content = data["response"]["body"]["choices"][0]["message"]["content"]
            custom_id = data["custom_id"]

            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

            output_file = os.path.join(self.output_path, f"{custom_id}")

            with open(output_file, "w") as file:
                if self.filetype == "json":
                    json.dump(content, file)
                else:
                    file.write(content)
