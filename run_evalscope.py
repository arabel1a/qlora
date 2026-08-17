import json
from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments, parse_args
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('custom')
class CustomDatasetPlugin(DatasetPluginBase):
    """Reads the dataset and returns prompts."""

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        """Construct the message list."""
        with open(self.query_parameters.dataset_path, 'r', encoding="utf-8") as f:
            text_generator = [json.loads(line) for line in f]
        
        for item in text_generator:
            prompt = item["question"].strip()
            if (
                len(prompt) > self.query_parameters.min_prompt_length
                and len(prompt) < self.query_parameters.max_prompt_length
            ):
                if self.query_parameters.apply_chat_template:
                    yield [{'role': 'user', 'content': prompt}]
                else:
                    yield prompt


if __name__ == '__main__':
    # see evalscope/perf/main.py
    args = Arguments.from_args(parse_args())
    run_perf_benchmark(args)
