---
language: en
license: llama3.3
pretty_name: Meta Evaluation Result Details for Llama-3.3-70B-Instruct
dataset_summary: >+
  This dataset contains the results of the Meta evaluation result details for
  **Llama-3.3-70B-Instruct**. The dataset has been created from 12 evaluation
  tasks. The tasks are: human_eval, mmlu_pro, gpqa_diamond, ifeval__loose,
  mmlu__0_shot__cot, nih__multi_needle, mgsm, math_hard, bfcl_chat,
  ifeval__strict, math, mbpp_plus.

   Each task detail can be found as a specific subset in each configuration nd each subset is named using the task name plus the timestamp of the upload time and ends with "__details".

  For more information about the eval tasks, please refer to this [eval
  details](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/eval_details.md)
  page.


  You can use the Viewer feature to view the dataset in the web browser easily.
  For most tasks, we provide an "is_correct" column, so you can quickly get our
  accuracy result of the task by viewing the percentage of "is_correct=True".
  For tasks that have both binary (eg. exact_match) and a continuous metrics
  (eg. f1), we will only consider the binary metric for adding the is_correct
  column. This might differ from the reported metric in the model card.


  Additionally, there is a model metrics subset that contains all the reported
  metrics, like f1, macro_avg/acc, for all the tasks and subtasks. Please use
  this subset to find reported metrics in the model card.


  Lastly, you can also use Huggingface Dataset APIs to load the dataset. For
  example, to load a eval task detail, you can use the following code:


  ```python

  from datasets import load_dataset

  data = load_dataset("meta-llama/Llama-3.3-70B-Instruct-evals",
          name="Llama-3.3-70B-Instruct-evals__mbpp_plus__details",
          split="latest"
  )

  ```


  Please check our [eval
  recipe](https://github.com/meta-llama/llama-recipes/tree/main/tools/benchmarks/llm_eval_harness/meta_eval)
  that demonstrates how to calculate the our reported benchmark numbers using
  the
  [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main)
  library on selected tasks.


  Here are the detailed explanation for each column of the task eval details:


  **task_type**: Whether the eval task was run as a ‘Generative’ or ‘Choice’
  task. Generative task returns the model output, whereas for choice tasks we
  return the negative log likelihoods of the completion. (The choice task
  approach is typically used for multiple choice tasks for non-instruct models)


  **task_name**: Meta internal eval task name


  **subtask_name**: Meta internal subtask name in cases where the benchmark has
  subcategories (Ex. MMLU with domains)


  **input_question**: The question from the input dataset when available. In
  cases when that data is overwritten as a part of the evaluation pipeline or it
  is a complex concatenation of input dataset fields, this will be the
  serialized prompt object as a string.


  **input_choice_list**: In the case of multiple choice questions, this contains
  a map of the choice name to the text.


  **input_final_prompt**: The final input text that is provided to the model for
  inference. For choice tasks, this will be an array of prompts provided to the
  model, where we calculate the likelihoods of the different completions in
  order to get the final answer provided by the model.


  **input_correct_responses**: An array of correct responses to the input
  question.


  **output_prediction_text**: The model output for a Generative task 


  **output_parsed_answer**: The answer we’ve parsed from the model output or
  calculated using negative log likelihoods.


  **output_choice_completions**: For choice tasks, the list of completions we’ve
  provided to the model to calculate negative log likelihoods


  **output_choice_negative_log_likelihoods**: For choice tasks, these are the
  corresponding negative log likelihoods normalized by different sequence
  lengths (text, token, raw) for the above completions.


  **output_metrics**: Metrics calculated at the example level. Common metrics
  include:

      acc - accuracy

      em - exact_match

      f1 - F1 score

      pass@1 - For coding benchmarks, whether the output code passes tests

  **is_correct**: Whether the parsed answer matches the target responses and
  consider correct. (Only applicable for benchmarks which have such a boolean
  metric)


  **input_question_hash**: The SHA256 hash of the question text encoded as UTF-8


  **input_final_prompts_hash**: An array of SHA256 hash of the input prompt text
  encoded as UTF-8


  **benchmark_label**: The commonly used benchmark name


  **eval_config**: Additional metadata related to the configurations we used to
  run this evaluation

      num_generations - Generation parameter - how many outputs to generate

      num_shots - How many few shot examples to include in the prompt.

      max_gen_len - generation parameter (how many tokens to generate)

      prompt_fn - The prompt function with jinja template when available

      max_prompt_len - Generation parameter. Maximum number tokens for the prompt. If the input_final_prompt is longer than this configuration, we will truncate

      return_logprobs - Generation parameter - Whether to return log probabilities when generating output.

configs:
- config_name: Llama-3.3-70B-Instruct-evals__bfcl_chat__details
  data_files:
  - split: latest
    path:
    - >-
      Llama-3.3-70B-Instruct-evals/Details_bfcl_chat_2024-12-05T16-44-53.191777.parquet.gzip
- config_name: Llama-3.3-70B-Instruct-evals__gpqa_diamond__details
  data_files:
  - split: latest
    path:
    - >-
      Llama-3.3-70B-Instruct-evals/Details_gpqa_diamond_2024-12-05T16-44-45.256031.parquet.gzip
- config_name: Llama-3.3-70B-Instruct-evals__human_eval__details
  data_files:
  - split: latest
    path:
    - >-
      Llama-3.3-70B-Instruct-evals/Details_human_eval_2024-12-05T16-44-41.991636.parquet.gzip
- config_name: Llama-3.3-70B-Instruct-evals__ifeval__loose__details
  data_files:
  - split: latest
    path:
    - >-
      Llama-3.3-70B-Instruct-evals/Details_ifeval__loose_2024-12-05T16-44-45.630953.parquet.gzip
- config_name: Llama-3.3-70B-Instruct-evals__ifeval__strict__details
  data_files:
  - split: latest
    path:
    - >-
      Llama-3.3-70B-Instruct-evals/Details_ifeval__strict_2024-12-05T16-44-53.785671.parquet.gzip
- config_name: Llama-3.3-70B-Instruct-evals__math__details
  data_files:
  - split: latest
    path:
    - >-
      Llama-3.3-70B-Instruct-evals/Details_math_2024-12-05T16-44-54.263573.parquet.gzip
- config_name: Llama-3.3-70B-Instruct-evals__math_hard__details
  data_files:
  - split: latest
    path:
    - >-
      Llama-3.3-70B-Instruct-evals/Details_math_hard_2024-12-05T16-44-52.695530.parquet.gzip
- config_name: Llama-3.3-70B-Instruct-evals__mbpp_plus__details
  data_files:
  - split: latest
    path:
    - >-
      Llama-3.3-70B-Instruct-evals/Details_mbpp_plus_2024-12-05T16-44-55.009331.parquet.gzip
- config_name: Llama-3.3-70B-Instruct-evals__metrics
  data_files:
  - split: latest
    path:
    - >-
      Llama-3.3-70B-Instruct-evals/Details_metrics_details_2024-12-05T16-44-55.360252.parquet.gzip
- config_name: Llama-3.3-70B-Instruct-evals__mgsm__details
  data_files:
  - split: latest
    path:
    - >-
      Llama-3.3-70B-Instruct-evals/Details_mgsm_2024-12-05T16-44-52.199436.parquet.gzip
- config_name: Llama-3.3-70B-Instruct-evals__mmlu__0_shot__cot__details
  data_files:
  - split: latest
    path:
    - >-
      Llama-3.3-70B-Instruct-evals/Details_mmlu__0_shot__cot_2024-12-05T16-44-46.184283.parquet.gzip
- config_name: Llama-3.3-70B-Instruct-evals__mmlu_pro__details
  data_files:
  - split: latest
    path:
    - >-
      Llama-3.3-70B-Instruct-evals/Details_mmlu_pro_2024-12-05T16-44-42.594223.parquet.gzip
- config_name: Llama-3.3-70B-Instruct-evals__nih__multi_needle__details
  data_files:
  - split: latest
    path:
    - >-
      Llama-3.3-70B-Instruct-evals/Details_nih__multi_needle_2024-12-05T16-44-48.274780.parquet.gzip
---

# Dataset Card for Meta Evaluation Result Details for Llama-3.3-70B-Instruct

<!-- Provide a quick summary of the dataset. -->

This dataset contains the results of the Meta evaluation result details for **Llama-3.3-70B-Instruct**. The dataset has been created from 12 evaluation tasks. The tasks are: human_eval, mmlu_pro, gpqa_diamond, ifeval__loose, mmlu__0_shot__cot, nih__multi_needle, mgsm, math_hard, bfcl_chat, ifeval__strict, math, mbpp_plus.

 Each task detail can be found as a specific subset in each configuration nd each subset is named using the task name plus the timestamp of the upload time and ends with "__details".

For more information about the eval tasks, please refer to this [eval details](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/eval_details.md) page.

You can use the Viewer feature to view the dataset in the web browser easily. For most tasks, we provide an "is_correct" column, so you can quickly get our accuracy result of the task by viewing the percentage of "is_correct=True". For tasks that have both binary (eg. exact_match) and a continuous metrics (eg. f1), we will only consider the binary metric for adding the is_correct column. This might differ from the reported metric in the model card.

Additionally, there is a model metrics subset that contains all the reported metrics, like f1, macro_avg/acc, for all the tasks and subtasks. Please use this subset to find reported metrics in the model card.

Lastly, you can also use Huggingface Dataset APIs to load the dataset. For example, to load a eval task detail, you can use the following code:

```python
from datasets import load_dataset
data = load_dataset("meta-llama/Llama-3.3-70B-Instruct-evals",
        name="Llama-3.3-70B-Instruct-evals__mbpp_plus__details",
        split="latest"
)
```

Please check our [eval recipe](https://github.com/meta-llama/llama-recipes/tree/main/tools/benchmarks/llm_eval_harness/meta_eval) that demonstrates how to calculate the our reported benchmark numbers using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) library on selected tasks.

Here are the detailed explanation for each column of the task eval details:

**task_type**: Whether the eval task was run as a ‘Generative’ or ‘Choice’ task. Generative task returns the model output, whereas for choice tasks we return the negative log likelihoods of the completion. (The choice task approach is typically used for multiple choice tasks for non-instruct models)

**task_name**: Meta internal eval task name

**subtask_name**: Meta internal subtask name in cases where the benchmark has subcategories (Ex. MMLU with domains)

**input_question**: The question from the input dataset when available. In cases when that data is overwritten as a part of the evaluation pipeline or it is a complex concatenation of input dataset fields, this will be the serialized prompt object as a string.

**input_choice_list**: In the case of multiple choice questions, this contains a map of the choice name to the text.

**input_final_prompt**: The final input text that is provided to the model for inference. For choice tasks, this will be an array of prompts provided to the model, where we calculate the likelihoods of the different completions in order to get the final answer provided by the model.

**input_correct_responses**: An array of correct responses to the input question.

**output_prediction_text**: The model output for a Generative task 

**output_parsed_answer**: The answer we’ve parsed from the model output or calculated using negative log likelihoods.

**output_choice_completions**: For choice tasks, the list of completions we’ve provided to the model to calculate negative log likelihoods

**output_choice_negative_log_likelihoods**: For choice tasks, these are the corresponding negative log likelihoods normalized by different sequence lengths (text, token, raw) for the above completions.

**output_metrics**: Metrics calculated at the example level. Common metrics include:

    acc - accuracy

    em - exact_match

    f1 - F1 score

    pass@1 - For coding benchmarks, whether the output code passes tests

**is_correct**: Whether the parsed answer matches the target responses and consider correct. (Only applicable for benchmarks which have such a boolean metric)

**input_question_hash**: The SHA256 hash of the question text encoded as UTF-8

**input_final_prompts_hash**: An array of SHA256 hash of the input prompt text encoded as UTF-8

**benchmark_label**: The commonly used benchmark name

**eval_config**: Additional metadata related to the configurations we used to run this evaluation

    num_generations - Generation parameter - how many outputs to generate

    num_shots - How many few shot examples to include in the prompt.

    max_gen_len - generation parameter (how many tokens to generate)

    prompt_fn - The prompt function with jinja template when available

    max_prompt_len - Generation parameter. Maximum number tokens for the prompt. If the input_final_prompt is longer than this configuration, we will truncate

    return_logprobs - Generation parameter - Whether to return log probabilities when generating output.