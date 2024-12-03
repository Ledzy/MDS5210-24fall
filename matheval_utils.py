import re
import json
import torch
import os
from statistics import mean
from cot_prompts import get_examples
from vllm import LLM, SamplingParams

if not os.path.exists("results"):
    os.makedirs("results")

def get_seperation_trigger(dataset: str):
    triggers = ['The answer is:', 'The answer is', 'the answer is']
    # the answer format of gsm8k is a bit different
    if dataset == 'gsm8k':
        triggers.append('####')
    return triggers

def answer_clean(dataset: str, pred: str):
    """ This function essentially extract the str after 'the answer is', with additional operation that cleans the string.
    Input:
        dataset: str, the dataset name (e.g., math, aqua, gsm8k, etc.)
        pred: str, the model's generation
    Output:
        pred: str, the extracted answer
    """
    direct_answer_trigger_for_fewshot = get_seperation_trigger(dataset)
    pred = pred.strip('\n')

    # Determine if this is in-context learning (ICL), if so, use \n\n to split the first chunk.
    # NOTE: in-context learning means that we add additional sample (question, answer) pairs into the prompt, which 
    # may boost the performance of the model
    ICL = False
    for trigger in direct_answer_trigger_for_fewshot:
        if pred.count(trigger) > 1:
            ICL = True
    if ICL:
        pred = pred.split('\n\n')[0]

    # Split the trigger to find the answer.
    preds = re.split('|'.join(direct_answer_trigger_for_fewshot), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip('\n').rstrip('.').rstrip('/').strip(' ')

    # Clean the answer based on the dataset
    if dataset in ("aqua", "sat", "arc") or "mmlu" in dataset:
        tmp = re.findall(r'\b(A|B|C|D|E|F|G|H|I|J)\b', pred.upper())
        if tmp:
            pred = tmp
        else:
            pred = [pred.strip().strip('.')]
    elif dataset in ("numglue",):
        tmp = re.findall(r'\b(A|B)\b', pred.upper())
        if tmp:
            pred = tmp
        else:
            pred = pred.replace(",", "")
            pred = [delete_extra_zero(s.replace(",", "")) for s in re.findall(r'-?\d+/?\.?\d*', pred)]
    elif dataset in ("gsm8k", "svamp", "deepmind", "simuleq"):
        pred = pred.replace(",", "")
        pred = [delete_extra_zero(s.replace(",", "")) for s in re.findall(r'-?\d+/?\.?\d*', pred)]
    elif dataset in ("math",):
        pred = [extract_math_answer(pred, answer_flag)]
    elif "gpqa" in dataset:
        tmp = re.findall(r'\b(A|B|C|D)\b', pred.upper())
        if tmp:
            pred = tmp
        else:
            pred = [pred.strip().strip('.')]
    elif dataset in ("theoremqa",):
        pred = [extract_theoremqa_answer(pred, answer_flag)]
    elif "bbh" in dataset:
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last e
            pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip('.').rstrip('/')

    return pred

def data_reader(dataset: str, base_path="data/math_finetuning"):
    """read the validation dataset
    
    Return:
        questions: list(str) questions
        answers: list(str) groundtruth answers
    """
    
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if dataset == "aqua":
        with open(f'{base_path}/AQuA/AQuA.json') as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "(" + "(".join(json_res["options"])
                choice = choice.replace("(", " (").replace(")", ") ")
                choice = "Answer Choices:" + choice
                questions.append(json_res["question"].strip() + "\n" + choice)
                answers.append(json_res["correct"])
    elif dataset == 'math':
        with open(f'{base_path}/math/MATH.json', 'r') as f:
            loaded = json.load(f)
        for d in loaded:
            questions.append(d['question'])
            answers.append(d['answer'])
    elif dataset == "gsm8k":
        with open(f'{base_path}/gsm8k/gsm8k.jsonl') as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(delete_extra_zero(json_res["answer"].split("#### ")[-1].replace(",", "")))
    elif dataset == "svamp":
        with open(f'{base_path}/SVAMP/SVAMP.json') as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(delete_extra_zero(a))
    elif dataset == 'theoremqa':
        with open(f'{base_path}/theoremqa/theoremqa_test.json') as f:
            test_set = json.load(f)
            for row in test_set:
                questions.append(row['Question'])
                if isinstance(row['Answer'], bool):
                    answers.append([str(row['Answer']), None])
                elif isinstance(row['Answer'], (list, int, float)):
                    answers.append([str(row['Answer']), row['Answer']])
                else:
                    answers.append([str(row['Answer']), None])
    elif dataset == 'arc':
        with open(f'{base_path}/arc/challenge.json') as f:
            test_set = json.load(f)
            for row in test_set:
                questions.append(row['question'])
                answers.append(row['answer'])
    elif dataset == 'mmlu_pro':
        with open(f'{base_path}/mmlu_pro/test.json') as f:
            json_data = json.load(f)
            for line in json_data:
                questions.append(line['question'])
                answers.append(line['answer'])
    elif 'mmlu' in dataset:
        with open(f'{base_path}/mmlu/{dataset.split("_")[1]}.json') as f:
            json_data = json.load(f)
            for line in json_data:
                options = f'(A) {line["choices"][0]} (B) {line["choices"][1]} (C) {line["choices"][2]} (D) {line["choices"][3]}'
                q = line["question"] + '\n' + 'Answer Choices: ' + options
                a = ['A', 'B', 'C', 'D'][line['answer']]
                questions.append(q)
                answers.append(a)
    elif dataset in ['numglue', 'simuleq', 'deepmind', 'sat']:
        with open(f'{base_path}/{dataset}/{dataset}.json') as f:
            json_data = json.load(f)
            for line in json_data:
                assert isinstance(line['question'], str) and isinstance(line['question'], str), line
                questions.append(line['question'])
                answers.append(str(line['answer']))
    elif 'gpqa' in dataset:
        with open(f'{base_path}/gpqa/{dataset}.jsonl') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                tmp = generate_question_and_answers(data)
                questions.append(tmp['question'])
                answers.append(tmp['answer'])
    elif 'bbh' in dataset:
        with open(f'{base_path}/bbh/bbh.json', 'r') as f:
            test_set = json.load(f)
        for entry in test_set:
            questions.append(entry['question'])
            answers.append(entry['answer'])
    else:
        raise ValueError("dataset is not properly defined ...")

    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)

    print("dataset : {}".format(dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))

    return questions, answers

def delete_extra_zero(n):
    try:
        n=float(n)
    except:
        try:
            n = eval(n)
        except:
            print("Conversion to floating number fails: {}".format(n))
            return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')  # 删除小数点后多余的0
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  # 只剩小数点直接转int，否则转回float
        n=str(n)
        return n

def compare_answer_with_groundtruth(answer: str, groundtruth_str: str, groundtruth_num = None):
    # Stripping away the text symbol
    symbols = ['\\text{', '\\boxed{']
    
    for symbol in symbols:
        if symbol in answer:
            answer = answer.replace(symbol, '').rstrip('}')
        if symbol in groundtruth_str:
            groundtruth_str = groundtruth_str.replace(symbol, '').rstrip('}')

    if groundtruth_str.lower() in ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']:
        return groundtruth_str.lower() in answer.lower()
    elif answer.lower() == groundtruth_str.lower():
        return True
    elif groundtruth_num is not None:
        if isinstance(groundtruth_num, (int, float)):
            return compare_two_numbers(number_it(answer), groundtruth_num)
        else:
            try:
                answer = list(eval(answer))
                answer = [number_it(a) for a in answer]
            except Exception as e:
                return False
            return compare_two_list(answer, groundtruth_num)
    else:
        return False
    
def compare_two_numbers(p, gt):
    try:
        if math.isnan(p):
            return False
        if isinstance(gt, int):
            return round(p) == gt
        else:
            return within_eps(pred=p, gt=gt)
    except Exception:
        return False


def compare_two_list(pred, gt):
    if not isinstance(pred, list):
        return False
    elif len(pred) != len(gt):
        return False
    elif any([not isinstance(x, (int, float)) for x in pred]):
        return False
    else:
        pred = sorted(pred)
        gt = sorted(gt)
        return all([compare_two_numbers(p, g) for p, g in zip(pred, gt)])

dataset = 'gsm8k' # options: 'gsm8k' 'math' 'svamp' 'simuleq' 'numglue'
template = "### Instruction:\n{}\n\n### Response:{}\n\n"
num_samples = -1
model_name = "meta-llama/Llama-3.2-3B"

inference_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float32"
model = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count(), dtype=inference_dtype, trust_remote_code=True, max_model_len=4096)
questions, groundtruths = data_reader(dataset)

cot_prompt = "" # TODO: implement the cot_prompt, you may find the function get_examples being helpful

prompts = [cot_prompt + "### Instruction:\n{}\n\n### Response:".format(q) for q in questions[:num_samples]]
outputs = model.generate(prompts, SamplingParams(temperature=0.8, top_p=0.95, max_tokens=500))

correct, wrong = 0, 0
for i, output in enumerate(outputs):
    result = {}
    prompt = output.prompt
    response = output.outputs[0].text
    
    # extract answer from the generated text
    # TODO: improve the answer_clean function
    model_answer = answer_clean(dataset, response)
    
    # TODO: improve the compare_answer_with_groundtruth function
    if compare_answer_with_groundtruth(model_answer, groundtruths[i]):
        correct += 1
        result["correctness"] = True
    else:
        wrong += 1
        result["correctness"] = False
    
    result["model_answer"] = model_answer
    result["groundtruth"] = groundtruths[i]
    result["prompt"] = prompt
    result["response"] = response
    
    with open(f"results/{model_name.split('/')[-1]}_{dataset}.jsonl", "a") as f:
        json.dump(result, f)
        f.write("\n")

accuracy = correct / (correct + wrong)
print(f"Accuracy: {accuracy}")