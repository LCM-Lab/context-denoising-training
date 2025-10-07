import sys, os, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import itertools
from peft import peft_model, PeftModelForCausalLM
import numpy as np
import datasets
logger.info(sys.path)
from retrieval_head_detection import SentenceSampler
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from utils import *

def begin_test(args, question, answer, selected_idx, model, tokenizer, depth_percent, background_text, disturb_pos,disturb_tok_needles, evidence, evidence_list, save_file_name, model_name, use_emoji, is_0k, with_adapter=False, start_layer = 0):
    if background_text is not None:
        if use_emoji:
            emojis10 = get_random_emoji(tokenizer, 10, return_idx = True, seed = 42)
            background_text, emoji_pos = random_combine(background_text, emojis10, 
                                                         return_snd_pos = True, seed = 42)
            emoji_pos = set(emoji_pos)
            cumsum_num = 0
            emoji_spans = []

        depth_percent = [i / 10 for i in depth_percent]
        updated_sample = [[] for _ in range(len(background_text) + 1)]
        real_pos = [int(len(background_text) * i) for i in depth_percent]
        for fact, pos in zip(evidence, real_pos):  # insert real needle
            updated_sample[pos].append(fact)
        for fact, pos in zip(disturb_tok_needles, disturb_pos):  # insert disturb needle
            updated_sample[pos].append(fact)

        for i, s in enumerate(background_text):  # insert irrevelent needle
            if use_emoji and (i in emoji_pos):
                cur_pos = sum((len(l) for l in updated_sample[i]), 0)
                emoji_spans +=[(cumsum_num + cur_pos, cumsum_num + cur_pos + len(s))]
            updated_sample[i].append(s)

            if use_emoji:
                cumsum_num += sum((len(l) for l in updated_sample[i]), 0)
    else:
        updated_sample = random_combine(evidence[:-1], disturb_tok_needles+[evidence[-1]], seed = 42)
        updated_sample = [[k] for k in updated_sample]
    
    if not use_emoji or is_0k:
        emoji_spans = []
        

    flat = [i for s in updated_sample for i in s]
    tokens = [i for s in flat for i in s]

    new_context = tokenizer.decode(tokens)
    input_context = new_context + f"\n{question}\nAnswer:"
    if tokenizer.chat_template is not None:
        shift = 30
        inp = tokenizer.apply_chat_template([{ "role": "user", "content": input_context}], tokenize=True, add_generation_prompt=True, return_tensors='pt')
    else:
        shift = 0
        inp = tokenizer(input_context, return_tensors='pt').input_ids
    emoji_spans = [(k[0] + shift, k[1] + shift) for k in emoji_spans]
    
    if use_emoji:
        print("emoji:")
        for emoji_span, emj in zip(emoji_spans,emojis10):
            print("Original:",tokenizer.decode(emj),emj)
            print("Detected:",tokenizer.decode(inp[0,emoji_span[0]:emoji_span[1]].tolist()),inp[0,emoji_span[0]:emoji_span[1]].tolist())
            print()

    search_pos = find_multi_needle_idx(inp[0], tokenizer, evidence_list[selected_idx])
    attack_pos = find_multi_needle_idx(inp[0], tokenizer, disturb_tok_needles)
    inp = inp.to(model.device)

    
    with torch.no_grad():
        pred_res = tokenizer.decode(model.generate(inp, max_new_tokens=32, do_sample=False)[0, inp.size(-1):])
        logger.info(pred_res)

    logger.info(inp.shape)

    if tokenizer.chat_template is not None:
        inp = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_context}, {"role": "assistant", "content": answer}], 
            tokenize=True, add_generation_prompt=False, return_tensors='pt'
        ).to(model.device)
    else:
        inp = tokenizer(input_context + "\n" + answer, return_tensors='pt').input_ids.to(model.device)
    

    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors='pt')["input_ids"].to(model.device)
    toks_length = answer_ids.size(-1)
    for j in range(inp.size(-1), toks_length, -1):
        if (inp[0, j-toks_length : j] == answer_ids).sum().item() == toks_length:
            target_pos = (j-toks_length, j) 
            break
    else:
        raise ValueError("Not find target in input tokens!")
    
    if args.loss_type == "label":
        label = torch.full(inp.shape, -100).to(model.device)
        for sub_pos in range(*target_pos):
            label[0, sub_pos] = inp[0, sub_pos]

        flow_res = test_model_with_adapter(model, inp, label, search_pos, attack_pos, 
                                                     emoji_spans,
                                                     (target_pos[0] - 1,
                                                      target_pos[1] - 1), 
                                                      is_0k,
                                                      model_name, tokenizer, with_adapter=with_adapter,
                                                      start_layer = start_layer)
    
    elif args.loss_type == "ce":
        flow_res = test_model_with_adapter(model, inp, inp, search_pos, attack_pos, 
                                                     emoji_spans,
                                                     (target_pos[0] - 1,
                                                      target_pos[1] - 1), 
                                                      is_0k,
                                                      model_name, tokenizer, with_adapter=with_adapter,
                                                      start_layer = start_layer)

    flow_res["pred_res"] = pred_res
    flow_res["score"] = 100 if answer.lower() in pred_res.lower() else 0

    logger.info(flow_res)
    auto_save_data(flow_res, f"{args.save_dir}/{save_file_name}.pkl")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected_idx', type=int, default=0, help='selected index')
    parser.add_argument('--needle_path', type=str, default="preliminary/data/reasoning_needle.jsonl",help='path to multi-hop file')
    parser.add_argument('--model_path', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help='path to model')
    parser.add_argument("--adapter_path", type=str, default="", help='path to adapter')
    parser.add_argument('--dataset_path', type=str, default="preliminary/data/pg19-test", help='path to `pg19-test` dataset')
    parser.add_argument('--save_dir', type=str, default="preliminary/results", help='path to dataset')
    parser.add_argument("--tag", type = str, default = "information_flow")
    parser.add_argument("--select-range",type = str, default = "0,200", help="Selected range of samples")
    parser.add_argument("--use-emoji", type = bool, default = True)
    parser.add_argument("--context_lengths", type = str, default = "11900,7900,3900,1900,900", help = 'contexts of lengths that will be tested')

    args = parser.parse_args()

    
    args.save_dir = f"{args.save_dir}/{args.tag}"

    print("Pid:",os.getpid())

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    needles_and_stacks = auto_read_data(args.needle_path)

    l,r = tuple(map(int,args.select_range.split(",")))
    step = 1
    selected_idx = list(range(l, r, step))

    needle_list = [l["needle"] for l in needles_and_stacks]
    retrieval_question_list = [l["question"] for l in needles_and_stacks]
    evidence_list = [l["real_needle"] for l in needles_and_stacks]
    golden_answer_list = [l["golden_answer"] for l in needles_and_stacks]
    tags = [l["tag"] for l in needles_and_stacks]


    for pe,pn in zip(evidence_list, needle_list):
        last_idx = pn.index(pe[-1])
        assert last_idx > -1

        pe += [pn[last_idx + 1]]
    
    
    random.seed(42)
    for context_length in list(map(int,args.context_lengths.split(","))):
        for loss_type in [ "label" ]:
            args.context_length = context_length
            args.loss_type = loss_type
            for s_id in selected_idx:
                logger.info(f"Selected idx: {s_id}")
                logger.info(f"Question: {retrieval_question_list[s_id]}")
                logger.info(f"Answer: {golden_answer_list[s_id]}")
                logger.info(f"Tag: {tags[s_id]}")
                logger.info(f"Needle: {needle_list[s_id]}")
                logger.info(f"Real Needle: {evidence_list[s_id]}")
                logger.info("=============================================")


                needle = [tokenizer(i, add_special_tokens=False)['input_ids'] for i in needle_list[s_id]]
                evidence = [tokenizer(i, add_special_tokens=False)['input_ids'] for i in evidence_list[s_id]]
                question = retrieval_question_list[s_id]
                answer = golden_answer_list[s_id]
                tag = tags[s_id]

                # 初始化采样器
                haystack = datasets.load_dataset(args.dataset_path, split="test")
                if args.context_length>0:
                    noise_sampler_test = SentenceSampler(haystack, tokenizer=tokenizer, shuffle=False, random_seed=42)
                    background_text = noise_sampler_test.get_sample(args.context_length)  
                    disturb_tok_needles = [i for i in needle if i not in evidence]
                    np.random.seed(42)
                    disturb_pos = np.random.choice(len(background_text)+1, len(disturb_tok_needles))
                    print("disturb:",disturb_pos)
                else:
                    background_text = None
                    disturb_tok_needles = [i for i in needle if i not in evidence]
                    disturb_pos = None

                combinations_number = 100
                all_combinations = list(itertools.combinations(list(range(10)), len(evidence)))
                all_combinations = random.sample(all_combinations, combinations_number)
                cnt = 0
                with tqdm(total=len(all_combinations)) as pbar:
                    for _, depth_percent in enumerate(all_combinations):


                        if cnt == 3: break
                        try:
                            model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                                                                        attn_implementation = "flash_attention_2"
                                                                        ).half()

                            device_map = {
                                "model.embed_tokens": 0, "model.rotary_emb" :0, 
                                "model.layers.0" :0, "model.layers.1" :0, "model.layers.2" :0,
                                "model.layers.3" :1, "model.layers.4" :1, "model.layers.5" :1,
                                "model.layers.6" :2, "model.layers.7" :2, "model.layers.8" :2,
                                "model.layers.9" :3, "model.layers.10" :3, "model.layers.11" :3,
                                "model.layers.12" :4, "model.layers.13" :4, "model.layers.14" :4,
                                "model.layers.15" :5, "model.layers.16" :5, "model.layers.17" :5,
                                "model.layers.18" :6, "model.layers.19" :6, "model.layers.20" :6,
                                "model.layers.21" :7, "model.layers.22" :7, "model.layers.23" :7,
                                "model.layers.24" :0, "model.layers.25" :1, "model.layers.26" :2,
                                "model.layers.27" :3, "model.layers.28" :4, "model.layers.29" :5,
                                "model.layers.30" :6, "model.layers.31" :7, "model.norm" :6,
                                "lm_head"  : 7
                                
                            }

                            model = dispatch_model(model, device_map=device_map)

                            pbar.set_description(f"Processing depth {depth_percent}")
                            depth_tag = "-".join([str(i) for i in depth_percent])
                            model_name = args.model_path.split("/")[-1]
                            
                            save_file_name = f"{model_name}/{args.context_length}/{args.loss_type}/{tag}_sid-{s_id}_pid-{cnt}_{depth_tag}"
                            
                            begin_test(args, question, answer, s_id, model, tokenizer, depth_percent, background_text, disturb_pos,disturb_tok_needles, evidence, evidence_list, save_file_name, model_name, args.use_emoji, is_0k = (context_length == 0), with_adapter= True if args.adapter_path else False,                                   start_layer = 24)
                            pbar.update(1)
                            cnt += 1
                            print("dep_p:",depth_percent)
                        except ZeroDivisionError as ze:
                            continue

                        except ValueError as e:
                            if str(e) =="evidence_list and disturb_tok_needles length not match!":
                                continue
                        finally:
                            del model
                            torch.cuda.empty_cache()
                    if cnt != 3:
                        print(f"args.context_length: {args.context_length}")
                        print(f"args.loss_type: {args.loss_type}")
                        print(f"cnt: {cnt}")
                        print(f"s_id: {s_id}")

            file_dir =f"{args.save_dir}/{model_name}/{args.context_length}/{args.loss_type}/"
        print("TESTING OVER:",context_length, loss_type)