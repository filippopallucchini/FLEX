from datasets import load_dataset
import numpy as np
import torch
import os
import fitz #pdf reader
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import pandas as pd
import pickle
from statistics import mean

def remove_quotes(sentence):
    # Strip leading and trailing quotation marks
    if sentence.startswith('"') and sentence.endswith('"'):
        sentence = sentence[1:-1]
    return sentence

def extract_text_from_pdf(pdf_path):
    text = ""

    try:
        with fitz.open(pdf_path) as pdf_document:
            num_pages = pdf_document.page_count

            for page_num in range(num_pages):
                page = pdf_document[page_num]
                text += page.get_text()

    except Exception as e:
        print(f"Error: {e}")

    return text

def enrichment_candidates(list_input_sentences, model_decoder_only, num_candidates, temp):
    
    all_candidates = []
    prompts = ["As an expert in finance, you are tasked with enriching the original sentence, making it more clear and explicit, ensuring these specifics: "
            + "1. DO NOT CHANGE subjects of the original sentence"
            + "2. AVOID factual inaccuracies, logical errors and paradoxes"
            + "3. DO NOT CHANGE the SENTIMENT of the orginal sentence"
            + "4. DO NOT expand the scope of the original sentence and DO NOT ADD unverifiable information"
            + "Exactly as in this example:"
            + "original sentence: \"Mid caps now turn into market darlings\""
            + "sentence enriched: \"Mid caps now turn into market darlings, meaning they are highly favored and popular among investors, analysts, and traders\""
            + "So, act as an expert in finance: could you enrich the original sentence '" + input_sentence + "'"
            + "Please report only and only the enriched sentence, as follows: "
            + "\"sentence enriched\"."
            + "Please report only and only the enriched sentence, enclosed within inverted commas."
            + "DO NOT PROVIDE FURTHER INFORMATION DESPITE THE ENRICHED SENTENCE ENCLOSED WITHIN INVERTED COMMAS." for input_sentence in list_input_sentences]

    mean_num_tokens = mean([len(i) for i in list_input_sentences])
    sampling_params = SamplingParams(temperature=temp, top_p=0.95, max_tokens=(mean_num_tokens*len(list_input_sentences)*(num_candidates*1.3)))
    outputs = model_decoder_only.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        response = output.outputs[0].text
        try:
            enriched_sentences = re.findall('"([^"]*)"', response)
            if len(enriched_sentences) < (num_candidates*0.2):
                response = response.replace('\n', '')
                enriched_sentences = re.split(r'(?<=[a-zA-Z\s])\.(?=[a-zA-Z0-9\s])', response)
                if len(enriched_sentences) > (num_candidates*1.5):
                    enriched_sentences_temp = [f'{enriched_sentences[i]} {enriched_sentences[i+1]}' for i in range(0, len(enriched_sentences)-1) if enriched_sentences[i][0].isdigit() == True]
                    enriched_sentences = enriched_sentences_temp
                    
        except:
            response = response.replace('\n', '')
            enriched_sentences = re.split(r'(?<=[a-zA-Z\s])\.(?=[a-zA-Z0-9\s])', response)
            if len(enriched_sentences) > (num_candidates*1.5):
                enriched_sentences_temp = [f'{enriched_sentences[i]} {enriched_sentences[i+1]}' for i in range(0, len(enriched_sentences)-1) if enriched_sentences[i][0].isdigit() == True]
                enriched_sentences = enriched_sentences_temp
            if len(enriched_sentences) < (num_candidates*0.2):
                print(f'error!!! check enriched_sentences: {enriched_sentences}')
                print(f'error!!! check response: {response}')
        all_candidates.append(enriched_sentences)
        
    return all_candidates, outputs

def main():
    #define path
    path = 'your_path'

    #DECODER ONLY
    torch.cuda.empty_cache()
    #model_name_decoder_only = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    model_name_decoder_only = "thesven/Mistral-7B-Instruct-v0.3-GPTQ"
    model_decoder_only = LLM(model=model_name_decoder_only, quantization="gptq", dtype="half")

    #ENRICH SENTENCES

    #evaluation on https://huggingface.co/datasets/ChanceFocus/fiqa-sentiment-classification
    dataset_origin = load_dataset("ChanceFocus/fiqa-sentiment-classification")
    dataset_sentences = []
    dataset_labels = []
    for i in dataset_origin['train']:
        if i['score']>0.3:
            dataset_sentences.append(i['sentence'])
            dataset_labels.append(2)
        elif i['score']<-0.3:
            dataset_sentences.append(i['sentence'])
            dataset_labels.append(0)
            
    for i in dataset_origin['valid']:
        if i['score']>0.3:
            dataset_sentences.append(i['sentence'])
            dataset_labels.append(2)
        elif i['score']<-0.3:
            dataset_sentences.append(i['sentence'])
            dataset_labels.append(0)
            
    for i in dataset_origin['test']:
        if i['score']>0.3:
            dataset_sentences.append(i['sentence'])
            dataset_labels.append(2)
        elif i['score']<-0.3:
            dataset_sentences.append(i['sentence'])
            dataset_labels.append(0)

    count_enrichment = 0
    count_no_enrichment_to_diff = 0
    X = []

    #save also sentences enriched and how they where enriched
    list_sentence_enriched = []

    #SET PARAMS

    #tot candidates enriched sentences generated by decoder-only
    #zeta
    num_candidates = 1
        
    #check if possible candidates have been already created
    check_presence_candidates = False
    os.chdir(f'{path}')
    directory = os.fsencode('DATA')
        
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if f'file_{num_candidates}_candidates_for_enrichment_fiqa_ablation_no_makeup.pkl' == filename:
            check_presence_candidates = True
            break
        
    if check_presence_candidates == False:
        list_input_sentences = []
        for index, input in tqdm(enumerate(dataset_sentences)):
            list_input_sentences.append(input)
        all_candidates, outputs = enrichment_candidates(list_input_sentences, model_decoder_only, num_candidates, 0)
        list_sentence_to_fix = []
        index_to_modify = []
        for index, i in enumerate(all_candidates):
            if len(i) > (num_candidates*1.5) or len(i) < (num_candidates*0.3):
                list_sentence_to_fix.append(list_input_sentences[index])
                index_to_modify.append(index)
        print(f"try to do again {len(list_sentence_to_fix)} wrong output")
        try:
            all_candidates_fixed, outputs = enrichment_candidates(list_sentence_to_fix, model_decoder_only, num_candidates, 1)  
            #update candidates 
            for index, i in enumerate(index_to_modify):
                if len(all_candidates_fixed[index]) < 1:
                    continue
                else:
                    all_candidates[i] = all_candidates_fixed[index]
        except:
            print('no need new trial')

        list_input_and_sentences_candidates = []
        for i in range(0, len(list_input_sentences)):
            input_sentence = list_input_sentences[i]
            candidates = all_candidates[i]
            list_input_and_sentences_candidates.append((input_sentence, candidates))

        #save possible enriched sentences
        # create a binary pickle file 
        f = open(f"{path}/DATA/file_{num_candidates}_candidates_for_enrichment_fiqa_ablation_no_makeup.pkl","wb")
        # write the python object (dict) to pickle file
        pickle.dump(list_input_and_sentences_candidates,f)
        # close file
        f.close()
    else:
        list_input_and_sentences_candidates = pickle.load(open(f"{path}/DATA/file_{num_candidates}_candidates_for_enrichment_fiqa_ablation_no_makeup.pkl", 'rb'))
        print(f'candidates computed yet')
        
    list_final_enriched = []
    for input_sentence, candidates in tqdm(list_input_and_sentences_candidates):
        if len(candidates) == 0:
            list_final_enriched.append((input_sentence, candidates, False, input_sentence)) 
        else:
            list_final_enriched.append((input_sentence, candidates, True, candidates[0]))
        
    list_sentence_original = []
    for input_sentence, candidates, enriched, sentence_enriched in list_final_enriched:
        if enriched == False:
            count_no_enrichment_to_diff += 1
        else:   
            count_enrichment += 1
            list_sentence_original.append(input_sentence)
            sentence_enriched = remove_quotes(sentence_enriched)
            list_sentence_enriched.append(sentence_enriched)
            
    #SAVE NEW DATASET
    data_sentences_enriched = {'original_sentence': list_sentence_original,
            'sentence': list_sentence_enriched, 
            }

    # Create DataFrame
    output_sentences_enriched = pd.DataFrame(data_sentences_enriched)
    output_sentences_enriched = output_sentences_enriched.drop_duplicates()
    output_sentences_enriched = output_sentences_enriched.drop_duplicates(subset=['original_sentence'])

    fiqa_enriched = {'original_sentence': dataset_sentences,
            'label': dataset_labels
            }
    output_fiqa_enriched = pd.DataFrame(fiqa_enriched)
    output_fiqa_enriched_temp = output_fiqa_enriched.merge(output_sentences_enriched, on = 'original_sentence', how = 'left')
    output_fiqa_enriched_temp.sentence.fillna(output_fiqa_enriched_temp.original_sentence, inplace=True)
    output_fiqa_enriched = output_fiqa_enriched_temp[['sentence', 'label']]
    output_fiqa_enriched.to_csv(f'{path}/OUTPUT/fiqa_enriched_ablation_no_makeup.csv')

if __name__ == '__main__':
    main()
