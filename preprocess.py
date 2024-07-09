import json
import os
import numpy as np

# Final Entry:
    # {
    #     "question": "What is the date when the circular was issued?",
    #     "document": "/data/circulars/DATA/LayoutLM/docvqa_dataset/Images/img_fp_random_selected_2667.png",
    #     "answer": "Dated 7.12.2020"
    # }

def preprocess(data_dir, output_dir):
    pass

def main():
    count = 0

    data_dir = './Dataset/jsons'
    jsons = ['first_page.json', 'middle_page.json', 'last_page.json', 'first_page_2.json']
    # jsons = ['test_set.json']

    final_dataset = []
    
    for json_file in jsons:
        with open(os.path.join(data_dir, json_file)) as f:
            train_data = json.load(f)

        print(train_data[0][0]['file_name'])

        temp_array = []

        for data in train_data:
            hindi_flag = 0
            temp_array = []
            for entry in data[0]["question_answer_pairs"]:
                temp = {}
                temp["document"] = f"/data/circulars/DATA/pix2struct_synth/Dataset/Split_Images/{json_file.split('.')[0]}/{data[0]['file_name']}"
                temp["question"] = entry["question"]
                temp["answer"] = entry["answer"]
                # If the question, or answer is in Hindi, then set the flag
                if any(ord(char) > 128 for char in entry["question"]):
                    hindi_flag = 1
                    print("Hindi Question: ", entry["question"])
                if any(ord(char) > 128 for char in entry["answer"]):
                    hindi_flag = 1
                    print("Hindi Answer: ", entry["answer"])
                # If the answer contains, "claude"  
                if "'model': 'claude-3-haiku" not in temp["answer"]: 
                    temp_array.append(temp)
            if hindi_flag == 0:
                # Only unique questions and answer pairs
                question_set = set()
                answer_set = set()
                unique_array = []
                for entry in temp_array:
                    if entry["question"] not in question_set and entry["answer"] not in answer_set:
                        question_set.add(entry["question"])
                        answer_set.add(entry["answer"])
                        unique_array.append(entry)
                final_dataset.extend(unique_array)
                
    print("Number of Hindi Entries: ", count)
    print("Final Dataset Length: ", len(final_dataset))

    with open('final_dataset.json', 'w') as f:
        json.dump(final_dataset, f)

if __name__ == '__main__':
    main()
    