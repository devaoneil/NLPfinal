''' Augmentation in comparison questions (x or y?)'''

import random
import regex as re


def find_answer_in_context(context, answer_text):
    """Find the character position of answer in context - written by Claude.ai"""
    # Convert to lowercase for matching if needed
    start = context.find(answer_text)
    if start == -1:
        # Try case-insensitive
        start = context.lower().find(answer_text.lower())
        if start != -1:
            # Return position but keep original casing
            return start
    return start if start != -1 else 0  # Default to 0 if not found
  
def get_new_answer(answer, choices):
    '''code from:
    https://github.com/dDua/contrastive-estimation/blob/main/data/utils.py
    Authors: Learning with Instance Bundles for Reading Comprehension
    Dheeru Dua et al
    '''
    print("answer in get_new_answer:", answer)
    print("Choices: ", choices)
    choices = [c.strip() for c in choices]
    new_answer = list(set(choices).difference(set([answer])))
    #select from choices whichever one isn't answer
    if len(new_answer) > 1:
        print("multiple choices left")
        return None
    new_answer = new_answer[0].strip()
    return new_answer

    '''code from:
    https://github.com/dDua/contrastive-estimation/blob/main/data/utils.py
    Authors: Learning with Instance Bundles for Reading Comprehension
    Dheeru Dua et al
    '''
'''def extract_answer_choices(question, answer):

    new_answer, choice1, choice2 = None, None, None
    regex1 = re.search(r'(,.+ or )', question)
    regex2 = re.search(r'( or .+?[,?])', question)
    #original: regex1, regex2 = re.search('(\,.+ or )', question),re.search('( or .+?[,|\,])', question)
    if regex1 and regex2 and regex2.start() > regex1.start():
        choice1 = question[regex1.start(): regex1.end()].split(" or ")[0].replace(",", "").strip()
        choice2 = question[regex2.start(): regex2.end()].split(" or ")[-1].replace(",", "").strip()
        new_answer = get_new_answer(answer, [choice1, choice2])
        return new_answer, choice1, choice2

    regex1 = re.search('(,.+ or )', question)
    regex2 = re.search('( or .+?[,|?])', question)
    if regex1 and regex2 and regex2.start() > regex1.start():
        choice1 = question[regex1.start(): regex1.end()].split(" or ")[0].replace(",", "").strip()
        choice2 = question[regex2.start(): regex2.end()].split(" or ")[-1].replace("?", "").strip()
        new_answer = get_new_answer(answer, [choice1, choice2])

    regex1, regex2 = re.search('(:.+ or )', question), re.search('( or .+)', question)
    if regex1 and regex2 and regex2.start() > regex1.start():
        choice1 = question[regex1.start(): regex1.end()].split(" or ")[0].replace(":", "").strip()
        choice2 = question[regex2.start(): regex2.end()].split(" or ")[-1].replace("?", "").strip()
        # choice1 = question[regex1.start(): regex1.end()].rstrip(" or").replace(":", "").strip()
        # choice2 = question[regex2.start(): regex2.end()].lstrip("or ").replace("?", "").strip()
        new_answer = get_new_answer(answer, [choice1, choice2])

    return new_answer, choice1, choice2
'''

def extract_option_text(question):
    """
    Finds 'or' in a question, goes backwards to find a comma or colon,
    and returns the text in the form 'A or B?'
    Returns the extracted text 'A or B' or None if pattern not found. 
    Helper function for extract_answer_choices
    """
    # Find the position of ' or ' (with spaces to avoid matching words containing 'or')
    or_match = re.search(' or ', question)
    if not or_match:
        print("Error: no 'or' found")
        return None
    # Get the text before ' or '
    text_before_or = question[:or_match.start()] #this line by Claude.ai
    if ':' in text_before_or:
        return extract_after_colon(question)
    else: #assume comma-separated
        return extract_after_comma(question)

'''helper function for extract_option_text(question)'''
def extract_after_comma(question):
    if ', or ' not in question:  #"Which gun was less common, T9 AA cannon or the Bofors 40 mm?"
        split_or = question.split(' or ')
        if ',' in split_or[-1]: #"Which location is closer to Bermuda, Martha's Vineyard or Miami, Florida?"
            separator = split_or[0].rfind(',')
            extracted = question[separator + 1:]
        else:
            separator = question.rfind(',')
            extracted = question[separator + 1:]
    else:
        split_or = question.split(', or ')
        #"Which is greater, the number of survivors from Yingxiu, or 10,000?"
        if ',' in split_or[-1]:
            splits = question.split(',')
             
            extracted =  splits[-3] + splits[-2] +  splits[-1]

        else:
            #"What weapon system had more course corrections, the Avenger, or the RIM-116?"
            splits = question.split(',')
            extracted =  splits[-2] +  splits[-1]
    print("Extracted text (should be A or B):", extracted.strip().rstrip('?'))
    return extracted.strip().rstrip('?')

'''helper function for extract_option_text(question)'''
def extract_after_colon(question):  
    last_colon = question.rfind(':') 
    extracted = question[last_colon + 1:].strip().rstrip('?')
    return extracted

def extract_answer_choices(question, answer):
    extracted = extract_option_text(question) #format:  "A or B" (no question mark)
    
    options = extracted.split(' or ')
    choice1, choice2 = options[0], options[1]
    if answer.strip() == options[0].strip(): new_answer = options[1]
    elif answer.strip() == options[1].strip(): new_answer = options[0]
    else:
        if answer in choice1: new_answer = options[1]
        elif answer in choice2: new_answer = options[0]
        else:
            if True in [i in choice1 for i in answer.split(',') ]: new_answer = options[1]
            else: new_answer = options[0]

            print("No matching answer. answer = ", answer, " | options: ", options)
    return new_answer, choice1, choice2

'''The following is written by me based on the code from Dheeru Dua et al'''
#augment a single qa pair by returning its flipped version
def pairwise_augmentation(pair): #randomizes punctuation after question stem and order of choices
    question, answer, id1 = pair
    other_answer, choice1, choice2 = extract_answer_choices(question, answer)
    comp_options = [[" more ", " less "], [" better", " worse"], [" more ", " fewer "],
                    [" higher", " lower"],[" faster", " slower"], [" bigger", " smaller"],
                    [" easier", " harder"], [" larger", " smaller"], [" decreased", " increased"],
                    [" decreased", " accelerated"], [" had ", " didn't have "],
                    [" had ", " did not have "],
                    [" more likely", " less likely"], [" has ", " doesn't have "],
                    [" should ", " shouldn't "],[" earlier"," later"],
                    [" has ", " does not have "],[" always ", " not always "], [" longer", " shorter"]]
    #if " or " in question:
    comp_options_match = [(c[0] in question or c[1] in question) for c in comp_options]
    #print(comp_options_match)
    match = False
    for boolean in comp_options_match:
        if boolean: match = True
    if match is False: return None
        #print(comp_options_match)
    for i in range(len(comp_options)):
        if comp_options_match[i]:
            options = comp_options[i]  #select desired comparison question by flipping
            #print("Options are: ", options)
            if re.search(" fewer ", question):
                new_question = question.replace(" fewer ", " more ") #special case - 'more' appears twice in options list
                print("%%replacing ", " fewer ", "with", " more ", "in ", question)
            elif  options[0] in question:
                print("%%replacing ", options[0], "with", options[1], "in ", question)
                new_question = question.replace(options[0], options[1])
            elif  options[1] in question:
                print("%%replacing ", options[1], "with", options[0], "in ", question)
                new_question = question.replace(options[1], options[0])
            else:
                print("Something is wrong, neither option in question")
            break
    if random.random() < 0.5:  #choose random order
        choices = [answer, other_answer]
    else:
        choices = [other_answer, answer]
                 
    new_question = question_truncate(new_question)
    punctuation = random.choice([",", ":"])
    aug_question1 = new_question+  punctuation
            #print("choices[0] = ", choices[0])
            #print("choices[1] = ", choices[1])
    aug_question1 += " " + choices[1]+ " or " + choices[0] + "?"
    augment_tuple = (aug_question1, other_answer, str(id1) + "_aug")
    print("returning: ", augment_tuple)
    return augment_tuple
    
def question_truncate(question): #removes answer choices from comparative questions
    if re.search(':',question):
        idx_separator = question.find(':')
    elif re.search(',',question):
        idx_separator = question.find(',')
    else: print("No separator found (: or ,)")
    return question[:idx_separator]

def qualifies_for_augmentation(ex):
    
    q = ex["question"]
    pattern = r'.+[,:]\s*.+\s+or\s+.+' #regex by Claude.ai
        #return false if there are multiple ' or '
    if re.search(r'.+\sor\s.+\sor\s.+',q): return False
    
        #return false if there are multiple ':',
        #eg Which of the following is NOT referenced by the EU's specification: the USB Battery Charging standard or IEC 62684:2011?
    if len(q.split(':'))>2: return False
    
        # Checks for comparison questions like
        #"Which is A: Y or Z?" or "What is A, X or Y?"  
    if re.search(pattern, q):
        if binary_question(q): return True#narrow down to binary questions only - see the following helper fxn

    #print("%%%%%%%%q = ", q)       
    return False

''' Helper function for qualifies_for_augmentation
    Return False if like this:
    Which of the following is not part of the Central Lancashire New Town:
    Leyland, Burnley, or Chorley?"
'''
def binary_question(q):
    
    #check for non-binary pattern
    if "lower 48" in q: #this question is going to mess up the replacement of lower->higher
        print("Found weird question")
        return False #"When referencing directions, what can be said about Alaska: westernmost or lower 48?"

    pattern =r":(?:\s*)([^,]+),\s*([^,]+?)(?:,?\s*or\s*)([^?]+)\?"  #regex by Claude.ai
    found_pattern = re.search(pattern, q)
    splits = q.split(',')
    #reject if trinary or if false binary (What do..., such as lifting a cup or walking?)
    if found_pattern or " such as " in q:
        #print("Non-binary Pattern: ", q)
        return False
    
    elif len(splits )>= 3 and ' or ' in splits[-1]:
        #print("WEIRD Non-binary Pattern: ", q)
        return False #"Which location is farther away from Bermuda, Miami, Florida or Cape Sable Island?"

    else: 
        print("Found binary pattern: ",q)
        return True
    return False

def help_create(example):
    try:
        q = example['question']
    except:
        print("NO question", flush = True)
    try:
        answer = example['answers']['text'][0]
        if answer not in q:
            return {"id": None} #not actually a comparison question
    except:
        print("NO ans", flush = True)
        return {"id": None}
    try:
        qa_pairs = (q, answer, example['id'])
    except:
        print("NO Q/A Pair (No example['id'])", flush = True)
        return {"id": None}
    #don't need this; already in run.py  if not qualifies_for_augmentation(example): return {"id": None} #throws error if you try returning None due to map function in run.py

    try:
        #if pairwise_augmentation(qa_pairs) is None: return None
        aug_q, aug_a, aug_id  = pairwise_augmentation(qa_pairs)
     
        print("\nNew question:",aug_q)
        print("original question:",example['question'])
        print("New Answer:",aug_a, "original answer:",answer)

        #written by Claude
        aug_example = { 
        "id": aug_id,
        "context": example["context"],
        "question": aug_q,
        "answers": {  
            "text": [aug_a],
            "answer_start": [find_answer_in_context(example["context"], aug_a)]
    }
}
        return aug_example
    except:
        print("\nbad example: ", qa_pairs, flush = True)
        return {"id": None}
    
def create_augmented_example(example):
    try:
        return help_create(example)
    except:
        print("Unable to create")
        return {"id": None}
    
def main():
    #generate two sample examples; both are comparison-type
    sample_passage = "Today, Warsaw has some of the best medical facilities in Poland and East-Central Europe. The city is home to the Children's Memorial Health Institute (CMHI), the highest-reference hospital in all of Poland, as well as an active research and education center. While the Maria Skłodowska-Curie Institute of Oncology it is one of the largest and most modern oncological institutions in Europe. The clinical section is located in a 10-floor building with 700 beds, 10 operating theatres, an intensive care unit, several diagnostic departments as well as an outpatient clinic. The infrastructure has developed a lot over the past years." 

    original_data = [{
    "id": 1,
    "context": sample_passage,
    "question": "Which is there fewer of, floors or beds?",
    "answers": [
        {
            "text": ["floors"],"answer_start": [find_answer_in_context(sample_passage, "floors")]
        }
    ]
    },{
    "id": 2,
    "context": sample_passage,
    "question": "Which is there more of, beds or operating theatres?",
    "answers": [
        {
            "text": ["beds"],
            "answer_start": [find_answer_in_context(sample_passage, "beds")]
        }
    ]
    }]
 
    for example in original_data: #construct tuples
        print(create_augmented_example(example))
# Generate augmentations
if __name__ == "__main__":
    main()
        
