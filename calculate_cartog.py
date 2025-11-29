''' Reads in training_data to calculate confidence and variability.'''
#input_file = "fake_log.csv"
#import pandas as pd
#import seaborn as sn
input_file = "cartog/training_dynamics_save.csv"
import csv
#assumes each line is [ex_id,predicted_ans, true_ans, prob, correctness]
# where prob is the joint prob of start & end tokens - call it p(y|x)
n_epochs = 2

#these definitions come from the original cartography paper:
#https://arxiv.org/pdf/2009.10795  see section 2
def confidence(prob_list): #calculated per example
    sum_of_prob = 0
    for e in range(n_epochs):
       sum_of_prob += prob_list[e]
    return sum_of_prob/n_epochs
 
def variability(prob_list, mu): #calculated per example
    sum_of_prob_diff = 0
    for e in range(n_epochs):
       sum_of_prob_diff += (prob_list[e] - mu)**2
    variab = (sum_of_prob_diff/n_epochs)**0.5
    return variab

#create dictionary where key = ex_id, value = prob_list
d = {} #eg, d{2} = [0.24, 0.15, 0.28...] keys = ex_id, values = prob_list
with open(input_file, 'r', encoding = 'utf8') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for row in csv_reader:
        ex_id = int(row[0])
        prob = float(row[3])
        if ex_id not in d:
            d[ex_id] = [prob]
        else:
            d[ex_id].append(prob) 
num_examples = max(d.keys())
print(f"There are {num_examples} examples")
max_points = 50
# for each example id (key), construct a confidence value
# store them in a dictionary
mu_dict = dict()  #key = ex_id, value = mu  (ie confidence)
for ex_id, prob_ls in d.items():
    if ex_id < max_points:
        mu_dict[ex_id] = confidence(prob_ls)
x = []
y = []
for ex_id, mean_val in mu_dict.items():
    variab = variability(d[ex_id], mu_dict[ex_id])
    print(f"Point{ex_id}: {variab}, {mean_val}",d[ex_id] )
    x.append(variab)
    y.append(mean_val)
assert len(x) <= max_points
#data = {'X':x, 'Y':y}
#df = pd.DataFrame(data)
#sn.scatterplot(data=df, x="variability", y="confidence")
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=x,y=y,mode = 'markers',
    connectgaps=False,name="Data Map for Adversarial Training Data") )
fig.update_yaxes(title="confidence")
fig.update_xaxes(title="variability")
fig.show()

