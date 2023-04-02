import tkinter
import time
import random
from itertools import groupby
from itertools import count

import pandas as pd
import pickle
import numpy as np
from functools import partial
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Progressbar

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

from DecisionTree import DecisionTree
from SubRule import SubRule
from TempSubRule import TempSubRule
from Operator import Operator
#CONSTANTS
font = ("Calibri",13,"normal")
button_font = ("Calibri",10,"normal")
header_font = ("Calibri",13,"bold")
load_file_base = "C:/Users/Sayit/Desktop/MasterThesisResults/Abalone"

#DEFINITIONS
model_path = ""
train_dataset_path = ""
test_dataset_path = ""
model = None
train_dataset = None
test_dataset = None
model_train_dataset_predictions = None
model_test_predictions = None
texai_test_predictions =  None

target_classes=[]
min_probability_at_leaf=0
min_probability_default_value = 40
min_coverage_at_leaf = 0
min_coverage_default_value = 13
max_gini_impurity_at_leaf = 0
max_gini_impurity_default_value = 0.4
max_tree_count = 200
max_tree_count_default_value = 200
max_sample_column_number = 0
tree_count = 0
decision_trees = []
rule_list = []
rule_set = []
augment_dataset_enabled_default_value = True

def load_file_model():
    global model_path
    model_path = filedialog.askopenfilename(initialdir=load_file_base, title='Select a File')
    if(model_path):
        model_input_button.config(bg='green')
def load_file_train_dataset():
    global train_dataset_path
    train_dataset_path = filedialog.askopenfilename(initialdir=load_file_base, title='Select a File')
    if(train_dataset_path):
        train_dataset_input_button.config(bg='green')
def load_file_test_dataset():
    global test_dataset_path
    test_dataset_path = filedialog.askopenfilename(initialdir=load_file_base, title='Select a File')
    if(test_dataset_path):
        test_dataset_input_button.config(bg='green')
def set_progress_bar_length():
    progress_bar_frame.update_idletasks()
    # set the length of the progress bar to be the same as the width of the bottom frame
    progress_bar.config(length=progress_bar_frame.winfo_width() - 30)
def handle_selection(event):
    global rule_set
    index = rule_set_list_box.curselection()[0]
    selected_obj = rule_set[index]
    print(f"Selected object: {selected_obj.ToString()}")
def initialize_variables():
    global model
    global test_dataset
    global train_dataset
    global max_tree_count
    global min_probability_at_leaf
    global min_coverage_at_leaf
    global max_gini_impurity_at_leaf
    global max_sample_column_number
    global model_train_dataset_predictions
    global target_classes

    train_dataset = pd.DataFrame(pickle.load(open(train_dataset_path, 'rb')))
    test_dataset = pd.DataFrame(pickle.load(open(test_dataset_path, 'rb')))
    model = pickle.load(open(model_path, 'rb'))
    model_train_dataset_predictions = model.predict(train_dataset)
    target_classes = sorted(list(set(model_train_dataset_predictions)))

    max_tree_count = int(max_tree_count_entry.get())
    min_probability_at_leaf = float(min_probability_entry.get())
    min_coverage_at_leaf = float(min_coverage_entry.get())
    max_gini_impurity_at_leaf = float(max_gini_impurity_entry.get())
    max_sample_column_number = int(len(train_dataset.columns)/2)+1
def create_decision_tree():
    global model_train_dataset_predictions
    global target_classes
    sample = train_dataset
    sampleWithSomeColumns = sample.sample(random.choice([i for i in range(2, max_sample_column_number)]), axis=1)
    decisionTreeTemp = DecisionTreeClassifier(criterion="log_loss", random_state=101)
    decisionTreeTemp.fit(sampleWithSomeColumns, model_train_dataset_predictions)
    dt = DecisionTree(decisionTreeTemp, sampleWithSomeColumns.columns.tolist(), target_classes)
    return dt
def create_forest():
    global decision_trees
    for i in range(max_tree_count):
        decision_trees.append(create_decision_tree())
        progress_bar["value"] = len(decision_trees)/max_tree_count*100
        main_screen.update_idletasks()
        time.sleep(0.05)
def check_rule_compliance(rule):
    global model_train_dataset_predictions
    if (rule.proba > min_probability_at_leaf):
        if (rule.classes[np.argmax(rule.classes)] > list(model_train_dataset_predictions).count(
                rule.targetClass) * min_coverage_at_leaf / 100):
            if(rule.giniImpurity<max_gini_impurity_at_leaf):
                return True
            else:
                return False
        else:
            return False
    else:
        return False
def create_rule_list():
    global rule_list
    global model_train_dataset_predictions
    global  decision_trees
    for dt in decision_trees:
        for rule in dt.Rules:
            if(check_rule_compliance(rule)):
                rule_list.append(rule)
def sort_rule_subrules():
    global rule_list
    for rule in rule_list:
        temp_subrules = []
        for subrule in rule.SubRules:
            exist = next((x for x in temp_subrules if subrule.feature == x.feature and subrule.operator.name == x.operator.name),
                         None)
            if (exist == None):
                temp_subrule = TempSubRule()
                temp_subrule.feature = subrule.feature
                temp_subrule.operator = subrule.operator
                temp_subrule.threshold = [subrule.threshold]
                temp_subrules.append(temp_subrule)
            else:
                exist.threshold.append(subrule.threshold)

        rule.SubRules = []
        temp_subrules.sort(key=lambda x: x.feature)
        for t in temp_subrules:
            sub_rule = SubRule()
            sub_rule.feature = t.feature
            sub_rule.operator = t.operator
            if (sub_rule.operator.name == Operator.GREATER_THAN.name):
                sub_rule.threshold = max(t.threshold)
            else:
                sub_rule.threshold = min(t.threshold)
            rule.SubRules.append(sub_rule)

def projection_for_rule_sort(val):
    ruleLength = len(val.SubRules)
    strTemp = str(ruleLength)
    for s in val.SubRules:
        strTemp = strTemp + str(s.feature) + s.operator.name
    strTemp = strTemp + str(val.targetClass)
    return strTemp

def pick_comprehensive_rule(rule_group):
    tempRule = rule_group[0]
    for rule in rule_group:
        if(rule.proba*rule.sampleCount > tempRule.proba * tempRule.sampleCount):
            tempRule = rule
    return tempRule

def create_rule_set():
    global rule_set
    global rule_list
    sort_rule_subrules()
    rule_list_sorted = sorted(rule_list, key=projection_for_rule_sort)
    rule_group_list = [list(it) for k, it in groupby(rule_list_sorted, projection_for_rule_sort)]
    for rule_group in rule_group_list:
        temp_rule = pick_comprehensive_rule(rule_group)
        rule_set.append(temp_rule)
        rule_set_list_box.insert(END, temp_rule.ToString())

def start_process():
    initialize_variables()
    create_forest()
    create_rule_list()
    create_rule_set()

def isRuleSuitableForObservation(rule, sampleDataset):
    for subrule in rule.SubRules:
        if(subrule.operator.name == Operator.LESS_OR_EQUAL.name):
            if(sampleDataset[subrule.feature].values[0] <= subrule.threshold):
                continue
            else:
                return False
        else:
            if(sampleDataset[subrule.feature].values[0] > subrule.threshold):
                continue
            else:
                return False
    return True

def texai_predict(dataset):
    global rule_set
    predictions = []
    for i in range(0,len(dataset)):
        proper_rules = []
        sample = dataset.iloc[[i]]
        for rule in rule_set:
            if (isRuleSuitableForObservation(rule, sampleDataset=sample)):
                proper_rules.append(rule)
        if(len(proper_rules) > 0 ):

            tempLists = [[y for y in proper_rules if y.targetClass == x] for x in target_classes]
            count_dict = {}
            for index, group, c in zip(count(), tempLists, target_classes):
                count_dict[c] = len(group)
            predictions.append(max(count_dict, key=count_dict.get))
        else:
            predictions.append(None)

    return predictions

def plot_visual_results():
    global model_test_predictions
    global texai_test_predictions
    global test_dataset
    model_predictions = model_test_predictions
    texai_predictions = texai_test_predictions
    colors = ["red","blue","green","orange","yellow","gray","black",""]
    X_norm = (test_dataset - test_dataset.min()) / (test_dataset.max() - test_dataset.min())
    lda = LDA(n_components=3)  # 2-dimensional LDA
    lda_transformed = pd.DataFrame(lda.fit_transform(X_norm, model_predictions))

    model_predictions_plot, ax1 = plt.subplots()
    model_predictions_plot.set_size_inches(5, 4)
    color_index=0
    for c in target_classes:
        plt.scatter(lda_transformed[model_predictions == c][0], lda_transformed[model_predictions == c][1],label='Class '+str(c), c=colors[color_index], s=4)
        color_index += 1
    plt.legend(loc=3)
    plt.title("ANN Predictions Plot")

    texai_predictions_plot, ax2 = plt.subplots()
    texai_predictions_plot.set_size_inches(5, 4)
    color_index = 0
    for c in target_classes:
        plt.scatter(lda_transformed[texai_predictions == c][0], lda_transformed[texai_predictions == c][1], label='Class ' + str(c), c=colors[color_index], s=4)
        color_index += 1
    plt.title("TEXAI Predictions Plot")
    plt.legend(loc=3)

    texai_vs_model_result_colors = pd.DataFrame(columns=["ModelPrediction", "Result"])
    for i in range(len(model_predictions)):
        newColorTempDf = pd.DataFrame(columns=texai_vs_model_result_colors.columns)
        newColorTempDf["ModelPrediction"] = [model_predictions[i]]
        if(texai_predictions[i] is None):
            newColorTempDf["Result"] = "#0000FF"
        elif(model_predictions[i] == texai_predictions[i]):
            newColorTempDf["Result"] = "#00FF0050"
        elif(model_predictions[i] != texai_predictions[i]):
            newColorTempDf["Result"] = "#FF0000"

        texai_vs_model_result_colors = pd.concat([texai_vs_model_result_colors, newColorTempDf], ignore_index=True)
        texai_vs_model_result_colors.reset_index()

    texai_vs_ann_plot, ax3 = plt.subplots()
    texai_vs_ann_plot.set_size_inches(5, 4)

    for c in target_classes:
        plt.scatter(lda_transformed[model_predictions==c][0], lda_transformed[model_predictions==c][1], c=texai_vs_model_result_colors[texai_vs_model_result_colors["ModelPrediction"] == c]["Result"],s=4)

    plt.legend(['Matched With NN', 'Wrong Prediction', 'No Prediction'], loc=3)
    plt.title("ANN vs TEXAI Predictions Plot")
    ax = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('#00FF00')
    leg.legendHandles[1].set_color('#FF0000')
    leg.legendHandles[2].set_color('#0000FF')

    visual_plots = Tk()
    visual_plots.title("ANN vs TEXAI Prediction Plots")

    # Create a frame with a scrollbar
    frame = Frame(visual_plots)
    frame.pack(fill="both", expand=True)

    # Create a canvas widget and place it within the frame
    canvas = Canvas(frame)
    canvas.pack(side="left", fill="both", expand=True)

    # Create a scrollbar widget and link it to the canvas
    scrollbar = Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)

    visual_results_frame = Frame(canvas)
    canvas.create_window((0, 0), window=visual_results_frame, anchor="nw")

    model_predictions_frame = LabelFrame(visual_results_frame, border=4, highlightthickness=2)
    model_predictions_frame.grid(row=0, column=0, padx=10, pady=10, sticky="SN")
    model_predictions_canvas = FigureCanvasTkAgg(model_predictions_plot, master=model_predictions_frame)
    model_predictions_canvas.draw()
    model_predictions_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    texai_predictions_frame = LabelFrame(visual_results_frame, border=4, highlightthickness=2)
    texai_predictions_frame.grid(row=0, column=1, padx=10, pady=10, sticky="SN")
    texai_predictions_canvas = FigureCanvasTkAgg(texai_predictions_plot, master=texai_predictions_frame)
    texai_predictions_canvas.draw()
    texai_predictions_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    texai_vs_ann_predictions_frame = LabelFrame(visual_results_frame, border=4, highlightthickness=2)
    texai_vs_ann_predictions_frame.grid(row=1, column=0,columnspan=2, padx=10, pady=10, sticky="SN")
    texai_vs_ann_predictions_canvas = FigureCanvasTkAgg(texai_vs_ann_plot, master=texai_vs_ann_predictions_frame)
    texai_vs_ann_predictions_canvas.draw()
    texai_vs_ann_predictions_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    visual_results_frame.columnconfigure(1, weight=1)
    def configure_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    visual_results_frame.bind("<Configure>", configure_scroll_region)
    mainloop()

def create_rule_set_summary():
    rowInd = 2
    for c in target_classes:
        count = len([rule for rule in rule_set if rule.targetClass == c])
        rule_set_summary = Label(rule_summary_frame, text="Class "+str(c), font=font, justify=LEFT).grid(sticky=W,row=rowInd,column=0,padx=10,pady=0)
        rule_set_summary_value = Label(rule_summary_frame, text=str(count), font=font, justify=RIGHT).grid(sticky=E, row=rowInd, column=1, padx=10, pady=0)
        rowInd+=1
    rule_set_summary = Label(rule_summary_frame, text="Total", font=font, justify=LEFT).grid(sticky=W,row=rowInd,column=0,padx=10, pady=0)
    rule_set_summary_value = Label(rule_summary_frame, text=str(len(rule_set)), font=font, justify=RIGHT).grid(sticky=E,row=rowInd,column=1,padx=10, pady=0)


def evaluation():
    global model
    global model_test_predictions
    global texai_test_predictions
    create_rule_set_summary()

    no_prediction_count = 0
    wrong_prediction_count =0
    correct_prediction_count = 0
    model_test_predictions = model.predict(test_dataset)
    texai_test_predictions = texai_predict(test_dataset)

    for i in range(len(texai_test_predictions)):
        if(texai_test_predictions[i] is None):
            no_prediction_count += 1
        elif(texai_test_predictions[i] != model_test_predictions[i]):
            wrong_prediction_count += 1
        else:
            correct_prediction_count += 1
    total_observation_count_value["text"] = len(test_dataset)
    correct_observation_count_label["text"] = correct_prediction_count
    wrong_observation_count_label["text"] = wrong_prediction_count
    no_prediction_observation_count_label["text"] = no_prediction_count
    match_score_label_value["text"] =  str(round(correct_prediction_count/len(test_dataset),2)*100)+"%"


main_screen = tkinter.Tk()
screen_width = main_screen.winfo_screenwidth()
screen_height = main_screen.winfo_screenheight()
main_screen.title("**Tree Ensemble for eXplainable Artificial Intelligence(TEXAI)**")
main_screen.geometry("{0}x{1}+0+0".format(screen_width, screen_height))

input_frame = LabelFrame(main_screen,width=(int)(screen_width/10*8), height=(int)(screen_height/10*2),border=4, highlightthickness=2)
input_frame.grid(row=0, column=0, padx=10, pady=10,sticky= "SN")
input_frame_title = Label(input_frame, text = "INPUT SECTION", font=header_font, justify=CENTER).grid(row=0, column=0, padx=10, pady=10, columnspan=2)
model_input_label = Label(input_frame, text = "Pick up model", font=font, justify=LEFT).grid(sticky = W, row=1, column=0, padx=10, pady=10)
model_input_button = Button(input_frame, text="Load", font=button_font , command=load_file_model)
model_input_button.grid(row=1,column=1, padx=10, pady=10)
train_dataset_input_label = Label(input_frame, text = "Pick up train dataset", font=font, justify=LEFT).grid(sticky = W, row=2, column=0, padx=10, pady=10)
train_dataset_input_button = Button(input_frame, text="Load", height=1, font=button_font,command=load_file_train_dataset)
train_dataset_input_button.grid(row=2,column=1, padx=10, pady=10)
test_dataset_input_label = Label(input_frame, text = "Pick up test dataset",font=font, justify=LEFT).grid(sticky = W, row=3, column=0, padx=10, pady=10)
test_dataset_input_button = Button(input_frame, text="Load",height=1, font=button_font,command=load_file_test_dataset)
test_dataset_input_button.grid(row=3,column=1, padx=10, pady=10)

tuning_frame = LabelFrame(main_screen,border=4, highlightthickness=2)
tuning_frame.grid(row=0, column=1, padx=10, pady=10, sticky= "SN")
tuning_frame_title = Label(tuning_frame, text = "PARAMETER TUNING",font=header_font,justify=CENTER).grid(row=0,column=0, padx=10, pady=10,columnspan=4)
min_probability_label = Label(tuning_frame, text = "Minimum probability value",font=font,justify=LEFT).grid(sticky = W,row=1,column=0, padx=10, pady=10)
min_probability_entry = Entry(tuning_frame, textvariable=StringVar(value=min_probability_default_value), font=font)
min_probability_entry.grid(row=1,column=1, padx=10, pady=10)

min_coverage_label = Label(tuning_frame, text = "Minimum probability value",font=font,justify=LEFT).grid(sticky = W,row=2,column=0, padx=10, pady=10)
min_coverage_entry = Entry(tuning_frame, textvariable=StringVar(value=min_coverage_default_value), font=font)
min_coverage_entry.grid(row=2,column=1, padx=10, pady=10)

max_gini_impurity_label = Label(tuning_frame, text = "Maximum gini impurity value",font=font,justify=LEFT).grid(sticky = W,row=3,column=0, padx=10, pady=10)
max_gini_impurity_entry = Entry(tuning_frame, textvariable=StringVar(value=max_gini_impurity_default_value), font=font)
max_gini_impurity_entry.grid(row=3,column=1, padx=10, pady=10)

max_tree_count_label = Label(tuning_frame, text = "Maximum tree count",font=font,justify=LEFT).grid(sticky = W,row=1,column=2, padx=10, pady=10)
max_tree_count_entry = Entry(tuning_frame, textvariable=StringVar(value=max_tree_count_default_value), font=font)
max_tree_count_entry.grid(row=1,column=3, padx=10, pady=10)

augment_dataset_enabled = BooleanVar(value=augment_dataset_enabled_default_value)
dataset_augmentation_enabled_label = Label(tuning_frame, text = "Augment train dataset",font=font,justify=LEFT).grid(sticky = W,row=2,column=2, padx=10, pady=10)
dataset_augmentation_enabled_check_button = Checkbutton(tuning_frame,variable = augment_dataset_enabled)
dataset_augmentation_enabled_check_button.grid(row=2,column=3, padx=10, pady=10)

progress_bar_frame = LabelFrame(main_screen,border=4, highlightthickness=2)
progress_bar_frame.grid(row=1, column=0, padx=10, pady=10,columnspan=3,sticky=EW)
progress_bar = Progressbar(progress_bar_frame,length=progress_bar_frame.winfo_width(), orient="horizontal", mode="determinate")
progress_bar.grid(row=0, column=0, padx=10, pady=10)

start_process_frame = LabelFrame(main_screen,border=4, highlightthickness=2)
start_process_frame.grid(row=0, column=2, padx=10, pady=10,columnspan=1,sticky=S+E+W+N)
start_process_button = Button(start_process_frame, text="Start", font=header_font,command=start_process)
start_process_button.grid(row=0,column=0, padx=10, pady=40, sticky=S+E+W+N)
result_process_button = Button(start_process_frame, text="Evaluation", font=header_font,command=evaluation)
result_process_button.grid(row=1,column=0, padx=10, pady=40, sticky=S+E+W+N)

rule_set_frame = Frame(main_screen,border=4,highlightthickness=2)
rule_set_frame.grid(row=2, column=0, padx=10, pady=10, columnspan=3, rowspan=5, sticky="NSEW")
rule_set_list_box = Listbox(rule_set_frame)
rule_set_list_box.grid(row=0, column=0, sticky="NSEW")
scrollbar_vertical = Scrollbar(rule_set_frame, orient="vertical")
scrollbar_vertical.grid(row=0, column=1, sticky="NS")
scrollbar_horizontal = Scrollbar(rule_set_frame, orient="horizontal")
scrollbar_horizontal.grid(row=1, column=0,columnspan=2, sticky="EW")
rule_set_list_box.config(yscrollcommand=scrollbar_vertical.set, xscrollcommand=scrollbar_horizontal.set)
rule_set_list_box.bind("<<ListboxSelect>>", handle_selection)
scrollbar_vertical.config(command=rule_set_list_box.yview)
scrollbar_horizontal.config(command=rule_set_list_box.xview)
rule_set_frame.grid_rowconfigure(0, weight=1)
rule_set_frame.grid_columnconfigure(0, weight=1)
main_screen.rowconfigure(2, weight=1)

evaluation_frame = LabelFrame(main_screen,border=4, highlightthickness=2)
evaluation_frame.grid(row=0, column=3, padx=10, pady=10,rowspan=3, sticky=S+E+W+N)
evaluation_frame_title = Label(evaluation_frame, text = "EVALUATION",font=header_font,justify=CENTER).grid(row=0,column=0, padx=10, pady=10,columnspan=2)

rule_summary_frame = LabelFrame(evaluation_frame,border=4, highlightthickness=2)
rule_summary_frame.grid(row=1, column=0, padx=10, pady=10, sticky=S+E+W+N)
rule_summary_frame_title = Label(rule_summary_frame, text = "RULE SET SUMMARY",font=header_font,justify=CENTER).grid(row=0,column=0, padx=10, pady=10,columnspan=2)
rule_summary_frame_title_1 = Label(rule_summary_frame, text = "Target Class",font=header_font,justify=CENTER).grid(row=1,column=0,sticky="E")
rule_summary_frame_title_2 = Label(rule_summary_frame, text = "Rule Count",font=header_font,justify=CENTER).grid(row=1,column=1,sticky="E")

test_frame = LabelFrame(evaluation_frame,border=4, highlightthickness=2)
test_frame.grid(row=2, column=0, padx=10, pady=10, sticky=S+E+W+N)
test_frame_title = Label(test_frame, text = "TEST RESULT",font=header_font,justify=CENTER).grid(row=0,column=0, padx=10, pady=10,columnspan=2)

total_observation = Label(test_frame, text = "Total observation Count",font=font,justify=LEFT).grid(sticky=W,row=1,column=0, padx=10, pady=0)
total_observation_count_value = Label(test_frame, text = "0",font=font,justify=RIGHT)
total_observation_count_value.grid(sticky=E,row=1,column=1, padx=10, pady=0)

correct_observation_label = Label(test_frame, text = "Correct prediction Count",font=font,justify=LEFT).grid(sticky=W,row=2,column=0, padx=10, pady=0,columnspan=1)
correct_observation_count_label = Label(test_frame, text = "0",font=font,justify=RIGHT)
correct_observation_count_label.grid(sticky=E,row=2,column=1, padx=10, pady=0,columnspan=1)

wrong_observation_label = Label(test_frame, text = "Wrong prediction Count",font=font,justify=LEFT).grid(sticky=W,row=3,column=0, padx=10, pady=0,columnspan=1)
wrong_observation_count_label = Label(test_frame, text = "0",font=font,justify=RIGHT)
wrong_observation_count_label.grid(sticky=E, row=3,column=1, padx=10, pady=0,columnspan=1)

no_prediction_observation_label = Label(test_frame, text = "No prediction Count",font=font,justify=LEFT).grid(sticky=W,row=4,column=0, padx=10, pady=0,columnspan=1)
no_prediction_observation_count_label = Label(test_frame, text = "0",font=font,justify=RIGHT)
no_prediction_observation_count_label.grid(sticky=E, row=4,column=1, padx=10, pady=0,columnspan=1)

match_score_label = Label(test_frame, text = "Match Score",font=font,justify=LEFT).grid(sticky=W,row=5,column=0, padx=10, pady=0,columnspan=1)
match_score_label_value = Label(test_frame, text = "0",font=font,justify=RIGHT)
match_score_label_value.grid(sticky=E, row=5,column=1, padx=10, pady=0,columnspan=1)


visualizations_frame = LabelFrame(evaluation_frame,border=4, highlightthickness=2)
visualizations_frame.grid(row=3, column=0, padx=10, pady=10,rowspan=3, sticky=S+E+W+N)
visualizations_frame_title = Label(visualizations_frame, text = "VISUALIZATION",font=header_font,justify=CENTER).grid(sticky=E+W+S+N,row=0,column=0, padx=10, pady=10,columnspan=2)


plot_visual_results_button = Button(visualizations_frame, text="Predictions Plot", font=header_font,anchor=CENTER,command=plot_visual_results)
plot_visual_results_button.grid(row=1,column=0, padx=10, pady=40,columnspan=2,sticky=E+W)





visualizations_frame.columnconfigure(1, weight=1)
rule_summary_frame.columnconfigure(1,weight=1)
test_frame.columnconfigure(1,weight=1)
evaluation_frame.columnconfigure(0,weight=1)
main_screen.columnconfigure(3, weight=1)
main_screen.columnconfigure(3, weight=1)
main_screen.after(0, set_progress_bar_length)
main_screen.mainloop()
