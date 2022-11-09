import numpy
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from scipy.stats import norm

# Project for AI class
# Tec de Monterrey
# Benjamín Ávila Rosas
# November, 2022.

plt.style.use('_mpl-gallery')
column_names = ['class', 'value']
classes_resume = {
    1: {},
    2: {},
    3: {}
}


# Get the CSV file to use the stored data
def get_csv_with_data():
    data_csv = None

    data_csv = pd.read_csv("naive_bayes_work.csv", names=column_names, skiprows=1)

    return data_csv


# Classify items on CSV based on the class they belong
def separate_rows_per_class_and_sort():
    class_one_list = []
    class_two_list = []
    class_tree_list = []

    for _, row in data_from_csv.iterrows():
        if row.values[0] == 1:
            class_one_list.append(row[1])
        if row.values[0] == 2:
            class_two_list.append(row[1])
        if row.values[0] == 3:
            class_tree_list.append(row[1])

    class_one_list.sort()
    class_two_list.sort()
    class_tree_list.sort()

    classes_resume[1]['list'] = class_one_list
    classes_resume[1]['size'] = len(class_one_list)
    classes_resume[1]['priori'] = numpy.divide(len(class_one_list), 1000)

    classes_resume[2]['list'] = class_two_list
    classes_resume[2]['size'] = len(class_two_list)
    classes_resume[2]['priori'] = numpy.divide(len(class_two_list), 1000)

    classes_resume[3]['list'] = class_tree_list
    classes_resume[3]['size'] = len(class_tree_list)
    classes_resume[3]['priori'] = numpy.divide(len(class_tree_list), 1000)


# Calculate the median for all items of a class
def get_median_for_class(items_list):
    median = statistics.median(items_list)

    return median


# Calculate the standard deviation for all items of a class
def get_standard_deviation_for_class(items_list):
    st_dev = statistics.stdev(items_list)

    return st_dev


# Do the statistics calculations required for the items separated by class
def get_statistics_per_class():
    for item_index in classes_resume.keys():
        classes_resume[item_index]['median'] = get_median_for_class(classes_resume[item_index]['list'])
        classes_resume[item_index]['std_dev'] = get_standard_deviation_for_class(classes_resume[item_index]['list'])


# Do the statistics calculations required for the items separated by class
def print_statistics_per_class():
    for item_index in classes_resume.keys():
        print("---- CLASS " + str(item_index) + " STATS ----")
        print("Priori: " + str(classes_resume[item_index]['priori']))
        print("Median: " + str(classes_resume[item_index]['median']))
        print("Standard Deviation: " + str(classes_resume[item_index]['std_dev']))
        print()


# Use median and standard deviation to create a normal distribution for each class data
def get_normal_distribution_for_classes():
    class_one_norm = norm(classes_resume[1]['median'], classes_resume[1]['std_dev'])
    class_two_norm = norm(classes_resume[2]['median'], classes_resume[2]['std_dev'])
    class_tree_norm = norm(classes_resume[3]['median'], classes_resume[3]['std_dev'])

    classes_resume[1]['norm'] = class_one_norm
    classes_resume[2]['norm'] = class_two_norm
    classes_resume[3]['norm'] = class_tree_norm


def pdf_probability_plot():
    class_one_norm = classes_resume[1]['norm']
    class_two_norm = classes_resume[2]['norm']
    class_tree_norm = classes_resume[3]['norm']

    x_min_value = min(class_one_norm.ppf(0.0001), class_two_norm.ppf(0.0001), class_tree_norm.ppf(0.0001))
    x_max_value = max(class_one_norm.ppf(0.9999), class_two_norm.ppf(0.9999), class_tree_norm.ppf(0.9999))

    x = np.linspace(x_min_value, x_max_value, 1000)

    x1, y1 = x, class_one_norm.pdf(x)
    x2, y2 = x, class_two_norm.pdf(x)
    x3, y3 = x, class_tree_norm.pdf(x)

    fig, ax = plt.subplots(figsize=(10, 5), clear=True)
    ax.plot(x1, y1, 'b-', lw=4, alpha=0.6, label='Clase 1')
    ax.plot(x2, y2, 'r-', lw=4, alpha=0.6, label='Clase 2')
    ax.plot(x3, y3, 'g-', lw=4, alpha=0.6, label='Clase 3')
    # ax.plot(0.1, 0.1, marker='o')
    ax.legend()
    plt.show()


data_from_csv = get_csv_with_data()
csv_size = len(data_from_csv)
separate_rows_per_class_and_sort()
get_statistics_per_class()

get_normal_distribution_for_classes()
pdf_probability_plot()

print_statistics_per_class()




