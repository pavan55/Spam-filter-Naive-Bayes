
import math
import argparse

def calc_accuracy(actual_labels, predicted_labels):
    correct = 0
    spam_correct, spam_predicted, spam_total = 0, 0, 0

    for act, pred in zip(actual_labels, predicted_labels):
        if act == 'spam':
            spam_total +=1
        if pred == 'spam':
            spam_predicted +=1
        if act == pred:
            correct +=1
            if pred == 'spam':
                spam_correct +=1
    accuracy = float(correct)/len(actual_labels)
    precision_spam = float(spam_correct)/float(spam_predicted)
    recall_spam = float(spam_correct)/float(spam_total)
    f1_score = 2 * precision_spam * recall_spam/(precision_spam+recall_spam)
    print('For spam class precision={:.2f}, recall={:.2f}, f1-score={:.2f}'.format(precision_spam, recall_spam, f1_score))
    print 'Overall Accuracy=', float(accuracy) * 100, '%'

def test(test_file, output_file, conditional_prob, classes_count, dictionary):
    test_data_fp = open(test_file, 'r')
    op_fp = open(output_file, 'w')
    actual_labels = []
    predicted_labels = []
    for line in test_data_fp:
        word_tokens = line.split(" ")
        id = word_tokens[0]
        actual_class_type = word_tokens[1]
        actual_labels.append(actual_class_type)
        word_tokens = word_tokens[2:]
        prob_ham = 0
        prob_spam = 0
        for index in range(0, len(word_tokens), 2):
            freq = word_tokens[index + 1]
            word = word_tokens[index]
            if word not in dictionary:
                prob_ham += math.log10(1 / float(classes_count['ham'] + len(dictionary)))
                prob_spam += math.log10(1 / float(classes_count['spam'] + len(dictionary)))
            else:
                if word in conditional_prob['ham']:
                    prob_ham += float(freq) * math.log10(conditional_prob['ham'][word])
                if word in conditional_prob['spam']:
                    prob_spam += float(freq) * math.log10(conditional_prob['spam'][word])
        if prob_ham>prob_spam:
            predicted_labels.append('ham')
            op_fp.write(id + ' ' + 'ham' + '\n')
        else:
            predicted_labels.append('spam')
            op_fp.write(id + ' ' + 'spam' + '\n')

    calc_accuracy(actual_labels, predicted_labels)
    op_fp.close()
    test_data_fp.close()

def BayesBinomialClassifier(train_file):
    #create word set
    dictionary = set()
    train_data_fp = open(train_file,'r')#fp.read()
    for line in train_data_fp:
        word_tokens = line.split(" ")[2:]
        word_tokens = [word for (index, word) in enumerate(word_tokens) if index % 2 == 0]
        for word in word_tokens:
            dictionary.add(word)
    train_data_fp.close()
    total_unique_words = 0
    classes_count = {'spam':0, 'ham':0}
    word_counts = {'spam':{}, 'ham':{}}
    word_count_aggr = {'spam':0, 'ham':0}

    train_data_fp = open(train_file, 'r')
    for line in train_data_fp:
        word_tokens = line.split(" ")
        type = word_tokens[1]
        classes_count[type] = classes_count[type] + 1
        word_tokens = word_tokens[2:]
        for index in range(0, len(word_tokens), 2):
            freq = word_tokens[index+1]
            try:
                freq = int(freq)
            except:
                print('Frequency of a word',word_tokens[index],'not a integer')

            word = word_tokens[index]
            if word not in word_counts[type]:
                word_counts[type][word] = freq#1
            else:
                word_counts[type][word] = word_counts[type][word] + freq
            word_count_aggr[type] += freq

    prior_prob_classes = {'spam': classes_count['spam']/(classes_count['spam']+classes_count['ham']), 'ham':classes_count['ham']/(classes_count['spam']+classes_count['ham'])}
    conditional_prob = {'spam':{}, 'ham':{}}
    smoothing_param = 100
    for class_label, attr in word_counts.items():
        for word, count in attr.items():
            conditional_prob[class_label].setdefault(word,0)
            conditional_prob[class_label][word] = float(count+ smoothing_param)/float(word_count_aggr[class_label] + smoothing_param * len(dictionary))
            #print word, count, classes_count[class_label], len(dictionary), conditional_prob[class_label][word]
    train_data_fp.close()
    return conditional_prob, dictionary, classes_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', help='training file in csv format', required=True)
    parser.add_argument('-f2', help='test file in csv format', required=True)
    parser.add_argument('-o', help='output labels for the test dataset', required=True)
    args = vars(parser.parse_args())

    train_file, test_file, output_file = args['f1'], args['f2'], args['o']
    conditional_prob, dictionary, classes_count = BayesBinomialClassifier(train_file)
    test(test_file, output_file, conditional_prob, classes_count, dictionary)

