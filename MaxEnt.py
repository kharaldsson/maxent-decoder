import re
import numpy as np
from collections import Counter, OrderedDict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


# Superclass
class Classifier:
    def __init__(self):
        self.train_raw = None
        self.test_raw = None
        self.model_lines = None
        self.n_classes = None
        self.n_docs = None
        # self.n_docs_in_class = None
        self.vocab = set()  # Set
        self.vocab_size = None  # int
        self.n_docs = None
        # self.classes = None
        self.vocab2idx = {}
        self.class2idx = {}
        self.idx2vocab = {}
        self.idx2class = {}
        self.neighbors = None
        self.y_hat_train = None
        self.y_hat_train_probs = None
        self.y_hat_test = None
        self.y_hat_test_probs = None

    @staticmethod
    def create_array(rows, columns):
        array_out = np.zeros((rows, columns))
        return array_out

    @staticmethod
    def prep_documents(data, type):
        data_clean = [re.sub('\n', '', x) for x in data]
        data_clean = [x for x in data_clean if x]
        data_clean = [re.split(r"\s+", x) for x in data_clean]

        # Split X and y
        y_str = [x[0] for x in data_clean]
        X_str = [x[1:] for x in data_clean]

        if type == 'count':
            X_str = [[sl for sl in l if sl] for l in X_str]
            X_str = [[tuple(re.split(r":", sl)) for sl in l] for l in X_str]
            X_str = [dict(l) for l in X_str]
            X_str = [dict((k, int(v)) for k, v in subdict.items()) for subdict in X_str]
        else:
            X_str = [[re.split(r":", sl)[0] for sl in l] for l in X_str]
            X_str = [[sl for sl in l if sl] for l in X_str]

        return X_str, y_str

    def confusion_matrix(self, y_actual, y_predicted):
        conf_matrix = np.zeros((len(self.idx2class), len(self.idx2class)))
        for actual, pred in zip(y_actual, y_predicted):
           conf_matrix[actual, pred] += 1
        return conf_matrix

    def get_acc(self, y_actual, y_predicted):
        actual = np.array(y_actual)
        pred = np.array(y_predicted)
        correct = (actual == pred)
        accuracy = correct.sum() / correct.size
        return accuracy

    def classification_report(self, test_only=None):
        output_lines = []
        test_matrix = self.confusion_matrix(self.y_test, self.y_hat_test)
        test_acc = self.get_acc(self.y_test, self.y_hat_test)

        class_labels = list(self.idx2class.values())

        class_labels_join = ' '.join(class_labels)
        class_labels_join = "\t\t" + class_labels_join

        if self.y_hat_train is not None:
            train_header_line = ['Confusion matrix for the training data:',
                                 'row is the truth, column is the system output',
                                 '\n']
            output_lines.append(train_header_line)

            train_matrix = self.confusion_matrix(self.y_train, self.y_hat_train)

            train_acc = self.get_acc(self.y_train, self.y_hat_train)

            output_lines.append(class_labels_join)

            for key, value in self.idx2class.items():
                matrix_counts = train_matrix[key, :].tolist()
                matrix_counts = [str(int(x)) for x in matrix_counts]
                matrix_counts = ' '.join(matrix_counts)
                matrix_line = str(value) + ' ' + matrix_counts
                output_lines.append(matrix_line)

            output_lines.append('\n')
            output_lines.append("Training accuracy=" + str(train_acc))
            output_lines.append('\n')

        second_title = ['Confusion matrix for the test data:', 'row is the truth, column is the system output',
                        '\n']
        output_lines += second_title
        output_lines.append(class_labels_join)

        for key, value in self.idx2class.items():
            matrix_counts = test_matrix[key, :].tolist()
            matrix_counts = [str(int(x)) for x in matrix_counts]
            matrix_counts = ' '.join(matrix_counts)
            matrix_line = str(value) + ' ' + matrix_counts
            output_lines.append(matrix_line)

        output_lines.append('\n')
        output_lines.append("Test accuracy=" + str(test_acc))

        for line in output_lines:
            print(line)

    def format_output_lines(self, predictions, set_header, line_header, actuals):
        all_lines = [set_header]
        for doc_idx, pred_list in enumerate(predictions):

            normed = pred_list
            line = []
            line_dict = {}
            l_header = line_header + str(doc_idx)
            l_actual = actuals[doc_idx]
            l_actual = self.idx2class[l_actual]
            line.append(l_header)
            line.append(l_actual)
            for class_idx, class_name in self.idx2class.items():
                line_dict[class_name] = normed[class_idx]
            line_dict = OrderedDict(line_dict)
            for class_name, prob in sorted(line_dict.items(), key=lambda item: item[1], reverse=True):
                line_item = str(class_name) + " " + str(prob)
                line.append(line_item)

            line_string = ' '.join(line)
            all_lines.append(line_string)

        return all_lines

    def save_sys_output(self, sys_output_dir, test_only=None):
        """
        Write predictions to file
        """
        output_lines = []
        train_header = "%%%%% training data:"
        test_header = "%%%%% test data:"
        line_header = "array:"
        if self.y_hat_train is not None:
            y_hat_tr = self.y_hat_train_probs
            # output_lines.append(train_header)
            train_lines = self.format_output_lines(y_hat_tr, train_header, line_header, self.y_train)
            output_lines += train_lines
            output_lines.append('\n')

        if self.y_hat_test is not None:
            y_hat_ts = self.y_hat_test_probs

            # output_lines.append(test_header)
            test_lines = self.format_output_lines(y_hat_ts, test_header, line_header, self.y_test)
            output_lines += test_lines

        with open(sys_output_dir, 'w', encoding='utf8') as f:
            f.writelines("%s\n" % line for line in output_lines)


# Sublcass
class MaxEntClassifier(Classifier):
    def __init__(self):
        self.X_train = None  # np array
        self.y_train = None  # np array
        self.y_train_M = None
        self.X_test = None  # np array
        self.y_test = None  # np array
        self.class_counts_raw = None
        self.emp_exp = None
        self.class_unigram_counts = None
        self.class_token_weights = None
        self.prob_y_given_x_i = None
        self.model_exp = None
        self.model_exp_counts = None
        self.class_defaults = None
        self.class_keys = None
        super().__init__()

        # self.process_train()
        # self.process_test()

    def load_model(self, model_lines):
        data_clean = [re.sub('\n', '', x) for x in model_lines]

        # Get Class Indices
        class_locs = set([i for i, x in enumerate(data_clean) if "FEATURES FOR CLASS" in x])
        class_names = [x for x in data_clean if "FEATURES FOR CLASS" in x]
        class_names = [re.sub(r'FEATURES FOR CLASS ', '', x) for x in class_names]
        self.n_classes = len(class_names)
        self.class2idx = {k: v for v, k in enumerate(class_names)}
        self.idx2class = {k: v for k, v in enumerate(class_names)}

        data_clean = [l.split(',') for l in ','.join(data_clean).split('FEATURES FOR CLASS ')]
        data_clean = data_clean[1:]
        data_clean = [x[1:] for x in data_clean]
        data_clean = [[item.strip() for item in sl] for sl in data_clean]
        data_clean = [[re.split(r"\s+", item) for item in sl if item] for sl in data_clean]

        default_weights = [sl[0][1] for sl in data_clean]
        self.class_defaults = np.array(default_weights, dtype=np.float)
        data_clean = [sl[1:] for sl in data_clean]

        # Set Vocab
        vocabulary = [[item[0] for item in sl] for sl in data_clean]
        vocabulary = [word for class_ls in vocabulary for word in class_ls if word]
        self.vocab = set(vocabulary)
        # self.vocab.remove('<default>')
        self.vocab_size = len(self.vocab)
        vocabulary = list(self.vocab)
        vocabulary.sort()

        # Create vocabulary decoding dicts
        self.vocab2idx = {k: v for v, k in enumerate(vocabulary)}
        self.idx2vocab = {k: v for k, v in enumerate(vocabulary)}

        self.class_token_weights = self.create_array(self.n_classes, self.vocab_size)


        for class_idx, word_pair in enumerate(data_clean):
            for pair in word_pair:
                word = pair[0]
                word_idx = self.vocab2idx[word]
                weight = float(pair[1])
                self.class_token_weights[class_idx, word_idx] = weight


    def process_train(self):
        X_tr, y_tr = self.prep_documents(self.train_raw, type='binary')

        if self.class_token_weights is None:
            # Set class information
            classes = list(dict.fromkeys(y_tr))  # set(y_tr)
            self.n_classes = len(classes)
            self.class2idx = {k: v for v, k in enumerate(classes)}
            self.idx2class = {k: v for k, v in enumerate(classes)}

            # Set vocabulary
            self.vocab = set([word for doc in X_tr for word in doc])
            self.vocab_size = len(self.vocab)
            vocabulary = list(self.vocab)
            vocabulary.sort()

            # Create vocabulary decoding dicts
            self.vocab2idx = {k: v for v, k in enumerate(vocabulary)}
            self.idx2vocab = {k: v for k, v in enumerate(vocabulary)}

        self.y_train = np.array([self.class2idx[c] for c in y_tr])
        class_count = dict(Counter(y_tr))
        self.n_docs_in_class = {self.class2idx[k]: v for k, v in class_count.items()}

        # Set number of docs
        self.n_docs = len(y_tr)

        # Set Y Matrix
        self.y_train_M = self.create_array(self.n_classes, self.n_docs)

        for doc_idx, class_idx in enumerate(self.y_train):
            self.y_train_M[class_idx, doc_idx] = 1

        X_array = [[self.vocab2idx[word] for word in doc] for doc in X_tr]

        self.X_train = self.create_array(self.n_docs, self.vocab_size)

        for doc_idx, doc in enumerate(X_array):
            for word in doc:
                self.X_train[doc_idx, word] = 1

        self.calc_emp_exp()

    def calc_emp_exp(self):
        self.class_counts_raw = np.dot(self.y_train_M, self.X_train)
        self.class_counts_raw = self.class_counts_raw.astype(int)
        self.emp_exp = np.divide(self.class_counts_raw, self.n_docs)

    def calc_model_exp(self):
        if self.class_token_weights is not None:
            self.prob_y_given_x_i = np.stack(self.y_hat_train_probs, axis=1)
        else:
            cond_prob_shape = (self.n_classes, self.n_docs)
            self.prob_y_given_x_i = np.empty(cond_prob_shape, dtype=np.float)
            uniform_dist = 1 / self.n_classes
            self.prob_y_given_x_i.fill(uniform_dist)

        self.model_exp = np.divide(self.prob_y_given_x_i, self.n_docs)
        self.model_exp = np.dot(self.model_exp, self.X_train)
        self.model_exp_counts = self.model_exp * self.n_docs

    def process_test(self):
        X_ts, y_ts = self.prep_documents(self.test_raw, type='binary')

        # Set class information
        self.y_test = np.array([self.class2idx[c] for c in y_ts if c in self.class2idx])

        # Set number of docs
        n_test_docs = len(y_ts)

        X_array = [[self.vocab2idx[word] for word in doc if word in self.vocab2idx] for doc in X_ts]
        self.X_test = self.create_array(n_test_docs, self.vocab_size)

        for doc_idx, doc in enumerate(X_array):
            for word in doc:
                self.X_test[doc_idx, word] = 1

    def predict(self, instances, save):
        y_pred = np.zeros(np.shape(instances)[0]).astype(int)
        y_probs = []

        for doc_idx, doc in enumerate(instances):
            c_pred, c_probs = self.predict_proba(instances[doc_idx, :])
            y_pred[doc_idx] = c_pred
            y_probs.append(c_probs)

        if save == 'test':
            self.y_hat_test = y_pred
            self.y_hat_test_probs = y_probs
        elif save == 'train':
            self.y_hat_train = y_pred
            self.y_hat_train_probs = y_probs

        return y_pred, y_probs

    def predict_proba(self, instance):
        lambdas = np.multiply(instance, self.class_token_weights)
        lambdas_summed = np.sum(lambdas, axis=1, keepdims=True).flatten()
        lambdas_default = np.add(lambdas_summed, self.class_defaults)
        lambdas_exp = np.exp(lambdas_default)
        Z = np.sum(lambdas_exp)
        probs = np.divide(lambdas_exp, Z)
        pred = np.argmax(probs)

        return pred, probs

    def save_emp_exp(self, output_file_path):
        output_lines = []
        for class_idx, class_name in self.idx2class.items():
            classXfeat_exp = self.emp_exp[class_idx, :]
            classXfeat_ct = self.class_counts_raw[class_idx, :]
            for feat_idx, feat_name in self.idx2vocab.items():
                feat_exp = classXfeat_exp[feat_idx]
                feat_ct = classXfeat_ct[feat_idx]
                line_str = class_name + " " + feat_name + " " + str(feat_exp) + " " + str(feat_ct)
                output_lines.append(line_str)

        with open(output_file_path, 'w', encoding='utf8') as f:
            f.writelines("%s\n" % line for line in output_lines)

    def save_exp(self, output_file_path, exp_type=None):
        output_lines = []

        if exp_type == 'empirical':
            exp_array = self.emp_ex
            class_counts = self.class_counts_raw
        else:
            exp_array = self.model_exp
            class_counts = self.model_exp_counts

        for class_idx, class_name in self.idx2class.items():
            classXfeat_exp = exp_array[class_idx, :]
            classXfeat_ct = class_counts[class_idx, :]
            for feat_idx, feat_name in self.idx2vocab.items():
                feat_exp = classXfeat_exp[feat_idx]
                feat_ct = classXfeat_ct[feat_idx]
                line_str = class_name + " " + feat_name + " " + str(feat_exp) + " " + str(feat_ct)
                output_lines.append(line_str)

        with open(output_file_path, 'w', encoding='utf8') as f:
            f.writelines("%s\n" % line for line in output_lines)