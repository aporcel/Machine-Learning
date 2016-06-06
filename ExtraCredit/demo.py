### modified from https://github.com/mbilalzafar/fair-classification
### based on the paper "Learning Fair Classifiers," http://arxiv.org/pdf/1507.05259v3.pdf

import os,sys
import numpy as np
import utils as ut
import loss_funcs as lf 

class fairness_demo(object):

    def train(self, X, y, x_sensitive, fairness_constraint):
        self.x_sensitive = {"s1": x_sensitive}
        self.X = ut.add_intercept(X)
        self.y = y

        if fairness_constraint==-1.0:
            self.w = ut.train_model(self.X, self.y, self.x_sensitive, lf._logistic_loss, 0, 0, 0, 
                                    ["s1"], {"s1":0}, None)
        else:
            self.w = ut.train_model(self.X, self.y, self.x_sensitive, lf._logistic_loss, 1, 0, 0, 
                                    ["s1"], {"s1": fairness_constraint}, None)

        train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(self.w, 
                                                                                   self.X, self.y, self.X, self.y, None, None)

        distances_boundary_test = (np.dot(self.X, self.w)).tolist()
        all_class_labels_assigned_test = np.sign(distances_boundary_test)
        correlation_dict_test = ut.get_correlations(None, None, 
                                    all_class_labels_assigned_test, self.x_sensitive, ["s1"])

        correlation_dict = ut.get_avg_correlation_dict([correlation_dict_test])
        non_prot_pos = correlation_dict["s1"][1][1]
        prot_pos = correlation_dict["s1"][0][1]
        p_rule = (prot_pos / non_prot_pos) * 100.0

	return self.w, p_rule, 100.0*test_score

    def plot(self, plt):
        
        X_s_0 = self.X[self.x_sensitive["s1"] == 0.0]
        X_s_1 = self.X[self.x_sensitive["s1"] == 1.0]
        y_s_0 = self.y[self.x_sensitive["s1"] == 0.0]
        y_s_1 = self.y[self.x_sensitive["s1"] == 1.0]
        plt.scatter(X_s_0[y_s_0==1.0][:, 1], X_s_0[y_s_0==1.0][:, 2], color='green', marker='x', s=30, linewidth=1.5)
        plt.scatter(X_s_0[y_s_0==-1.0][:, 1], X_s_0[y_s_0==-1.0][:, 2], color='red', marker='x', s=30, linewidth=1.5)
        plt.scatter(X_s_1[y_s_1==1.0][:, 1], X_s_1[y_s_1==1.0][:, 2], color='green', marker='o', facecolors='none', s=30)
        plt.scatter(X_s_1[y_s_1==-1.0][:, 1], X_s_1[y_s_1==-1.0][:, 2], color='red', marker='o', facecolors='none', s=30)
    
        x1, x2 = max(self.X[:,1]), min(self.X[:,1])
        y1, y2 = ut.get_line_coordinates(self.w, x1, x2)
        plt.plot([x1,x2], [y1,y2], 'k--', linewidth=3)
