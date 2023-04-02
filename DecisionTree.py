import numpy as np
import sklearn.tree
from sklearn.tree import _tree

from Operator import Operator
from Rule import Rule
from SubRule import SubRule


class DecisionTree(sklearn.tree.DecisionTreeClassifier):
    def __init__(self, decisionTreeClassifier,featureNames, classNames):
        self.deicisonTreeClassifier = decisionTreeClassifier
        self.featureNames = featureNames
        self.classNames = classNames
        self.Rules = self.get_rules(self.deicisonTreeClassifier, self.featureNames, self.classNames);

    def get_rules(self, tree, feature_names, class_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        paths = []
        path = []

        rule = Rule()
        rules = []
        subRules = []

        def recurse(node, subRules, rules):
            rule = Rule()
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                p1, p2 = list(subRules), list(subRules)
                name = feature_name[node]
                threshold = tree_.threshold[node]
                subRule = SubRule()
                subRule.feature = name
                subRule.operator = Operator.LESS_OR_EQUAL
                subRule.threshold = np.round(threshold, 5)
                classes = tree_.value[node][0]
                subRule.classes = classes
                l = np.argmax(classes)
                subRule.targetClass = class_names[l]
                subRule.sampleCount = tree_.n_node_samples[node]
                subRule.proba = np.round(100.0 * classes[l] / np.sum(classes), 2)
                subRule.giniImpurity = self.calculateNodeGiniImpurity(classes)
                p1.append(subRule)
                recurse(tree_.children_left[node], p1, rules)
                subRule = SubRule()
                subRule.feature = name
                subRule.operator = Operator.GREATER_THAN
                subRule.threshold = np.round(threshold,5)
                classes = tree_.value[node][0]
                subRule.classes = classes
                l = np.argmax(classes)
                subRule.targetClass = class_names[l]
                subRule.sampleCount = tree_.n_node_samples[node]
                subRule.proba = np.round(100.0 * classes[l] / np.sum(classes), 2)
                subRule.giniImpurity = self.calculateNodeGiniImpurity(classes)
                p2.append(subRule)
                recurse(tree_.children_right[node], p2, rules)
            else:
                classes = tree_.value[node][0]
                l = np.argmax(classes)
                rule.targetClass = class_names[l]
                rule.SubRules = subRules
                rule.sampleCount = tree_.n_node_samples[node]
                rule.proba = np.round(100.0 * classes[l] / np.sum(classes), 2)
                rule.giniImpurity = self.calculateNodeGiniImpurity(classes)
                rule.classes = classes
                rules.append(rule)

        recurse(0, subRules, rules)
        # sort by samples count
        """
        samples_count = [p.sampleCount for p in rules]
        ii = list(np.argsort(samples_count))
        rules = [rules[i] for i in reversed(ii)]
        """


        return rules

    def calculateNodeGiniImpurity(self, targetClassCountArray):
        totalSampleCount = sum(targetClassCountArray)

        giniImpurity= 1.0
        for count in targetClassCountArray:
            giniImpurity = giniImpurity - (count**2/totalSampleCount**2)

        return np.round(giniImpurity,2)















