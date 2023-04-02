import Operator
from SubRule import SubRule


class Rule():
    SubRules = []
    targetClass = ""
    classes = []
    sampleCount = 0
    proba = 0.0
    giniImpurity =0.0

    def ToString(self):
        result = ""
        for subRule in self.SubRules:
            result += "("
            result += str(subRule.feature) + " "
            result += str(subRule.operator.value) + " "
            result += str(subRule.threshold) + " "
            #result += "(P = " +str(subRule.proba) + ")"
            #result += "(N = " + str(subRule.sampleCount) + ")"
            #result += "(Classes = " + str(subRule.classes) + ")"
            #result += "(GI = " + str(subRule.giniImpurity) + ")"
            result += ")"
        result += "--> RESULT "
        result += str(self.targetClass)
        result += " with proba = "
        result += str(self.proba)
        result += " sample count = "
        result += str(self.sampleCount)
        result += " GINI IMPURITY = "
        result += str(self.giniImpurity)
        result += "(Classes = " + str(self.classes) + ")"

        return result

    def ToStringSummary(self):
        result = ""
        for subRule in self.SubRules:
            result += "("
            result += str(subRule.feature) + " "
            result += str(subRule.operator.value) + " "
            result += str(subRule.threshold)
            result += ")"
        result += "--> Class "
        result += str(self.targetClass)
        return result

    def isSimilar(self,rule):
        if(self.targetClass == rule.targetClass):
            if(len(self.SubRules) == len(rule.SubRules)):
                for s,r in zip(self.SubRules, rule.SubRules):
                    if(s.feature == r.feature and s.operator.name == r.operator.name and (abs(s.threshold - r.threshold) / s.threshold) == 0):
                        continue
                    else:
                        return False
                return True
            else:
                return False
        else:
            return False

    def isSimilar2(self,rule):
        if(self.targetClass == rule.targetClass):
            if(len(self.SubRules) == len(rule.SubRules)):
                for s,r in zip(self.SubRules, rule.SubRules):
                    if(s.feature == r.feature and s.operator.name == r.operator.name and (abs(s.threshold - r.threshold) / s.threshold) == 0):
                        continue
                    else:
                        return False
                return True
            else:
                return False
        else:
            return False

