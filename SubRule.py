from Operator import Operator


class SubRule():
    feature = ""
    operator = Operator.NONE
    threshold = None
    targetClass = None
    classes = []
    proba = 0.0
    sampleCount = 0
    giniImpurity = 0.0
