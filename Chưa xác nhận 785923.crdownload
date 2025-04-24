import csv
import math
from collections import Counter
import random
import heapq


class Node:
    def __init__(self, is_leaf=False, attributeIndex=None, classValue=None, size=0):
        """
        Node class for Tree. Include leaf node and non-leaf node

        Attributes:
        ----------
        is_leaf(boolean): The title of the book.
        attributeIndex(int): index of attribute for the non-leaf node
        classValue(str): the value assign for leaf node (yes/no)
        size(str): number of instance from training data in leaf node
        children(list of tuple (branch, Node)): list of children of non-leaf node
        parent(Node): parent
        yesValidation(int): number of "yes" instance from validation data go through non-leaf node
        noValidation(int): number of "no" instance from validation data go through non-leaf node
        correctValidation(int): number of correct instance from validation data go through leaf node
        """
        self.is_leaf = is_leaf
        self.attributeIndex = attributeIndex
        self.classValue = classValue
        self.size = size
        self.children = []
        self.parent = None
        self.yesValidation = 0
        self.noValidation = 0
        self.correctValidation = 0

    def __lt__(self, other):
        return self.size < other.size

    def addChild(self, child, value):
        self.children.append([child, value])

class DT:
    def __init__(self):
        """
        DT class

        Attributes:
        ----------
        root(Node): root node of the tree
        candidate(min-heap list of tuple): list of (gain correction if prunning, potential candidate node) 
        """
        self.root = Node()
        self.candidate = []

    def same_class(self, training_data):
        """
        Check if all the data has the same class
        """
        if(len(training_data) == 0):
            return False
        row1 = training_data[0]
        for row in training_data:
            if(row[-1] != row1[-1]):
                return False
        return True
    
    def same_attribute(self, training_data):
        """
        Check if all the data has the same attributes
        """
        if(len(training_data) == 0):
            return False
        row1 = training_data[0]
        for row in training_data:
            if(row[:-1] != row1[:-1]):
                return False
        return True
        
    def entropy(self, labels):
        """
        Calculate the entropy
        """
        total = len(labels)
        counts = Counter(labels)
        ent = 0.0
        for label in counts:
            p = counts[label] / total
            if p > 0:
                ent -= p * math.log2(p)
        return ent
    
    def split_information(self, data, attributeIndex):
        """
        Calculate the split_information
        """
        total_len = len(data)
        if total_len == 0:
            return 0

        dic = {
            'low': 0,
            'medium': 0,
            'high': 0,
            'very high': 0
        }
        for row in data:
            dic[row[attributeIndex]] += 1

        split_info = 0.0
        for key in dic.keys():
            p = dic[key] / total_len
            if p > 0:
                split_info -= p * math.log2(p)

        return split_info

    def build(self, node, training_data, parent_majority="yes"):
        """
        Build the tree
        """

        # Calculating the majority class
        yes = 0
        no = 0
        for row in training_data:
            if(row[-1] == "yes"):
                yes = yes + 1
            else:
                no = no + 1
        majority = "yes"
        if(yes < no):
            majority = "no"
        
        # Check the current node is leaf or not. If yes, stop the recursion
        if(self.same_class(training_data) == True):
            node.is_leaf = True
            node.classValue = training_data[0][-1]
            node.size = len(training_data)
        elif(self.same_attribute((training_data)) == True):
            node.is_leaf = True
            node.classValue = majority
            node.size = len(training_data)
        elif(len(training_data) == 0):
            node.is_leaf = True
            node.classValue = parent_majority
            node.size = len(training_data)

        if(node.is_leaf == True):
            return
        
        node.is_leaf = False
        
        classes = []
        for row in training_data:
            classes.append(row[-1])
        E = self.entropy(classes)

        # Calculating the Gain Ratio amongs all attribute and c
        attribute_choose = -1
        attribute_G = 0
        for i in range(len(training_data[0]) - 1):
            row1 = training_data[0]
            flag = True
            for row in training_data:
                if(row[i] != row1[i]):
                    flag = False
                    break
            if(flag == True):
                continue

            T = 0
            dic = {
                'low': [],
                'medium': [],
                'high': [],
                'very high': []
            }
            for row in training_data:
                dic[row[i]].append(row[-1])
            
            for key in dic.keys():
                T = T + self.entropy(dic[key]) * (len(dic[key]) / len(training_data))
            gain = T - E
            gain_ratio = gain / self.split_information(training_data, i)
            if(attribute_choose == -1):
                attribute_choose = i
                attribute_G = gain_ratio
            elif(gain_ratio < attribute_G):
                attribute_choose = i
                attribute_G = gain_ratio

        # Split the data and continue build the children node
        dic = {
            'low': [],
            'medium': [],
            'high': [],
            'very high': []
        }
        for row in training_data:
            dic[row[attribute_choose]].append(row)

        node.attributeIndex = attribute_choose
        for key in dic.keys():
            newNode = Node()
            self.build(node=newNode, training_data=dic[key], parent_majority=majority)
            newNode.parent = node
            node.addChild(newNode, key)
        
        # Check if this node would be candidate or not
        is_candidate = True
        for child in node.children:
            if(child[0].is_leaf == False):
                is_candidate = False
        if(is_candidate == True):
            self.candidate.append((0, node))

    def get_class(self, instance, usingValidation = False):
        """
        Calculate the class for given instance.
        If we are using validation data, use it to calculate yesValidation, 
        noValidation and correctValidation
        """
        node = self.root
        while(node.is_leaf == False):
            if(usingValidation == True):
                if(instance[-1] == "yes"):
                    node.yesValidation += 1
                else:
                    node.noValidation += 1
            for child in node.children:
                if child[1] == instance[node.attributeIndex]:
                    node = child[0]
                    break
                        
        if(usingValidation == True):
            if(instance[-1] == node.classValue):
                node.correctValidation += 1
        return node.classValue

    def prune(self, validation_data):
        """
        Prunning the tree (Subtree Replacement)
        """

        for row in validation_data:
            self.get_class(row, usingValidation=True)

        # Calculate the gain for candidates
        new_candidate = []
        for k in self.candidate:
            node = k[1]
            old_correct = 0
            majority = "yes"
            yes = 0
            no = 0
            for child in node.children:
                if(child[0].classValue == "yes"):
                    yes += child[0].size
                else:
                    no += child[0].size
                old_correct += child[0].correctValidation
            if(no > yes):
                majority = "no"

            new_correct = 0
            if(majority == "yes"):
                new_correct = node.yesValidation
            else:
                new_correct = node.noValidation
            gain = new_correct - old_correct
            heapq.heappush(new_candidate, (-gain,node))
        
        self.candidate = new_candidate

        # Prunning iteration
        while True:
            # If there is no candidate, stop prunning
            if(len(self.candidate) == 0):
                break

            gain, node = heapq.heappop(self.candidate)

            correct = 0
            for row in validation_data:
                if(self.get_class(row, usingValidation=False) == row[-1]):
                    correct += 1
            if(-gain < 0):
                break

            # Get the parent node of deleted node
            parent_node = node.parent
            if(parent_node == None):
                return
            
            # Replace with new leaf
            majority = "yes"
            yes = 0
            no = 0
            size = 0
            for child in node.children:
                size += child[0].size
                if(child[0].classValue == "yes"):
                    yes += child[0].size
                else:
                    no += child[0].size
                
            if(no > yes):
                majority = "no"
            new_node = Node(is_leaf=True,classValue=majority,size=size)
            if(majority == "yes"):
                new_node.correctValidation = node.yesValidation
            else:
                new_node.correctValidation = node.noValidation

            denote = 0
            for i in range(len(parent_node.children)):
                child = parent_node.children[i]
                if(child[0] == node):
                    denote = i
            parent_node.children[denote] = (new_node, parent_node.children[denote][1])
            new_node.parent = parent_node

            # Consider parent is a candidate or not
            is_candidate = True
            for child in parent_node.children:
                if(child[0].is_leaf == False):
                    is_candidate = False
            if(is_candidate == True):
                old_correct = 0
                majority = "yes"
                yes = 0
                no = 0
                for child in parent_node.children:
                    if(child[0].classValue == "yes"):
                        yes += child[0].size
                    else:
                        no += child[0].size
                    old_correct += child[0].correctValidation
                if(no > yes):
                    majority = "no"

                new_correct = 0
                if(majority == "yes"):
                    new_correct = parent_node.yesValidation
                else:
                    new_correct = parent_node.noValidation

                gain = new_correct - old_correct
                if(gain >= 0):
                    heapq.heappush(self.candidate, (-gain,parent_node))

            correct = 0
            for row in validation_data:
                if(self.get_class(row, usingValidation=False) == row[-1]):
                    correct += 1

        return

    def print_tree(self, node, depth):
        """
        Print the tree
        """
        if(node.is_leaf == False):
            for child in node.children:
                pre_space = "    " * depth
                print(f"{pre_space}├── Attribute{node.attributeIndex} = {child[1]}")
                self.print_tree(child[0], depth + 1)
        else:
            pre_space = "    " * (depth - 1)
            print(f"{pre_space}| -> {node.classValue},{node.size}")

def load_csv_file(filename):
    """
    Load csv file
    """
    data = []
    
    with open(filename, newline='') as filedata:
        reader = csv.reader(filedata)
        for row in reader:
            data.append(row)
    return data

def split_training_validation(training_data, validation_ratio=0.2, seed=42):
    """
    Split the training data into training data and validation data. The ratio is 80:20
    """
    random.seed(seed)
    shuffled_data = training_data[:]
    random.shuffle(shuffled_data)
    
    split_index = int(len(shuffled_data) * (1 - validation_ratio))
    new_training_data = shuffled_data[:split_index]
    validation_data = shuffled_data[split_index:]
    
    return new_training_data, validation_data

def classify_dtstar(training_data, testing_data):
    """
    Build(include prune) tree and test on testing_data
    """
    # Split the training data to get validation data for prunning
    training_data, validation_data = split_training_validation(training_data=training_data)

    # Build tree from training_data
    tree = DT()
    tree.build(node=tree.root, training_data=training_data)

    # Prune tree
    tree.prune(validation_data)

    # Test tree
    total = len(testing_data)
    correct = 0
    for row in testing_data:
        if(tree.get_class(row, usingValidation=False) == row[-1]):
            correct += 1
    # print(f"Post Prune: {correct}/{total} -> {correct/total}")
    return (correct/total)

def ten_fold_validation(filename):
    data_by_fold = []
    current_fold_data = []

    with open(filename) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("fold"):
                if current_fold_data:
                    data_by_fold.append(current_fold_data)
                    current_fold_data = []
            else:
                parts = line.split(",")
                current_fold_data.append(parts)

        # Add the last fold's data
        if current_fold_data:
            data_by_fold.append(current_fold_data)

        res = 0

        for i in range(len(data_by_fold)):
            training_data = []
            testing_data = []
            for j in range(len(data_by_fold)):
                if(i == j):
                    testing_data = testing_data + data_by_fold[j]
                else:
                    training_data = training_data + data_by_fold[j]
            res += classify_dtstar(training_data, testing_data)

        return res/10
if __name__ == '__main__':
    print(ten_fold_validation("pima-folds.csv"))
    





