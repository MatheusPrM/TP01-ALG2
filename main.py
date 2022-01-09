#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Realizando as importações dos módulos necessários e permitidos

import numpy as np
import random
import math
import heapq
import sys


# In[2]:


#Função para obter os dados de um arquivo recebido e organizá-los especificamente

def get_data(lines):
    attribute_count = 0
    class_count = 0
    data_attribute = []
    data_class = []
    data = []
    
    for l in lines:
        
        if l.startswith('@'):
            if l.startswith('@attribute'):
                try:
                    lines_aux = l.split("[")
                    lines_aux = lines_aux[1].split("]")
                    lines_aux = lines_aux[0].split(",")
                    if(lines_aux[1][0] != " "):
                        data_attribute.append((float(lines_aux[0]),float(lines_aux[1])))
                    else:
                        data_attribute.append((float(lines_aux[0]),float(lines_aux[1].replace(' ',''))))
                    attribute_count += 1
                except:
                    lines_aux = l.split("{")
                    lines_aux = lines_aux[1].split("}")
                    lines_aux = lines_aux[0].split(",")
                    if(lines_aux[1][0] != " "):
                        data_class.append((lines_aux[0],lines_aux[1]))
                    else:
                        data_class.append((lines_aux[0],lines_aux[1].replace(' ','')))
                    class_count += 1
               
        else:
            lines_aux = l.split(",")
            l_aux = []
            for i in range(attribute_count):
                if(lines_aux[i][0] != " "):
                    l_aux.append(float(lines_aux[i]))
                else:
                    l_aux.append(float(lines_aux[i].replace(' ','')))
            if(lines_aux[attribute_count][0] != " "):
                l_aux.append(lines_aux[attribute_count].replace("\n",""))
            else:
                lines_aux[attribute_count] = lines_aux[attribute_count].replace(' ','')
                l_aux.append(lines_aux[attribute_count].replace("\n",""))
            l_tuple = tuple(l_aux)
            data.append((l_tuple))
            
    return data_attribute, data_class, data


# In[3]:


#Abertura do arquivo contendo a base de dados e Atribuição desses dados à estruturas lista

argv = sys.argv
dataset_arq = argv[1]
with open(dataset_arq) as f1:
    lines = f1.readlines()

dataset_attribute, dataset_class, dataset_data = get_data(lines)


# In[4]:


#Função para dividir as instâncias em listas de treino e teste

def train_test(list):
    train_num = math.floor( len(list) * 0.7)
    train = random.sample(list, train_num)
    test = []
    
    for x in list:
        if x not in train:
            test.append(x)
            
    return train, test


# In[5]:


#Utilizando a função acima para obter as listas de treino e teste

dataset_train, dataset_test = train_test(dataset_data)


# In[6]:


#Criação da classe nó para auxiliar na estrutura árvore

class node:
    
    def __init__(self,l,r,p, median):
        #O nó armazenará atributos do filho esquerdo, filho direio, ponto e mediana
        self.itens = p
        self.left = l
        self.right = r
        self.m = median
    
    
    def __repr__(self):
        s = str(self.itens)
        if(self.left != None):
            s += " " + str(self.left.itens)
        if(self.right != None):
            s += " " + str(self.right.itens)
        s += " " + str(self.m)
        
        return s


# In[7]:


#Criação da classe árvore kd e de métodos para construir a estrutura adequadamente
#(Explicação de métodos na documentação)

class kd_tree:
       
    def __init__(self,l):
        self.l = l
        self.n = (len(l[0]) - 1)
        self.tree = self.build_kd_tree(l, 0)
        
    
    def build_kd_tree(self, l, depth):

        current = depth % self.n

        if(len(l) == 1):
            return node(None,None,l[0],None)

        else:

            def sort_aux(x):
                return x[current]

            def median_index(l):
                med = (len(l) / 2 )
                med = math.floor(med)
                return med

            l.sort(key=sort_aux)
            med = median_index(l)

            l1 = l[:med]
            l2 = l[med:]

            v_left = self.build_kd_tree(l1, depth+1)
            v_right = self.build_kd_tree(l2, depth+1)

            v = node(v_left, v_right, None, med)

            return v


# In[8]:


#Criação da classe x_NN e de métodos para classificar e calcular as estatísticas da classificação 
#(Explicação de métodos na documentação)

class x_NN:

    def __init__(self, dataset_data, dataset_train, dataset_test, dataset_class, dataset_attribute):
        self.tree = kd_tree(dataset_train)
        self.predict = self.classify_by_nearest(dataset_test, self.tree)
        self.accuracy, self.precision, self.recall = self.get_metrics(dataset_test, self.predict, dataset_class, dataset_attribute)
        print("Número de Dados:", len(dataset_data))
        print("Pontos e suas respectivas classificações determinadas:", self.predict)
        print("Accuracy:", self.accuracy)
        print("Precision:", self.precision)
        print("Recall:", self.recall)
        

    def eucl_distance(self, point_1, point_2):

        if(point_1 == None or point_2 == None):
            return np.inf

        distance = 0

        for i in range(len(point_1)):
            distance += ( ( point_1[i] - point_2[i] ) ** 2 ) 

        distance = distance ** 0.5

        return distance


    def nearest_node(self, neigh_aux, k, neigh_list):

        for x in neigh_aux:
            heapq.heappush(neigh_list, x)

        while(len(neigh_list) > k):
            del neigh_list[len(neigh_list)-1]
            neigh_list.sort()

        return neigh_list


    def kd_tree_search_knn(self, v, p, depth, k, neigh_list):

        if(v == None):
            return None

        if(v.m == None):
            eucl_dist = self.eucl_distance(p, v.itens)
            heapq.heappush(neigh_list, (eucl_dist, v.itens[:-1], v.itens[-1]))
            return neigh_list

        else:

            visit_next = None
            visit_later = None

            if(p[depth % k] <= v.m):
                visit_next = v.left
                visit_later = v.right
            else:
                visit_next = v.right
                visit_later = v.left

            neigh_list = self.kd_tree_search_knn(visit_next, p, depth+1, k, neigh_list)

            eucl_dist = self.eucl_distance(p, neigh_list[0][1])
            dist = p[depth % k] - v.m

            if((eucl_dist >= dist) or (len(neigh_list) < k)):
                neigh_aux = self.kd_tree_search_knn(visit_later, p, depth+1, k, [])
                neigh_list = self.nearest_node(neigh_aux, k, neigh_list)

            return neigh_list
        

    def get_test_values(self, l_aux, i):
        new_list = []

        for j in l_aux[i][:-1]:
            new_list.append(j)

        return new_list

    
    def predict_class(self, p_list, k):
        p_dict = {}

        for i in range(k):
            key = p_list[i][-1]
            if key not in p_dict.keys():
                p_dict[key] = 1
            else:
                p_dict[key] += 1

        greatest_value = 0
        greatest_key = None

        for key in p_dict:
            if(p_dict[key] > greatest_value):
                greatest_value = p_dict[key]
                greatest_key = key

        return greatest_key


    def classify_by_nearest(self, l, tree):
        pred_list = []

        for i in range(len(l)):
            test_values = self.get_test_values(l, i)
            neighboor_list = []
            neigh_test = self.kd_tree_search_knn(tree.tree, test_values, 0, 2, neighboor_list)
            class_predict = self.predict_class(neighboor_list, 2)
            test_values.append(class_predict)
            test_values_tuple = tuple(test_values)
            pred_list.append(test_values_tuple)

        return pred_list


    def get_conf_matrix(self, l_test, l_predict, l_classes, l_attributes):
        classes = []

        for j in range(len(l_classes)):
            classes.append(l_classes[j])

        c = len(l_attributes)
        tp_aux = 0
        tn_aux = 0
        fp_aux = 0
        fn_aux = 0

        for i in range(len(l_test)):
            if(classes[0][0] == l_predict[i][c] and l_predict[i][c] == l_test[i][c]):
                tp_aux += 1
            if(classes[0][1] == l_predict[i][c] and l_predict[i][c] == l_test[i][c]):
                tn_aux += 1
            if(classes[0][0] == l_predict[i][c] and l_predict[i][c] != l_test[i][c]):
                fp_aux += 1
            if(classes[0][1] == l_predict[i][c] and l_predict[i][c] != l_test[i][c]):
                fn_aux += 1

        tp = tp_aux / len(l_test)
        tn = tn_aux / len(l_test)
        fp = fp_aux / len(l_test)
        fn = fn_aux / len(l_test)

        confusion_matrix = np.array([[tp,fn],[fp,tn]])

        return confusion_matrix

    
    def get_metrics(self, l_test, l_predict, l_class, l_attribute):
        confusion_matrix = self.get_conf_matrix(l_test, l_predict, l_class, l_attribute)

        tp = confusion_matrix[0][0]
        fn = confusion_matrix[0][1]
        fp = confusion_matrix[1][0]
        tn = confusion_matrix[1][1]

        accuracy = ( (tp + tn) / (tp + tn + fp + fn) )
        precision = ( tp / (tp + fp) )
        recall = ( tp / (tp + fn) )

        return accuracy, precision, recall


# In[9]:


x_NN( dataset_data, dataset_train, dataset_test, dataset_class, dataset_attribute )

