class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor '''
        
        # Karar dugumleri icin
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # Deger dugumleri icin
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, max_depth=5):
        ''' constructor '''
        
        # Agacin kokunun yapilandirilmasi
        self.root = None
        
        # Classificationun durma kosullari
        self.min_samples_split = 2
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        ''' Karar agacini olusturmak icin recursive fonksiyon ''' 
        
        X = dataset.copy()
        Y = list()
        temp_row_num = len(X[0])-1
        
        for row in X:
            Y.append(int(row[temp_row_num]))
            
        num_samples = len(X)
        num_features = len(X[0])-1
        #print(num_samples)
        
        # Durma kosullari saglanana kadar ayirma devam edecek
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # En iyi ayrimin bulunmasi
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # Information gainin pozitif olmasi kontrolu
            if best_split["info_gain"]>0:
                # Sola ayril
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # Saga ayril
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # Karar dugumune geri don
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # Deger dugumunun hesaplanmasi
        leaf_value = self.calculate_leaf_value(Y)
        # Deger dugumunun geri donulmesi
        return Node(value=leaf_value)

    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' En iyi ayirimin bulunmasini saglayan fonksiyon '''
        
        best_split = {}
        max_info_gain = -float("inf")
        
        feature_values = []

        for feature_index in range(num_features):

            feature_values = []
            for rowA in range(num_samples): 
                temp = dataset[rowA][feature_index] 
                feature_values.append(temp)
            
            possible_thresholds = []
            for x in feature_values:
                if x not in possible_thresholds:
                    possible_thresholds.append(x)

            possible_thresholds.sort()

            # Verisetinde bulunan tum ozniteikler içinde gezilmesi
            for threshold in possible_thresholds:
                # Anlik ayrimin alinmasi
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)

                # NULL deger kontrolu
                if len(dataset_left)>0 and len(dataset_right)>0:
                        
                    y = []
                    left_y = []
                    right_y = []

                    y_first = []
                    left_y_first = []
                    right_y_first = []

                    temp_col = 0
                    temp_col_l = 0
                    temp_col_r = 0

                    y_first.extend(dataset) 
                    temp_col = len(y_first[0])-1
                    for col in y_first:
                        y.append(int(col[temp_col]))

                    left_y_first.extend(dataset_left)
                    temp_col_l = len(left_y_first[0])-1
                    for col in left_y_first:
                        left_y.append(int(col[temp_col_l]))

                    right_y_first.extend(dataset_right)
                    temp_col_r = len(right_y_first[0])-1
                    for col in right_y_first:
                        right_y.append(int(col[temp_col_r])) 
                        
                    # Information gainin hesaplanmasi
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    # Gerekirse en iyi ayrimin guncellenmesi
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                                             
        return best_split


    def split(self, dataset, feature_index, threshold):
        ''' Verinin ayirilmasi icin gerekli fonksiyon '''
        
        dataset_left = []
        dataset_right = []

        for row in dataset:
            if row[feature_index] <= threshold:
                dataset_left.append(row)

        for row in dataset:
            if row[feature_index] > threshold:
                dataset_right.append(row)

        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child): 
        ''' Inf. gainin hesaplanmasi icin gereli fonksiyon '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))

        return gain
      
    def gini_index(self, y):
        ''' Gini indexin hesaplanmasi icin gerekli fonksiyon '''

        class_labels = []
        for i in y:
            if i not in class_labels:
                class_labels.append(i)

        counter = 0
        gini = 0
        for label in class_labels:
            for value in y:
                if value == label:
                    counter+=1
            p_cls = counter / len(y)
            gini += p_cls**2
            counter = 0

        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        ''' Deger dugumlerinin hesaplanmasi icin gerekli fonksiyon '''
        
        return max(Y, key=Y.count)

        
    def fit(self, X, Y):
        ''' function to train the tree '''

        for rowX,rowY in zip(X,Y):
            rowX.append(rowY)
        
        dataset = X.copy()    
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' Antrenman verisinin öğrenilmesi icin gerekli fonksiyon '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        '''Tek bir verinin tahmini icin gerekli fonksiyon '''
        
        if tree.value!=None: 
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
