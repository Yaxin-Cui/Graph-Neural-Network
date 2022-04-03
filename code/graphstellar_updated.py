import numpy as np
import networkx as nx
import pandas as pd
#import os


import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.layer.link_inference import LinkEmbedding
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification, link_regression

from stellargraph.mapper import FullBatchNodeGenerator, FullBatchLinkGenerator
from stellargraph.layer import GAT
#from stellargraph import globalvar
from stellargraph import StellarGraph

import tensorflow.keras as keras # DO NOT USE KERAS DIRECTLY
#from sklearn import preprocessing, feature_extraction, model_selection
import matplotlib.pyplot as plt
import matplotlib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#import tensorflow.keras.backend as K
#import scipy

from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score



#from sklearn.model_selection import train_test_split
#from sklearn.neural_network import MLPClassifier
#from sklearn.linear_model import LogisticRegression

from sklearn.metrics.pairwise import cosine_similarity

import time



def predicted_Abin_14(Abin_14, features_14, num_neighbors):
    """Function to predict 2014 links from features only
    Any ML model can be used here to predict the adjacency matrix
    1 IS A k-nearest neighbor approach
    
    """
    way_to_predict = 1
    
    num_nodes_14 = len(Abin_14)

    if(way_to_predict == 0):
       
    #Based on feature similarity and fixed cut-off
        simil = cosine_similarity(features_14,features_14)
        c = np.percentile(simil, 96)
        Abin_14_pred = (simil > c)*1  
        
        
    elif(way_to_predict == 1):   
    #Based on feature similarity and fixed cut-off
    
        Abin_14_pred = np.zeros((num_nodes_14, num_nodes_14))
        simil = cosine_similarity(features_14,features_14)
        #num_neighbors = 10
        for i in range(num_nodes_14):
            arr = simil[i,:]
            idx = np.argsort(-arr)[:num_neighbors]
            Abin_14_pred[i,idx[1:]] = 1
        
        

        
    elif(way_to_predict == 2):
    #Predict a random matrix with same density as 2014
        
        Arand = np.random.rand(num_nodes_14, num_nodes_14)
        dens = np.sum(Abin_14)/(num_nodes_14**2)
        Abin_14_pred = (Arand<=dens)*1
        
    elif(way_to_predict == 3):  
    #Predicted 2014 is same as true 2014
        Abin_14_pred = np.copy(Abin_14)
    
    elif(way_to_predict == 4):    
    #Imported Abin_14 for combining cars from 2013 and leaving blank all cars from 2014
        A_14_pred = np.array(pd.read_csv("data/premarket_14_net.csv", usecols = lambda column : column not in ['Unnamed: 0']))
        Abin_14_pred = (A_14_pred >= cut_off)*1

    elif(way_to_predict == 5):    
    #Imported Abin_14 for combining cars for logistic regression to predict 2014
        Abin_14_pred = np.array(pd.read_csv("data/augment_14_high_cut.csv", usecols = lambda column : column not in ['Unnamed: 0']))

    
    return Abin_14_pred





def remove_isolated_A(A1):
    #This function removes nodes which are not connected to any other node and returns the index of nodes removed and the new adjacency matrix
    isl = np.where(np.sum(A1, axis = 0) == 0)[0]
    A1 = np.delete(A1, isl, axis = 0)
    A1 = np.delete(A1, isl, axis = 1)
    
    if(len(isl)==0):
        return [], A1
    return isl[0], A1



def permute_test(A0, b, model, batch_size, num_samples, colnums, edge_ids, edge_labels):
    #Calculate permutation based feature importance 
    g = nx.from_numpy_matrix(A0)

    features1 = np.copy(b)
     
    tempfeat = np.copy(features1[:, colnums])
    tempfeat = shuffle(tempfeat)
    features1[:, colnums] =  np.copy(tempfeat)
    
    for node_id, node_data in g.nodes(data=True):
        node_data["feature"] = features1[node_id,:]
        
    G = StellarGraph.from_networkx(g, node_features = "feature")
    gen = GraphSAGELinkGenerator(G,  batch_size, num_samples)

    
    t_metrics = model.evaluate_generator(gen.flow(edge_ids,edge_labels))
    #print(t_metrics)
    
    return t_metrics[1], features1
  
def new_test(A0, Abin_14, features1, model, batch_size, num_samples, edge_labels_test, edge_ids_test):
    #Calculate performance for a new adjacency matrix 
    g = nx.from_numpy_matrix(A0)


    for node_id, node_data in g.nodes(data=True):
        node_data["feature"] = features1[node_id,:]
     

    G = StellarGraph.from_networkx(g, node_features = "feature")
    gen = GraphSAGELinkGenerator(G,  batch_size, num_samples)

    
    t_metrics = model.evaluate_generator(gen.flow(edge_ids_test,edge_labels_test))
    
    pred_prob = model.predict(gen.flow(edge_ids_test,edge_labels_test))
    
    
    adjacency_method = 0
    #predicted_label = np.round(pred_prob)
    
    if(adjacency_method == 0):
        predicted_label = np.round(pred_prob)
    elif(adjacency_method == 1):
        coff = np.percentile(pred_prob, 90)   
        predicted_label = (pred_prob >= coff)*1  

    
    newA = np.zeros((num_nodes_14,num_nodes_14))
    for seq, ied in enumerate(edge_ids_test):
        newA[ied[0],ied[1]] = predicted_label[seq]
    
    return t_metrics, newA, edge_ids_test, pred_prob, predicted_label



# binary/real(0), categorical (1), 
# [0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0]
def encode_carattr(node_attr,types, vec):
    
    #Change mixed features (categorical and continuous) to one hot encoded vectors
    
    
    num_cols = []
    num_items, num_f = np.shape(node_attr)
    attr_names = list(node_attr)
     
    print(list(node_attr))
    #lastvalue = 0
    feat = []
    for i in vec:
        print(i)
        typevec = types[i]
        if(typevec == 0): #binary feature
            feat = feat + [np.array(node_attr[attr_names[i]].values).reshape(num_items,1)]
            num_cols.append(1)
        elif(typevec == 1): #categorical feature
            values = node_attr[attr_names[i]]
            label_encoder = LabelEncoder()
            integer_encoded = label_encoder.fit_transform(values)
            # binary encode
            onehot_encoder = OneHotEncoder(sparse=False)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
            feat = feat + [np.array(onehot_encoded)]
            num_cols.append(np.shape(onehot_encoded)[1])
            #print(np.shape(onehot_encoded))
        elif(typevec == 2): #continuous feature
            values = node_attr[attr_names[i]].values
            #feat.append(np.array((values>np.median(values))*1).reshape(1,num_nodes))
            feat = feat + [np.array((values>np.median(values))*1).reshape(num_items,1)]
        else:
            #print(np.shape(feat))
            
            feat = feat + [np.reshape(node_attr[attr_names[i]].values, (num_items,1))]

    features = np.hstack(feat)
    print(num_cols)
    return features


def plot_roc(y_test, y_hat, auc, graphmethod):
    
    #Plot ROC curve

    fpr, tpr, threshold = metrics.roc_curve(y_test, y_hat)    
    plt.title(graphmethod + ' ROC')
    plt.plot(fpr, tpr, label = ' AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    #Save the results to a text file
    np.savetxt('results/auc.txt', np.array([fpr, tpr]))
    

def plot_history(history):
    
    #Plot history of training
    metrics = sorted(history.history.keys())
    metrics = metrics[:len(metrics)//2]

    f,axs = plt.subplots(1, len(metrics), figsize=(12,4))
    for m,ax in zip(metrics,axs):
        # summarize history for metric m
        ax.plot(history.history[m])
        ax.plot(history.history['val_' + m])
        ax.set_title(m)
        ax.set_ylabel(m)
        ax.set_xlabel('epoch')
        ax.legend(['train', 'test'], loc='upper right')



if __name__ == "__main__":
    
    part = 0
    
    font = {'family' : 'normal',
            'size'   : 18}
    
    matplotlib.rc('font', **font)    
    
    np.random.seed(0)
    


    if(part==0):   

        """This part uses graphsage to calculate auc for a binary network. 
        
        Use the following settings:
        
        small_set_feats = 1, if prediction only uses 6 features instead of 29 features
        future_predict = 1, if test data is from 2014 and not held out edges from 2013
        predict_2015 = 1, if test data is from 2015
        report_alltest = 1, if AUC is also reported for entire network and not balanced positive and negative edges
        
        Abin is the binary adjacency matrix
        Abin_14 is the binary adjacency matrix for future year (2014 or 2015)
        node_attr is the node attributes for all cars
        num_nodes is the number of nodes in the graph
        """

        
        small_set_feats = 0
        total_runs = 5
        future_predict = 1
        predict_2015 = 0
        report_alltest = 0
        num_neighbors = 5
        gat_method = 1
        batch_size = 20
        epochs = 100
        


        #Read connectivity of every car from 2013 in df_attr
        if(small_set_feats == 1):
            #Read link matrix for 2013 data
            df_attr = pd.read_csv("data/consider_net_2013c.csv", usecols = lambda column : column not in ['Car'])
        else:
            #Read link matrix for 2013 data
            df_attr = pd.read_csv("data/consider_net_2013.csv", usecols = lambda column : column not in ['Unnamed: 0'])            
        
        
        A = np.array(df_attr)

        
        #Remove nodes which are isolated
        idx_rm, A = remove_isolated_A(A)
        
        
        #Edge weight cutoff to generate the binary network. 
        #cut_off = np.percentile(A, 99)   
        cut_off = 1.0   
        
        #Use some edge weight cutoff options to generate a binary network. 
        #To try multiple cut-offs or number of neighbors, specify cvec and neighborset as a list
        
        #cvec = [1.0, 5.0, 10.0, 15.0, 20.0]
        cvec = [1.0]
        
        #neighborset = [2, 5, 10]
        neighborset = [5]
        
        #Performance list
        all_perf = []        
        
        #Loop over all cut-off values.
        for cut_off in cvec:
            #Loop over all possible number of neighbors 
            for num_neighbors in neighborset:
            
                #Convert matrix to binary adjacency matrix
                Abin = (A >= cut_off)*1  

                    
                #binary graph
                g_nx = nx.from_numpy_matrix(Abin)             
                num_nodes = len(g_nx.nodes())
                    
        

                    
                    
                #Read adjacency matrix of every car from 2014 network (or 2015) in df_attr_14 for test set
                if(predict_2015 == 1):
                    
                    df_attr_14 = pd.read_csv("data/consider_net_2015.csv", usecols = lambda column : column not in ['Unnamed: 0'])
               
                else:    
                    
                    if(small_set_feats == 1):
                        #Read adjacency matrix for 2014 data
                        df_attr_14 = pd.read_csv("data/consider_net_2014c.csv", usecols = lambda column : column not in ['Car'])
            
                    else:
                        df_attr_14 = pd.read_csv("data/consider_net_2014.csv", usecols = lambda column : column not in ['Unnamed: 0'])
                    #df_attr_14 = pd.read_csv("data/consider_net_2015.csv", usecols = lambda column : column not in ['Unnamed: 0'])
 
                   
                #Abin_14 is the binary network for test set (future year)  
                A_14 = np.array(df_attr_14)
                num_nodes_14 = len(A_14)
                Abin_14 = (A_14 >= cut_off)*1
                
                #Find networkx graph from binary matrix
                g_nx_14 = nx.from_numpy_matrix(Abin_14) 
                
                
        
                
                # small_set_feats is 1 if only 6 features are to be used else it is 0.
                
                if(small_set_feats == 1):
                    node_attr = pd.read_csv("data/node_att_2013_few.csv", usecols = lambda column : column not in ['Model_id','Model_name'])
                    
                    if(predict_2015 == 1):
                        node_attr_14 = pd.read_csv("data/node_att_2015_few.csv", usecols = lambda column : column not in ['Model_id','Model_name'])
                    else:
                        node_attr_14 = pd.read_csv("data/node_att_2014_few.csv", usecols = lambda column : column not in ['Model_id','Model_name'])
                else:     
                    node_attr = pd.read_csv("data/node_attr_2013.csv", usecols = lambda column : column not in ['Model_id','Model'])
                    
                    if(predict_2015 == 1):
                        node_attr_14 = pd.read_csv("data/node_attr_2015.csv", usecols = lambda column : column not in ['Model_id','Model']) 
                    else:
                        node_attr_14 = pd.read_csv("data/node_attr_2014.csv", usecols = lambda column : column not in ['Model_id','Model']) 
                    
                    node_attr_14 = node_attr_14.reset_index(drop=True)
                    
                    #Remove features for cars which were removed due to isolated node
                    node_attr = node_attr.drop(idx_rm)
                    node_attr = node_attr.reset_index(drop=True)
                    
                #Join the two attributes list so that one-hot encoding has similar number of columns in train and test
                all_node_attr = pd.concat([node_attr, node_attr_14], axis=0)    
                    
                #Total features
                numall_feats = np.shape(node_attr)[1]

                

                    
                input_vec = list(range(numall_feats))

                    
                    
                if(small_set_feats == 1):
                    
                    #feattype is type of feature-> non-categorical (0) or categorical (1)
                    #This is manually coded
                    feattype = [0,0,0,0,1,1]
                    
                    #encode_carattr does one hot encoding of all features
                    allfeatures = encode_carattr(all_node_attr, vec = input_vec, types = feattype)
                    
                    #Normalize range of features
                    allfeatures = np.where(np.max(allfeatures, axis=0)==0, allfeatures, allfeatures*1./np.max(allfeatures, axis=0))
                    
                    #Remove nan values
                    allfeatures = np.nan_to_num(allfeatures)
                    features = allfeatures[0:num_nodes,:]
                    features_14 = allfeatures[num_nodes:,:]
                    
    
                else:
    
                    feattype = [1,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
                    allfeatures = encode_carattr(all_node_attr, vec = input_vec, types = feattype)
                    
                    #Normalize features
                    allfeatures = np.where(np.max(allfeatures, axis=0)==0, allfeatures, allfeatures*1./np.max(allfeatures, axis=0))
                    
                    #Remove nan values
                    allfeatures = np.nan_to_num(allfeatures)
                    features = allfeatures[0:num_nodes,:]
                    features_14 = allfeatures[num_nodes:,:]
                    
                
                num_feats = np.shape(features)[1]
                
    
                #Predicted A_14. Use below function with regression models
                #So-called adjacency matrix in the paper
                Abin_14_pred = predicted_Abin_14(Abin_14, features_14, num_neighbors)
        
                
                
                #Find networkx graph from predicted binary matrix
                g_nx_14_pred = nx.from_numpy_matrix(Abin_14_pred)      
                
                
                    
                #Predicting a link classification or regression
                
                
        
                #Allocate features to nodes (2013)
                for nid in range(num_nodes):
                    
                    g_nx.nodes[nid]["feature"] = features[nid,:]        
                    
                    
                #Find the largest connected component for training
        
                #print("Largest connected component: {} nodes, {} edges".format(g_nx.number_of_nodes(), g_nx.number_of_edges()))   
                #g_nx_ccs = (g_nx.subgraph(c).copy() for c in nx.connected_components(g_nx))
                #g_nx = max(g_nx_ccs, key=len)
                #print("Largest connected component: {} nodes, {} edges".format(g_nx.number_of_nodes(), g_nx.number_of_edges()))   
                
                current_perf = []
                
                #Multiple runs with different seeds
                
                for numrun in range(total_runs):    
    
                    #Each run has a separate seed                
                    sg.random.set_seed(numrun)
                    
                    #Test train split in two ways. If future_predict = 0, then held out
                    #edges from the same year. Otherwise a future year.
                    
                    #Predict 2014 network or held-out edges from the same network
                    
                    
                    
                    if(future_predict==0):
                    
                        edge_splitter_test = EdgeSplitter(g_nx)
                        
    #                    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(p=0.1, method="global", keep_connected=True)        
    #        
    #                    edge_splitter_train = EdgeSplitter(G_test)
    #                    
    #                    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(p=0.1, method="global", keep_connected=True)
    #            
    #                    G_train = sg.StellarGraph(G_train, node_features="feature")
    #                    G_test = sg.StellarGraph(G_test, node_features="feature")
                        
                       
                        G_train, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(p=0.1, method="global", keep_connected=False)        
            
                        G_test = nx.difference(g_nx, G_train)
                        
                        #Predicted A_14. Use below function with regression models
                        Abin_13_pred = predicted_Abin_14(Abin, features, num_neighbors)
    #    
    #            
    #            
    #                    #Find networkx graph from predicted binary matrix
                        g_nx_13_pred = nx.from_numpy_matrix(Abin_13_pred)   
                        
                        #Allocate features to nodes                        
                        for node_id, node_data in g_nx_13_pred.nodes(data=True):
                            node_data["feature"] = features[node_id,:]
                                
                        edge_ids_trpos = list(G_train.edges())
                        edge_ids_trpos = [[it[0], it[1]] for it in edge_ids_trpos]                    
    
                        
                        num_pos_edges = len(edge_ids_trpos)
    
                        #Find all missing edges which exist in 2013 data
                        Abintr = nx.to_numpy_matrix(G_train)
                        allneg_edges = np.where(Abintr==0)
                        total_neg_edges = np.arange(len(allneg_edges[0]))
                        np.random.shuffle(total_neg_edges)
                        neg_edges = [[allneg_edges[0][i], allneg_edges[1][i]] for i in total_neg_edges[0:num_pos_edges]]
    
                        edge_ids_train = edge_ids_trpos + neg_edges                    
                        
                        edge_labels_train=[]
                        for i in edge_ids_train:
                            edge_labels_train.append(Abintr[i[0],i[1]])
                        edge_labels_train = np.array(edge_labels_train)
                        
                        
                        G_test = StellarGraph.from_networkx(g_nx_13_pred, node_features = "feature")
                        #G_test = sg.StellarGraph(G_train, node_features="feature")                     
                        G_train = sg.StellarGraph(G_train, node_features="feature")
                       
                        
                        
                    else:
                        
                        #Edges from future year for test data
                        
                        G_train = StellarGraph.from_networkx(g_nx, node_features="feature")
                        
                        for node_id, node_data in g_nx_14_pred.nodes(data=True):
                            node_data["feature"] = features_14[node_id,:]
                            
                        G_test = StellarGraph.from_networkx(g_nx_14_pred, node_features = "feature")
                        
                        
                        #Find all edges which exist in 2014 data
                        edge_ids_tepos = list(g_nx_14.edges())
                        edge_ids_tepos = [[it[0], it[1]] for it in edge_ids_tepos]
                        
                        num_pos_edges = len(edge_ids_tepos)
                        
                        #Find all missing edges which exist in 2014 data
                        allneg_edges = np.where(Abin_14==0)
                        total_neg_edges = np.arange(len(allneg_edges[0]))
                        np.random.shuffle(total_neg_edges)
                        
                        #Test set balancing
                        #Sample same number of missing links as existing links for balanced testing
                        #Alternatively, you can have missing links in test set too (slows the code)
                        
                        neg_edges = [[allneg_edges[0][i], allneg_edges[1][i]] for i in total_neg_edges[0:num_pos_edges]]
                        #neg_edges = [[allneg_edges[0][i], allneg_edges[1][i]] for i in total_neg_edges]#
                        
                        # test edges combine both positive and negative edges (the same number)
                        edge_ids_test = edge_ids_tepos + neg_edges
                        
    
    
                        #Find all edges which exist in 2013 data
                        edge_ids_trpos = list(g_nx.edges())
                        edge_ids_trpos = [[it[0], it[1]] for it in edge_ids_trpos]
                        
                        num_pos_edges = len(edge_ids_trpos)
    
                        #Find all missing edges which exist in 2013 data
                        allneg_edges = np.where(Abin==0)
                        total_neg_edges = np.arange(len(allneg_edges[0]))
                        np.random.shuffle(total_neg_edges)
                        neg_edges = [[allneg_edges[0][i], allneg_edges[1][i]] for i in total_neg_edges[0:num_pos_edges]]
    
                        edge_ids_train = edge_ids_trpos + neg_edges
                        
                        
                        #edge_labels is the label (0/1) of an edge in train or test set
                        
                        edge_labels_test=[]
                        for i in edge_ids_test:
                            edge_labels_test.append(Abin_14[i[0],i[1]])
                        edge_labels_test = np.array(edge_labels_test)
                        
                        edge_labels_train=[]
                        for i in edge_ids_train:
                            edge_labels_train.append(Abin[i[0],i[1]])
                        edge_labels_train = np.array(edge_labels_train)
                        
                        #edge_splitter_train = EdgeSplitter(G_test)
                        
                        #G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(p=0.1, method="global", keep_connected=True)
    
                    
                    

                    if(gat_method==0):
                    
                        #num_samples = [20, 10]
                        #num_samples = [10, 5]
                        num_samples = [5, 5]
        
                        
                        # linkgenerator
                        train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples)
                        test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)

                        #layer_sizes = [20, 20]
                        #layer_sizes = [10, 10]
                        layer_sizes = [5, 5]
                    
                        assert len(layer_sizes) == len(num_samples)

                        graphsage = GraphSAGE(layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.3)

                        x_inp, x_out = graphsage.in_out_tensors()

                        #Combine node embeddings to find edge embedding using dot product                
                        prediction = link_classification(output_dim=1, output_act="relu", edge_embedding_method='ip')(x_out)
                        #prediction = link_classification(output_dim=1, output_act="relu", edge_embedding_method='avg')(x_out)
                        #prediction = link_classification(output_dim=1, output_act="relu", edge_embedding_method='mul')(x_out)
                        #prediction = link_classification(output_dim=1, output_act="sigmoid", edge_embedding_method='ip')(x_out)
                        #prediction = link_classification(output_dim=1, output_act="relu", edge_embedding_method='l2')(x_out)
                    


                    else:

                        # GAT link gererator
                        train_gen = FullBatchLinkGenerator(G_train,method="gat")
                        test_gen = FullBatchLinkGenerator(G_test,method="gat")
                        
                        gat = GAT(layer_sizes=[8, 4],
                        activations=["elu", "softmax"],
                        attn_heads=8,
                        bias = True,
                        generator=train_gen,
                        in_dropout=0.5,
                        attn_dropout=0.5,
                        normalize=None)

                        x_inp, x_out = gat.in_out_tensors()

                        prediction = LinkEmbedding(activation="relu",method ="ip")(x_out)
                        prediction = keras.layers.Reshape((-1,))(prediction)
                    
                    
                   
                    
                    #Train tf model for classification
                    model = keras.Model(inputs=x_inp, outputs=prediction)
                    
                    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss=keras.losses.binary_crossentropy, metrics=[keras.metrics.AUC()])
                    
                    
                    
                    init_train_metrics = model.evaluate(train_gen.flow(edge_ids_train,edge_labels_train))
                    init_test_metrics = model.evaluate(test_gen.flow(edge_ids_test, edge_labels_test))
                    
                    
            
                    print("\nTrain Set Metrics of the initial (untrained) model:")
                    for name, val in zip(model.metrics_names, init_train_metrics):
                        print("\t{}: {:0.4f}".format(name, val))
                        #current_perf.append(val)
                    
                    print("\nTest Set Metrics of the initial (untrained) model:")
                    for name, val in zip(model.metrics_names, init_test_metrics):
                        print("\t{}: {:0.4f}".format(name, val))
                        #current_perf.append(val)
                    start_time = time.time()
                    
                    history = model.fit(train_gen.flow(edge_ids_train,edge_labels_train),
                                epochs=epochs, validation_data=test_gen.flow(edge_ids_test, edge_labels_test),
                                verbose=2)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    
                    plot_history(history)  
                    
                    
                    train_metrics = model.evaluate(train_gen.flow(edge_ids_train,edge_labels_train))
                    test_metrics = model.evaluate(test_gen.flow(edge_ids_test, edge_labels_test))
                    
                    print("\nTrain Set Metrics of the trained model:")
                    for name, val in zip(model.metrics_names, train_metrics):
                        print("\t{}: {:0.4f}".format(name, val))
                        #current_perf.append(val)
                    print("\nTest Set Metrics of the trained model:")
                    for name, val in zip(model.metrics_names, test_metrics):
                        print("\t{}: {:0.4f}".format(name, val))
                    #Current run test performance 
                    current_perf.append(val)
                #All runs test performance
                #all_perf.append(current_perf) 
                all_perf.append(test_metrics)                   
                    
                A = np.array(df_attr)
    
                
                #Test confusion matrix
                Ytest = model.predict(test_gen.flow(edge_ids_test, edge_labels_test))
                if(gat_method==1):
                    Ytest = Ytest.T
                print("\nTesting confusion matrix\n")
                print(confusion_matrix(edge_labels_test, np.round(Ytest)))
                                
                #Training confusion matrix
                print("\nTraining confusion matrix\n")
                Ytrain = model.predict(train_gen.flow(edge_ids_train, edge_labels_train))

                if(gat_method == 1):
                    Ytrain = Ytrain.T

                print(confusion_matrix(edge_labels_train, np.round(Ytrain)))
                
                print("Classification report and F1 score")
                print(classification_report(edge_labels_test, np.round(Ytest), target_names=['0', '1']))
                
                
              
                #Check performance on prediction of all edges instead of equal positive and negative edges
                if(report_alltest == 1):
                    print("\n Print the performance for entire future network")
                    neg_edges = []
                    for i in range(num_nodes_14-1):
                        for j in range(i+1,num_nodes_14):
                            if(Abin_14[i,j] ==0):
                                neg_edges.append([i,j])            
                                
                    #neg_edges = [[allneg_edges[0][i], allneg_edges[1][i]] for i in total_neg_edges]#
                    edge_ids_test = edge_ids_tepos + neg_edges
                    edge_labels_test=[]
                    for i in edge_ids_test:
                        edge_labels_test.append(Abin_14[i[0],i[1]])
                    edge_labels_test = np.array(edge_labels_test)
                    test_gen = GraphSAGELinkGenerator(G_test,  batch_size, num_samples)
                    model.evaluate_generator(test_gen.flow(edge_ids_test, edge_labels_test))
                    
                    #Test confusion matrix
                    Ytest = model.predict(test_gen.flow(edge_ids_test, edge_labels_test))
                    print("\nTesting confusion matrix\n")
                    print(confusion_matrix(edge_labels_test, np.round(Ytest)))        
                    
                    print(classification_report(edge_labels_test, np.round(Ytest), target_names=['0', '1']))
                    
                    test_metrics = model.evaluate_generator(test_gen.flow(edge_ids_test, edge_labels_test))
                
        all_perf = np.array(all_perf)
        
        print(["Average testing performance for all runs: ", np.mean(all_perf)])
            

        

        

        
    elif(part==1):  

        #Permutation based feature importance for any model run above. Lower prediction performance means more important feature.

        """
        num_repeat is the number of repeats for permutation testing. Higher the better.
        test flag is 0 when not testing. 1 if training feature importance is evaluated.
        fvec is the number of categories for each feature
        
        
        baseline is model evaluation for a given data
        
        """
        num_repeat = 50
        test = 0
        
        bb =[]
        
        
        if(small_set_feats == 1):
            fvec = [1, 1, 1, 1, 5, 17]
        else:
            
            fvec = [2, 1, 67, 5, 9, 34, 25, 17, 8, 1, 8, 15, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        numf = len(fvec)
        perf = np.zeros((numf,num_repeat))
        
        if(test == 1):
        
            baseline = model.evaluate_generator(test_gen.flow(edge_ids_test, edge_labels_test))[1]
            
            #Loop for every feature
            for i in range(numf):
                cols = np.arange(np.sum(fvec[0:i]), np.sum(fvec[0:i+1])).astype(np.int64)
                print(cols)
                
                for j in range(num_repeat):
                    perf[i,j],b = permute_test(A_14, features_14, model, batch_size, num_samples, cols,  edge_ids_test, edge_labels_test)
                    bb.append(b)
                print([i, np.mean(perf[i,:])/baseline])  
                
            Y = model.predict(test_gen.flow(edge_ids_test, edge_labels_test))
            
            plt.figure()
            auc = roc_auc_score(edge_labels_test, Y)
            plot_roc(edge_labels_test, Y, auc,'GraphSage')      
            print('AUC: {:.4f}'.format(auc))
            print("Ranking from highest to least important attribute")
            print([list(node_attr)[i] for i in np.argsort(np.mean(perf, axis = 1))])
            plt.savefig('results/'+str(test)+'AUC.eps', transparent = 'True', dpi=200, format='eps' )
        else:
            
            baseline = model.evaluate_generator(train_gen.flow(edge_ids_train, edge_labels_train))[1]
            
            for i in range(numf):
                cols = np.arange(np.sum(fvec[0:i]), np.sum(fvec[0:i+1])).astype(np.int64)
                print(cols)
                for j in range(num_repeat):
                    perf[i,j], b = permute_test(Abin, features, model, batch_size, num_samples, cols,  edge_ids_train, edge_labels_train)
                    bb.append(b)
                print([i, np.mean(perf[i,:])/baseline])    
                
            Y = model.predict(train_gen.flow(edge_ids_train, edge_labels_train))
            
            plt.figure()
            auc = roc_auc_score(edge_labels_train, Y)
            plot_roc(edge_labels_train, Y, auc,'GraphSage')      
            print('AUC: {:.4f}'.format(auc))
            print("Ranking from highest to least important attribute")
            print([list(node_attr)[i] for i in np.argsort(np.mean(perf, axis = 1))])
            plt.savefig('results/'+str(test)+'AUC.eps', transparent = 'True', dpi=200, format='eps' )
            
        
        plt.figure()
        plt.bar(np.arange(numf), np.mean(perf, axis = 1))
        
        #Save performance for each feature
        np.savetxt('results/perf.txt',perf)
        
        plt.ylabel('Feature Importance', fontsize=18)
        plt.xticks(np.arange(numall_feats),list(node_attr), fontsize=18, rotation = 45)
        plt.yticks(fontsize=18)
        plt.savefig('results/'+str(test)+'featureimportance.eps', transparent = 'True', dpi=200, format='eps' )
        
    elif(part==2): 
        
        """Test recursive improvement of test performance by calculating the predicted adjacency matrix and
        using it again to improve the predictions
        """
        num_repeat = 3
        
        newA = np.copy(Abin_14_pred)
        
        #auc_t = np.zeros(num_repeat)
        allA = []
        all_perf = []
        allnewedge_ids_test = []
        allnewpred_prob =[]
        allnewpredicted_label =[]
        
        
        
        
        allpos_edges = np.where(Abin_14==1)
        total_pos_edges = np.arange(len(allpos_edges[0]))
        
        pos_edges = [[allpos_edges[0][i], allpos_edges[1][i]] for i in total_pos_edges]
        
        #Find all missing edges which exist in 2014 data
        allneg_edges = np.where(Abin_14==0)
        total_neg_edges = np.arange(len(allneg_edges[0]))


        
        #neg_edges = [[allneg_edges[0][i], allneg_edges[1][i]] for i in total_neg_edges[0:num_pos_edges]]
        neg_edges = [[allneg_edges[0][i], allneg_edges[1][i]] for i in total_neg_edges]#
        
        edge_ids_test = pos_edges + neg_edges        
        
        
        #edge_labels is the label (0/1) of an edge in train or test set
        
    
        edge_labels_test=[]
        for i in edge_ids_test:
            edge_labels_test.append(Abin_14[i[0],i[1]])
        edge_labels_test = np.array(edge_labels_test)
        
        for j in range(num_repeat):
            
            allA.append(newA)
            perf, newA, newedge_ids_test, newpred_prob, newpredicted_label = new_test(newA, Abin_14, features_14, model, batch_size, num_samples, edge_labels_test, edge_ids_test)
            print(perf)
            
            all_perf.append(perf)
            allnewedge_ids_test.append(newedge_ids_test)
            allnewpred_prob.append(newpred_prob)
            allnewpredicted_label.append(newpredicted_label)
            

        
        
        

#['turbo',
# 'autotrans',
# 'awd',
# 'parent_brand',
# 'V16',
# 'V18',
# 'V20',
# 'V34',
# 'V44',
# 'segment_num',
# 'community',
# 'luxury',
# 'enginesize_modelave',
# 'power_modelave',
# 'fuelconsump_modelave',
# 'price_modelave',
# 'makeorigin',
# 'Fuel_per_power',
# 'thirdrow',
# 'Price_log',
# 'Power_log',
# 'import']             
       
    
