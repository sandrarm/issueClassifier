'''
@author: Sandra

Program for predicting issue success using issue descriptions and comments and considering days between issue creation and comment registration

'''

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import xlsxwriter


ngram_range_min= 1
ngram_range_max= 3
max_features= 2000    
num_features_top = 1000


#Days between issue creation and comment registration
days=[1,5,10,20,30,40,50,60,70,80,90,100,200,300,400,500]

excelLine = 3

# loading the data set   
df = pd.read_excel('Data.xlsx', sheetname="Issues")


#Create an excel file to save the results
workbookResults= xlsxwriter.Workbook('Resultados.xlsx')
workSheetIssues = workbookResults.add_worksheet()
workSheetIssues.write('A1','Days')
workSheetIssues.write('B1','TN')
workSheetIssues.write('C1','FP')
workSheetIssues.write('D1','FN')
workSheetIssues.write('E1','TP')
workSheetIssues.write('F1','Precision')
workSheetIssues.write('G1','Accuracy')
workSheetIssues.write('H1','Recall')
workSheetIssues.write('I1','F1')



######Training method########
def train_model(classifier, X_TrainDataset, train_y, X_ValidDataset, valid_y, excelLine):  
    
    #Classifier training
    classifier.fit(X_TrainDataset, train_y)

    #Predicting classes
    predictions = classifier.predict(X_ValidDataset)
      
    #Calculating performance measures
    accuracy = metrics.accuracy_score(valid_y, predictions)
    precision = metrics.precision_score(valid_y, predictions)
    recall = metrics.recall_score(valid_y, predictions)
    f1 = metrics.f1_score(valid_y, predictions)
    tn, fp, fn, tp = confusion_matrix(valid_y,predictions, labels=[0,1]).ravel()
        
    #Writing results in excel file
    workSheetIssues.write('A'+str(excelLine),numMaxDays)
    workSheetIssues.write('B'+str(excelLine),tn)
    workSheetIssues.write('C'+str(excelLine),fp)
    workSheetIssues.write('D'+str(excelLine),fn)
    workSheetIssues.write('E'+str(excelLine),tp)
    workSheetIssues.write('F'+str(excelLine),precision)
    workSheetIssues.write('G'+str(excelLine),accuracy)
    workSheetIssues.write('H'+str(excelLine),recall)
    workSheetIssues.write('I'+str(excelLine),f1)
      
               



for numMaxDays in days:
    
    #Reading data from source file
    labels, texts = [], []
    for i in df.index:
        if(int(df['TimeComment'][i])<=numMaxDays):
            comment= str(df['CommentIssuePreprocessed'][i])
            label = str(df['Class'][i])
            texts.append(comment)
            labels.append(label)
                
    
    
    #Creating data frames
    trainDF = pd.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels
    
    #Split data sets into training and test
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], random_state=42)
    
    
    #coding data sets
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)
    
    #Creating vectors
    
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=max_features, stop_words='english')
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(ngram_range_min,ngram_range_max), max_features=max_features)#, stop_words='english')
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(ngram_range_min,ngram_range_max), max_features=max_features)
    
    #Join features
    TFIDF = FeatureUnion([
      #  ('tfidfWord', tfidf_vect)])
       ('tfidfNGram', tfidf_vect_ngram)])
        #('tfifdChar', tfidf_vect_ngram_chars)])
    
    
    xtrain_TFIDF =  TFIDF.fit_transform(train_x)
    xvalid_TFIDF =  TFIDF.transform(valid_x)
    
    xtrain_TFIDF_DF = pd.DataFrame(xtrain_TFIDF.toarray(), columns=TFIDF.get_feature_names())
    xvalid_TFIDF_DF = pd.DataFrame(xvalid_TFIDF.toarray(), columns=TFIDF.get_feature_names())
    
        
        
    #Selecting classifier
    classif =  naive_bayes.MultinomialNB().fit(xtrain_TFIDF_DF, train_y)
    #classif = linear_model.LogisticRegression()
    #classif = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    
    #Calling classification method
    train_model(classif, xtrain_TFIDF_DF, train_y, xvalid_TFIDF_DF, valid_y, int(excelLine))
    excelLine=excelLine+1
    
        

        
workbookResults.close()
