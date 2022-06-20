from metaflow import FlowSpec, step, IncludeFile
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

class TextClassification(FlowSpec):

    @step
    def start(self):
        self.next(self.load)

    @step
    def load(self):
        train = pd.read_csv('treino.csv')
        test = pd.read_csv('teste.csv')

        self.x_train = train['texto']
        self.y_train = train['classificacao']

        self.x_test = test['texto']
        self.y_test = test['classificacao']


        self.next(self.preprocessing)

    @step
    def preprocessing(self):
        self.scoring = {
            'f1_micro',
            'f1_macro',
            'f1_weighted'
        }

        self.pipeline = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])

        self.parameters = {
            'vect__min_df': [1, 2, 3],
            'vect__smooth_idf': [True],
            'vect__ngram_range': [(1, 1), (1, 2), (1, 3)]
        }

        self.next(self.train)

    @step
    def train(self):
        grid_search = GridSearchCV(estimator=self.pipeline, param_grid=self.parameters, n_jobs=-1, verbose=1, scoring=self.scoring, refit='f1_micro')
        grid_search.fit(self.x_train, self.y_train)

        print("Best parameters:")
        print(grid_search.best_params_)

        print("Best scorers: ")
        print(grid_search.best_score_)

        self.tfidf_naive = grid_search.best_estimator_
        

        self.next(self.test)
    
    @step
    def test(self):
        print('F1-Score (micro) Test: ', f1_score(self.y_test, self.tfidf_naive.predict(self.x_test), average='micro'))
        
        self.next(self.end)
   
    @step
    def end(self):
        pass

if __name__ == '__main__':
    TextClassification()