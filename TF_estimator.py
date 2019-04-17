import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# estimator.LinearRegressor
feat_cols = [tf.feature_column.numeric_column("x", [1])]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_true,test_size = 0.3, random_state = 101)
input_funnc = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8, num_epochs=None,shuffle=True)
train_input_funnc = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8, num_epochs=1000,shuffle=False)
test_input_funnc = tf.estimator.inputs.numpy_input_fn({'x':x_test},y_test,batch_size=8, num_epochs=1000,shuffle=False)

estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

estimator.train(input_fn=input_funnc, steps=1000)
train_metrics = estimator.evaluate(input_fn=train_input_funnc, steps=1000)
test_metrics = estimator.evaluate(input_fn=test_input_funnc, steps=1000)

new_data = np.linspace(0,10,10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({"x":new_data}, shuffle=False)

prediction = []
for pred in estimator.predict(input_fn=input_fn_predict):
  prediction.append(pred["predictions"])

#categroical feature engineering
diabetes = pd.read_csv("pima-indians-diabetes.csv")
diabetes.columns
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x-x.min())/(x.max()-x.min()) )
num_preg = tf.feature_column.numeric_column("Number_pregnant")
plasma_gluc = tf.feature_column.numeric_column("Glucose_concentration")
dias_press = tf.feature_column.numeric_column("Blood_pressure")
tricep = tf.feature_column.numeric_column("Triceps")
insulin = tf.feature_column.numeric_column("Insulin")
bmi = tf.feature_column.numeric_column("BMI")
diabetes_pedigree = tf.feature_column.numeric_column("Pedigree")
age = tf.feature_column.numeric_column("Age")
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list("Group", ["A", 'B', "C", "D"])
# assigned_group = tf.feature_column.categorical_column_with_hash_bucket("Group", hash_bucket_size=10)
diabetes["Age"].hist(bins=20)
age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20, 30, 40, 50, 60, 70, 80])
x_data = diabetes.drop("Class", axis =1)
labels = diabetes["Class"]
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size = 0.3, random_state = 101)
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y = y_train, batch_size=10, num_epochs=1000, shuffle=True)
model  = tf.estimator.LinearClassifier(feature_columns=feature_cols, n_classes=2)
model.train(input_fn= input_func, steps=1000)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
result = model.evaluate(input_fn=eval_input_func)
predict_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
predictions = model.predict(input_fn=predict_input_func)

#DNN categorical feature columns must be wraped by embedding column
embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension=4)
feature_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, embedded_group_col, age_bucket]
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y = y_train, batch_size=10, num_epochs=1000, shuffle=True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,20, 10, 10], feature_columns=feature_cols, n_classes=2)
dnn_model.train(input_fn=input_func, steps=1000)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
result = model.evaluate(input_fn=eval_input_func)
