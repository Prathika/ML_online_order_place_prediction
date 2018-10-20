
# coding: utf-8

# In[1]:


# Read the CSV data using pandas
import pandas as pd
train_data = pd.read_csv("training_data.csv")
test_data = pd.read_csv("test_data .csv")


# In[2]:


# A glimpse of train data
train_data.describe(include='all').head()


# In[3]:


# A glimpse of test_data
test_data.describe(include='all')


# In[4]:


# Check the data count for both test and train data
print(train_data.shape)
print(test_data.shape)


# In[5]:


# Use seaborn to check make EDA(Exploratory Data Analysis)
import seaborn as sns
sns.countplot(data=train_data,x='checkout',hue='order_placed')


# In[6]:


#Dropping checkout field as it will not be accounted in the test data
train_data.drop(['checkout'],axis=1)


# In[7]:


sns.set_style('whitegrid')
sns.countplot(x='order_placed',data=train_data,palette='RdBu_r')
# This plot shows us that the order_placed is less than order_ not placed


# In[8]:


#Visualise data based on the funnel_level field
sns.set_style('whitegrid')
sns.countplot(x='order_placed',hue='funnel_level',data=train_data,palette='rainbow')


# In[9]:


# Get an idea of which value of the target attribute is dominant with the traindata
train_data.order_placed.value_counts()


# In[10]:


# Get field info of all train data
train_data.info()


# In[11]:


def trim_prefix(data):
    return data.split('grp')[1]


# In[12]:


# For futher model prediction we need the numeric data. Cleanse and tranform the grp field accordingly
train_data['grp'] = train_data['grp'].apply(trim_prefix)
test_data['grp'] = test_data['grp'].apply(trim_prefix)
train_data.head()


# In[13]:


test_data.head()


# In[14]:


# Cleansing the id field
def trim_session(data):
    return data.split('session')[1]
train_data['id'] = train_data['id'].apply(trim_session)
test_data['id'] = test_data['id'].apply(trim_session)


# In[15]:


# Susbtitue the value for categorical data
funnel = {'upper':1, 'middle':2, 'lower':3}

# X_train - Independet variables
# Y_train - Dependent or Target variables
# The group and funnel_level data seems to be a dominant contriutor, so using those fields alone 
# for priliminary predictions
train_data['funnel_level'] = train_data['funnel_level'].apply(lambda x: funnel.get(x))
X_train = train_data[['id','grp', 'funnel_level']]
Y_train = train_data['order_placed']


test_data = test_data[['id','grp', 'funnel_level']]
test_data['funnel_level'] = test_data['funnel_level'].apply(lambda x: funnel.get(x))
test_data.head()


# In[16]:


X_train.head()


# In[17]:


X_test = test_data
X_test.head()


# In[18]:


Y_train.head()


# In[31]:


def prepend_session(data):
    return 'session{}'.format(data)


# In[34]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, Y_train)
model.score(X_train, Y_train)

my_log_prediction = rf_model.predict(X_test)
my_log_solution = pd.DataFrame(my_log_prediction, X_test.id.apply(prepend_session), columns = ["order_placed"])
my_log_solution.to_csv("TG3_ML_solution_log_regression.csv", index_label = ["id"])


# In[42]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth=10,min_samples_split=2,n_estimators=100 , random_state=1 )
rf_model = forest.fit(X_train,Y_train)
rf_model.score(X_train, Y_train)
my_prediction = rf_model.predict(X_test)

my_solution = pd.DataFrame(my_prediction, X_test.id.apply(prepend_session), columns = ["order_placed"])

my_solution.to_csv("TG3_ML_solution_random_forest.csv", index_label = ["id"])

