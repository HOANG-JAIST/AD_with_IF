
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[2]:


columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
          "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
          "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", 
           "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
          "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", 
          "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", 
           "dst_host_srv_count","dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
           "dst_host_same_src_port_rate", "dst_host_srv_dist_host_rate", "dst_host_serror_rate", 
           "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]


# ## Loading data

# In[3]:


df = pd.read_csv("../data/kddcup.data", sep=",", names=columns, index_col=None)


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


df.shape


# ## Filter out the entire data frame to only include data entries that involve an HTTP attact, and drop the service column.

# In[8]:


df = df[df["service"] == "http"]
df = df.drop("service", axis=1)
columns.remove("service")


# In[9]:


df.shape


# In[10]:


df["label"].value_counts()


# ## Label encoder
# 
# Some of the columns have categorical data values, meaning the model will have trouble training on them.

# In[11]:


df.head()


# In[12]:


print(df.head())


# In[13]:


for col in df.columns:
    if df[col].dtype == "object":
        encoded = LabelEncoder()
        encoded.fit(df[col]) # encoded.fit() gives the label encoder all of the data in the col
                             # from which it extracts the unique categorical values from
        df[col] = encoded.transform(df[col]) # assigning the encoded representation of 
                                             # each categorical value to df[col]


# In[14]:


df.head()


# ## Create training, validation, testing data

# In[16]:


for f in range(0, 3):
    df = df.iloc[np.random.permutation(len(df))] # randomly shuffling all the entries in the data
    # ...set to avoid the problem of abnormal entries pooling in any one region of the data set.
    
df2 = df[:500000] 
labels = df2["label"]
df_validate = df[500000:]

x_train, x_test, y_train, y_test = train_test_split(df2, labels, test_size=0.2, random_state=42)

x_val, y_val = df_validate, df_validate["label"]


# ## Define the term
# 
# . Training data: is the data that the model trains and learns on. 
#     - For an isolation forest, this set is what the model partitions on. 
#     - For neural networks, this set is what the model adjusts its weights on.
# 
# . Testing data: is the data that is used to test the model’s performance. The train_test_split() function basically splits the data into a portion used to train on and a portion used to test the model’s
# performance on.
# 
# . Validation data: is used during training to gauge how the model’s
# training is going. It basically helps ensure that as the model gets
# better at performing the task on the training data, it also gets better
# at performing the same task over new, but similar data. This way,
# the model doesn’t only get really good at performing the task on the
# training data, but can perform similarly on new data as well. In other
# words, you want to avoid overfitting, a situation where the model
# performs very well on a particular data set, which can be the training
# data set, yet the performance noticeably drops when new data is
# presented. A slight drop in performance is to be expected when the
# model is exposed to new variations in the data, but in this case, it is
# more pronounced.
# 
# Source: Book - Begining Abnomaly Detection (2019)

# In[17]:


print("Shapes: \nx_train:%s\ny_train:%s\n" % (x_train.shape, y_train.shape))
print("x_test:%s\ny_test:%s\n" % (x_test.shape, y_test.shape))
print("x_val:%s\ny_val:%s\n" % (x_val.shape, y_val.shape))


# # Buil Isolation Forest Model

# In[19]:


IF_model = IsolationForest(n_estimators=100, max_samples=256, contamination=0.1, random_state=42)


# In[20]:


IF_model.fit(x_train)


# ## Abnomaly Scores

# In[22]:


anomaly_scores = IF_model.decision_function(x_val)

plt.figure(figsize=(15, 10))
plt.hist(anomaly_scores, bins=100)
plt.xlabel('Average Path Lengths', fontsize=14)
plt.ylabel('Number of Data Point', fontsize=14)
plt.show()
plt.savefig("../figures/average_path_length_vs_number_of_datapoint.png")
# A histogram plotting the average path lengths for the data points. It helps you to determine
#what is an anomaly by using the shortest set of path lengths, since that indicates that the model
#was able to easily isolate those points


# ## AUC calculation

# In[23]:


from sklearn.metrics import roc_auc_score

anomalies = anomaly_scores > -0.19
matches = y_val == list(encoded.classes_).index("normal.")
auc = roc_auc_score(anomalies, matches)
print("AUC: {:.2%}".format (auc))


# In[24]:


anomaly_scores_test = IF_model.decision_function(x_test)

plt.figure(figsize=(15, 10))
plt.hist(anomaly_scores_test, bins=100)
plt.xlabel('Average Path Lengths', fontsize=14)
plt.ylabel('Number of Data Points', fontsize=14)
plt.show()


# In[25]:


anomalies_test = anomaly_scores_test > -0.19
matches = y_test == list(encoded.classes_).index("normal.")
auc = roc_auc_score(anomalies_test, matches)
print("AUC: {:.2%}".format (auc))


# # Conclusion
# 
# It seems to perform very well on both the validation and the test data.
# 
# In this lession, we gained a better understanding of what a isolation forest is and how to apply it. IF work well for multi-dimensional data, and can be used for unsupervised anomaly detection when applied in the manner implemented in this section.
