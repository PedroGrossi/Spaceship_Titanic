import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def split_dataset(dataset, test_ratio=0.20):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]


print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)

# Load a dataset into a Pandas Dataframe
dataset_df = pd.read_csv('train.csv')
print("Full train dataset shape is {}".format(dataset_df.shape))

# Display the first 5 examples
print(dataset_df.head(5))

dataset_df.describe()
dataset_df.info()

plot_df = dataset_df.Transported.value_counts()
plot_df.plot(kind="bar")

fig, ax = plt.subplots(5,1,  figsize=(10, 10))
plt.subplots_adjust(top=2)

sns.histplot(dataset_df['Age'], color='b', bins=50, ax=ax[0])
sns.histplot(dataset_df['FoodCourt'], color='b', bins=50, ax=ax[1])
sns.histplot(dataset_df['ShoppingMall'], color='b', bins=50, ax=ax[2])
sns.histplot(dataset_df['Spa'], color='b', bins=50, ax=ax[3])
sns.histplot(dataset_df['VRDeck'], color='b', bins=50, ax=ax[4])

dataset_df = dataset_df.drop(['PassengerId', 'Name'], axis=1)
print(dataset_df.head(5))

dataset_df.isnull().sum().sort_values(ascending=False)

dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
dataset_df.isnull().sum().sort_values(ascending=False)

label = "Transported"
dataset_df[label] = dataset_df[label].astype(int)

dataset_df['VIP'] = dataset_df['VIP'].astype(int)
dataset_df['CryoSleep'] = dataset_df['CryoSleep'].astype(int)

dataset_df[["Deck", "Cabin_num", "Side"]] = dataset_df["Cabin"].str.split("/", expand=True)

try:
    dataset_df = dataset_df.drop('Cabin', axis=1)
except KeyError:
    print("Field does not exist")

print(dataset_df.head(5))

train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples in testing.".format(
    len(train_ds_pd), len(valid_ds_pd)))

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label)

tfdf.keras.get_all_models()

rf = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1")

rf = tfdf.keras.RandomForestModel()
rf.compile(metrics=["accuracy"]) # Optional, you can use this to include a list of eval metrics

rf.fit(x=train_ds)

tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)
