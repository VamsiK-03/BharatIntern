{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "dbdd6e55",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "%matplotlib inline\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "babbcf56",
   "metadata": {},
   "outputs": [],
   "source": [
    "wine_dataset = pd.read_csv('winequality-red.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "a3d4a579",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1599, 12)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wine_dataset.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b80be716",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fixed acidity</th>\n",
       "      <th>volatile acidity</th>\n",
       "      <th>citric acid</th>\n",
       "      <th>residual sugar</th>\n",
       "      <th>chlorides</th>\n",
       "      <th>free sulfur dioxide</th>\n",
       "      <th>total sulfur dioxide</th>\n",
       "      <th>density</th>\n",
       "      <th>pH</th>\n",
       "      <th>sulphates</th>\n",
       "      <th>alcohol</th>\n",
       "      <th>quality</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7.4</td>\n",
       "      <td>0.70</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.076</td>\n",
       "      <td>11.0</td>\n",
       "      <td>34.0</td>\n",
       "      <td>0.9978</td>\n",
       "      <td>3.51</td>\n",
       "      <td>0.56</td>\n",
       "      <td>9.4</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>7.8</td>\n",
       "      <td>0.88</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2.6</td>\n",
       "      <td>0.098</td>\n",
       "      <td>25.0</td>\n",
       "      <td>67.0</td>\n",
       "      <td>0.9968</td>\n",
       "      <td>3.20</td>\n",
       "      <td>0.68</td>\n",
       "      <td>9.8</td>\n",
       "      <td>5</td>\n",
Blaming machinelearning2/wine quality prediction.ipynb at main · kandagaddala13/machinelearning2
