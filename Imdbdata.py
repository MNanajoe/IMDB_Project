{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "09d208dd-d983-4c4e-817b-cd067538ec23",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import re\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.preprocessing.text import Tokenizer\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout,SimpleRNN, Reshape, Bidirectional\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "9955a748-847c-4200-a9b4-54c752e3b825",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"IMDB Dataset.csv\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "756390f1-4dff-4c5f-9b7a-919d2f74c2d6",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "2ab79bd7-65da-4baf-b26a-d0de63155e74",
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_text(text):\n",
    "    text = text.lower()\n",
    "    text = re.sub(r'<br\\s*/?>', ' ', text)  # Remove HTML tags\n",
    "    text = re.sub(r'[^a-z\\s]', '', text)  # Remove special characters and numbers\n",
    "    return text\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "0c97581e-c030-49a7-96d0-903a13ca1431",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df['cleaned_review'] = df['review'].apply(preprocess_text)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "35f99cb7-fbe8-42ea-a9d6-5f18105a28c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "vectorizer = TfidfVectorizer(max_features=5000)\n",
    "X_tfidf = vectorizer.fit_transform(df['cleaned_review'])\n",
    "y = df['sentiment']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "6bbeb88f-0781-4510-8cb6-142a952e83c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "3331c423-a860-4830-aee1-0816a718ac9f",
   "metadata": {},
   "outputs": [],
   "source": [
    "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "rf_model.fit(X_train, y_train)\n",
    "y_pred = rf_model.predict(X_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "34f95437-39d6-45e8-9bda-23c56ed20315",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Accuracy: 0.8535\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.84      0.86      0.85      4961\n",
      "           1       0.86      0.84      0.85      5039\n",
      "\n",
      "    accuracy                           0.85     10000\n",
      "   macro avg       0.85      0.85      0.85     10000\n",
      "weighted avg       0.85      0.85      0.85     10000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(\"Random Forest Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(classification_report(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "b8b5375f-eff0-466a-b690-86d534bc31ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "tokenizer = Tokenizer(num_words=5000, oov_token=\"<OOV>\")\n",
    "tokenizer.fit_on_texts(df['cleaned_review'])\n",
    "X_seq = tokenizer.texts_to_sequences(df['cleaned_review'])\n",
    "X_padded = pad_sequences(X_seq, maxlen=200, padding='post', truncating='post')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "aa1352f0-39e9-44bd-8f33-b2a40f3079cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "cf0865d5-83e1-40f9-9ae2-8aeb5bd3bd3e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "D:\\Anoconda\\Lib\\site-packages\\keras\\src\\layers\\core\\embedding.py:90: UserWarning: Argument `input_length` is deprecated. Just remove it.\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "model = Sequential([\n",
    "    Embedding(input_dim=5000, output_dim=64, input_length=200),\n",
    "    LSTM(64, return_sequences=True),\n",
    "    Dropout(0.2),\n",
    "     LSTM(32),\n",
    "    Dropout(0.2),\n",
    "    Dense(1, activation='sigmoid')\n",
    "])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "da9d75cd-46f0-4a89-9b91-6f7e281ff60a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "\u001b[1m1250/1250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m154s\u001b[0m 121ms/step - accuracy: 0.5123 - loss: 0.6918 - val_accuracy: 0.5454 - val_loss: 0.6889\n",
      "Epoch 2/5\n",
      "\u001b[1m1250/1250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m141s\u001b[0m 112ms/step - accuracy: 0.5737 - loss: 0.6726 - val_accuracy: 0.5937 - val_loss: 0.6570\n",
      "Epoch 3/5\n",
      "\u001b[1m1250/1250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m143s\u001b[0m 114ms/step - accuracy: 0.7191 - loss: 0.5809 - val_accuracy: 0.6696 - val_loss: 0.6108\n",
      "Epoch 4/5\n",
      "\u001b[1m1250/1250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m212s\u001b[0m 169ms/step - accuracy: 0.7603 - loss: 0.5095 - val_accuracy: 0.8527 - val_loss: 0.3465\n",
      "Epoch 5/5\n",
      "\u001b[1m1250/1250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m147s\u001b[0m 117ms/step - accuracy: 0.8665 - loss: 0.3271 - val_accuracy: 0.8521 - val_loss: 0.3295\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.history.History at 0x2ab8754c530>"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
    "model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "2b883dc7-18b0-4f41-94de-aa658bc501bb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m12s\u001b[0m 38ms/step - accuracy: 0.8563 - loss: 0.3248\n"
     ]
    }
   ],
   "source": [
    "\n",
    "test_loss, test_acc = model.evaluate(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "f3b43c21-e79b-4a4a-9926-7a539c727f07",
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "561ccf66-1405-4e1f-98db-ad994c6f8d75",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAk0AAAHFCAYAAADv8c1wAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA59ElEQVR4nO3de1xVdb7/8feOm6Kw5SK3CdE6ShpaEzYKZmoqeEEzLeswh7QxtPEWKTU/u2mdRktLraymy4ya5eBMpVOjB8W8lCOaUWSUOVo4moJ4wY2aAuL390eHddqCukAIcF7Px2M9Hq7v+uy1Pou9d7z7rrU3DmOMEQAAAC7oioZuAAAAoCkgNAEAANhAaAIAALCB0AQAAGADoQkAAMAGQhMAAIANhCYAAAAbCE0AAAA2EJoAAABsIDQB9WDr1q267bbb1KZNG/n4+Cg0NFRxcXGaOnVqvR73hx9+0IwZM7Rhw4Yq2xYtWiSHw6E9e/bUaw+XaunSpZo/f36NHnP27FktWbJE/fr1U3BwsLy8vBQSEqKkpCR98MEHOnv2rCRpz549cjgcWrRoUd03fgkqn5vKpVmzZgoLC1OfPn00a9YsFRUVVXnMjBkz5HA4anScC70+LqS6Y7Vt21ZJSUk12s/FXOi5dzgcmjFjRp0eD6gpQhNQx1auXKn4+HiVlJRo9uzZWrNmjZ5//nn16NFDy5Ytq9dj//DDD3riiSeq/aU4ePBgZWdnKzw8vF57uFQ1DU2nT5/WoEGDNGrUKIWEhOiVV17RunXr9Ic//EERERG644479MEHH9Rfw3Vo4cKFys7OVlZWll566SVdf/31euaZZ9SxY0etXbvWrfbee+9VdnZ2jfZ/odfHhdTmWLVxoec+Oztb9957b733AFyIZ0M3AFxuZs+erXbt2mn16tXy9Py/t9hdd92l2bNnN1hfrVu3VuvWrRvs+PVlypQpWr16tRYvXqy7777bbdvw4cP14IMP6tSpUw3UXc3ExMSoa9eu1vqIESP0wAMP6KabbtLw4cO1a9cuhYaGSpKuvPJKXXnllfXazw8//CBfX9+f5VgX07179wY9PiAx0wTUuSNHjig4ONgtMFW64oqqb7lly5YpLi5OLVq0UMuWLZWYmKjPP//crWb06NFq2bKldu/erUGDBqlly5aKjIzU1KlTVVpaKunHS0+VoeiJJ56wLvWMHj1aUvWX53r37q2YmBhlZ2crPj5ezZs3V9u2bbVw4UJJP86a3XDDDfL19VXnzp2VmZlZpf9du3YpOTlZISEh8vHxUceOHfXSSy+51WzYsEEOh0N//vOf9cgjjygiIkL+/v7q16+fdu7c6dbPypUr9a9//cvtctX5FBYW6o033lBiYmKVwFSpffv26tKly3n3sXv3bt1zzz1q3769fH199Ytf/EJDhgzRl19+6VZ39uxZPfXUU4qOjlbz5s3VqlUrdenSRc8//7xVc+jQIY0dO1aRkZHy8fFR69at1aNHjyqzRDXRpk0bPffcczp+/LheffVVa7y6S2br1q1T7969FRQUpObNm6tNmzYaMWKEfvjhh4u+Pir399lnn+n2229XQECArr766vMeq9Ly5cvVpUsXNWvWTFdddZVeeOEFt+3nuyxc+ZqonPW62HNf3eW5vLw83XrrrQoICFCzZs10/fXXa/HixdUe52KvPcAOZpqAOhYXF6c33nhDkydP1q9//WvdcMMN8vLyqrZ25syZevTRR3XPPffo0UcfVVlZmebMmaOePXvqk08+UadOnaza8vJyDR06VGPGjNHUqVP10Ucf6b//+7/ldDr1+OOPKzw8XJmZmRowYIDGjBljXcq42OxSYWGh7rnnHj300EO68sor9eKLL+o3v/mN9u3bp3feeUcPP/ywnE6nnnzySQ0bNkzfffedIiIiJElff/214uPjrV/sYWFhWr16tSZPnqzDhw9r+vTpbsd6+OGH1aNHD73xxhsqKSnR7373Ow0ZMkQ7duyQh4eHXn75ZY0dO1bffvutli9fftGf9fr161VeXq5hw4ZdtPZ8Dhw4oKCgID399NNq3bq1jh49qsWLF6tbt276/PPPFR0dLenHGcQZM2bo0Ucf1c0336zy8nJ98803OnbsmLWvlJQUffbZZ/r973+vDh066NixY/rss8905MiRWvcnSYMGDZKHh4c++uij89bs2bNHgwcPVs+ePfWnP/1JrVq10v79+5WZmamysjLbr4/hw4frrrvu0n333aeTJ09esK/c3FylpaVpxowZCgsL09tvv637779fZWVlSk9Pr9E51vS537lzp+Lj4xUSEqIXXnhBQUFBeuuttzR69GgdPHhQDz30kFv9xV57gC0GQJ06fPiwuemmm4wkI8l4eXmZ+Ph4M2vWLHP8+HGrbu/evcbT09NMmjTJ7fHHjx83YWFhZuTIkdbYqFGjjCTzl7/8xa120KBBJjo62lo/dOiQkWSmT59epa+FCxcaSSY/P98a69Wrl5FkPv30U2vsyJEjxsPDwzRv3tzs37/fGs/NzTWSzAsvvGCNJSYmmiuvvNK4XC63Y02cONE0a9bMHD161BhjzPr1640kM2jQILe6v/zlL0aSyc7OtsYGDx5soqKiqvRfnaefftpIMpmZmbbq8/PzjSSzcOHC89acOXPGlJWVmfbt25sHHnjAGk9KSjLXX3/9BfffsmVLk5aWZquXn6p8brZt23bemtDQUNOxY0drffr06ean/wl/5513jCSTm5t73n1c6PVRub/HH3/8vNt+KioqyjgcjirH69+/v/H39zcnT550O7efvu6M+b/XxPr1662xCz335/Z91113GR8fH7N37163uoEDBxpfX19z7Ngxt+PYee0BF8PlOaCOBQUF6eOPP9a2bdv09NNP69Zbb9U///lPTZs2TZ07d9bhw4clSatXr9aZM2d0991368yZM9bSrFkz9erVq8rNug6HQ0OGDHEb69Kli/71r39dUr/h4eGKjY211gMDAxUSEqLrr7/emlGSpI4dO0qSdbzTp0/rww8/1G233SZfX1+3cxg0aJBOnz6tLVu2uB1r6NChVfr/6T4bwpkzZzRz5kx16tRJ3t7e8vT0lLe3t3bt2qUdO3ZYdb/61a/0xRdfaPz48Vq9erVKSkqq7OtXv/qVFi1apKeeekpbtmxReXl5nfVpjLng9uuvv17e3t4aO3asFi9erO+++65WxxkxYoTt2muvvVbXXXed21hycrJKSkr02Wef1er4dq1bt059+/ZVZGSk2/jo0aP1ww8/VLlxvTG+9tD0EJqAetK1a1f97ne/01//+lcdOHBADzzwgPbs2WPdDH7w4EFJ0o033igvLy+3ZdmyZVa4quTr66tmzZq5jfn4+Oj06dOX1GdgYGCVMW9v7yrj3t7ekmQd78iRIzpz5oxefPHFKv0PGjRIkqqcQ1BQUJX+JdX6Ru02bdpIkvLz82v1eOnHG8kfe+wxDRs2TB988IG2bt2qbdu26brrrnPra9q0aXr22We1ZcsWDRw4UEFBQerbt68+/fRTq2bZsmUaNWqU3njjDcXFxSkwMFB33323CgsLa92fJJ08eVJHjhxxC7Hnuvrqq7V27VqFhIRowoQJuvrqq3X11Ve73XNlR00+XRkWFnbesUu9JHkxR44cqbbXyp/Rucev69ce/j1xTxPwM/Dy8tL06dM1b9485eXlSZKCg4MlSe+8846ioqIasr1aCQgIkIeHh1JSUjRhwoRqa9q1a1evPfTp00deXl5asWKF7rvvvlrt46233tLdd9+tmTNnuo0fPnxYrVq1stY9PT01ZcoUTZkyRceOHdPatWv18MMPKzExUfv27ZOvr6+Cg4M1f/58zZ8/X3v37tX777+v//f//p+KioqqvYnerpUrV6qiokK9e/e+YF3Pnj3Vs2dPVVRU6NNPP9WLL76otLQ0hYaG6q677rJ1rJp891N1YbByrDKkVAb9yg8sVDo3UNdUUFCQCgoKqowfOHBA0v+9v4C6xEwTUMeq+w+5JOtST+X/CScmJsrT01PffvutunbtWu1SUz/n/z37+vqqT58++vzzz9WlS5dq+z/3/+7t8PHxsd1/WFiY7r33Xq1evVpvvvlmtTXffvuttm/fft59OBwO6+dWaeXKldq/f/95H9OqVSvdfvvtmjBhgo4ePVrtF4a2adNGEydOVP/+/S/pUtXevXuVnp4up9OpcePG2XqMh4eHunXrZn2KsfL4df36+Oqrr/TFF1+4jS1dulR+fn664YYbJP34JZiSqjwH77//fpX91eS579u3r9atW2eFpEpvvvmmfH19+YoC1AtmmoA6lpiYqCuvvFJDhgzRNddco7Nnzyo3N1fPPfecWrZsqfvvv1/Sj79MnnzyST3yyCP67rvvNGDAAAUEBOjgwYP65JNP1KJFCz3xxBM1Orafn5+ioqL0t7/9TX379lVgYKCCg4OtX1x17fnnn9dNN92knj176re//a3atm2r48ePa/fu3frggw+0bt26Gu+zc+fOeu+99/TKK68oNjZWV1xxxQUD5Ny5c/Xdd99p9OjRWr16tW677TaFhobq8OHDysrK0sKFC5WRkXHerx1ISkrSokWLdM0116hLly7KycnRnDlzqnwv0ZAhQ6zvUWrdurX+9a9/af78+YqKilL79u3lcrnUp08fJScn65prrpGfn5+2bdumzMxMDR8+3Na55+XlWfeFFRUV6eOPP9bChQvl4eGh5cuXX/CTkH/4wx+0bt06DR48WG3atNHp06f1pz/9SZLUr18/SXX/+oiIiNDQoUM1Y8YM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AAAuB7UKTfPmzat2fMaMGTpx4sQlNQQAANAY1ek9Tf/1X//F350DAACXpToNTdnZ2WrWrFld7hIAAKBRqNXlueHDh7utG2NUUFCgTz/9VI899lidNAYAANCY1Co0OZ1Ot/UrrrhC0dHRevLJJ5WQkFAnjQEAADQmtQpNCxcurOs+AAAAGrVahaZKOTk52rFjhxwOhzp16qRf/vKXddUXAABAo1Kr0FRUVKS77rpLGzZsUKtWrWSMkcvlUp8+fZSRkaHWrVvXdZ8AAAANqlafnps0aZJKSkr01Vdf6ejRoyouLlZeXp5KSko0efLkuu4RAACgwdVqpikzM1Nr165Vx44drbFOnTrppZde4kZwAABwWarVTNPZs2fl5eVVZdzLy0tnz5695KYAAAAam1qFpltuuUX333+/Dhw4YI3t379fDzzwgPr27VtnzQEAADQWtQpNCxYs0PHjx9W2bVtdffXV+o//+A+1a9dOx48f14svvljXPQIAADS4Wt3TFBkZqc8++0xZWVn65ptvZIxRp06d1K9fv7ruDwAAoFGo0UzTunXr1KlTJ5WUlEiS+vfvr0mTJmny5Mm68cYbde211+rjjz+ul0YBAAAaUo1C0/z585Wamip/f/8q25xOp8aNG6e5c+fWWXMAAACNRY1C0xdffKEBAwacd3tCQoJycnIuuSkAAIDGpkah6eDBg9V+1UAlT09PHTp06JKbAgAAaGxqFJp+8Ytf6Msvvzzv9u3btys8PPySmwIAAGhsahSaBg0apMcff1ynT5+usu3UqVOaPn26kpKS6qw5AACAxqJGXznw6KOP6r333lOHDh00ceJERUdHy+FwaMeOHXrppZdUUVGhRx55pL56BQAAaDA1Ck2hoaHavHmzfvvb32ratGkyxkiSHA6HEhMT9fLLLys0NLReGgUAAGhINf5yy6ioKK1atUrFxcXavXu3jDFq3769AgIC6qM/AACARqFW3wguSQEBAbrxxhvrshcAAIBGq1Z/ew4AAODfDaEJAADABkITAACADYQmAAAAGwhNAAAANhCaAAAAbCA0AQAA2EBoAgAAsIHQBAAAYAOhCQAAwAZCEwAAgA2EJgAAABsITQAAADYQmgAAAGwgNAEAANhAaAIAALCB0AQAAGBDg4amjz76SEOGDFFERIQcDodWrFjhtt0YoxkzZigiIkLNmzdX7969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emhoqNq2bauWLVu6jVXu+7PPPpMxRh06dHDbT2lpqVvP5+63tv0BqD+EJgBNQrNmzdS/f3/1799fjz/+uO69915Nnz5d48ePV3h4uDZs2FDlMa1atbL+7eXl5bbN4XDo7NmzNe6juv1caN9nz56Vh4eHcnJy5OHh4Vb306BV3T6MMTXuD0D9ITQBaJI6deqkFStW6IYbblBhYaE8PT3Vtm3bWu/P29tbFRUVddfg//rlL3+piooKFRUVqWfPnrXeT331B8A+bgQH0KgdOXJEt9xyi9566y1t375d+fn5+utf/6rZs2fr1ltvVb9+/RQXF6dhw4Zp9erV2rNnjzZv3qxHH31Un376qe3jtG3bVlu3btWePXt0+PDhWs1CVadDhw769a9/rbvvvlvvvfee8vPztW3bNj3zzDNatWpVjfrbvn27du7cqcOHD6u8vLxO+gNgH6EJQKPWsmVLdevWTfPmzdPNN9+smJgYPfbYY0pNTdWCBQvkcDi0atUq3XzzzfrNb36jDh066K677tKePXsUGhpq+zjp6eny8PBQp06d1Lp1a+3du7fOzmHhwoW6++67NXXqVEVHR2vo0KHaunXreT+hV53U1FRFR0era9euat26tf7xj3/UWX8A7HEYLpoDAABcFDNNAAAANhCaAAAAbCA0AQAA2EBoAgAAsIHQBAAAYAOhCQAAwAZCEwAAgA2EJgAAABsITQAAADYQmgAAAGwgNAEAANhAaAIAALDh/wP7ZBAWfSElxgAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x=df['sentiment'])\n",
    "plt.title(\"Sentiment Class Distribution\")\n",
    "plt.xlabel(\"Sentiment\")\n",
    "plt.ylabel(\"Count\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "77dc0dbc-46c0-48cd-8165-97e3ed3579cf",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    }
   ],
   "source": [
    "# Save RNN Model and Tokenizer\n",
    "model.save(\"rnn_model.h5\")\n",
    "with open('tokenizer.pkl', 'wb') as f:\n",
    "    pickle.dump(tokenizer, f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "9b2c09ff-6b4a-49f3-94ca-632f5531f43f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "D:\\Anoconda\\Lib\\site-packages\\keras\\src\\layers\\core\\embedding.py:90: UserWarning: Argument `input_length` is deprecated. Just remove it.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1250/1250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m85s\u001b[0m 66ms/step - accuracy: 0.4986 - loss: 0.7194 - val_accuracy: 0.5039 - val_loss: 0.6931\n",
      "Epoch 2/5\n",
      "\u001b[1m1250/1250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m80s\u001b[0m 64ms/step - accuracy: 0.5035 - loss: 0.6980 - val_accuracy: 0.5099 - val_loss: 0.6933\n",
      "Epoch 3/5\n",
      "\u001b[1m1250/1250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m77s\u001b[0m 62ms/step - accuracy: 0.4991 - loss: 0.6945 - val_accuracy: 0.4917 - val_loss: 0.6943\n",
      "Epoch 4/5\n",
      "\u001b[1m1250/1250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m77s\u001b[0m 62ms/step - accuracy: 0.5044 - loss: 0.6935 - val_accuracy: 0.4936 - val_loss: 0.6935\n",
      "Epoch 5/5\n",
      "\u001b[1m1250/1250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m77s\u001b[0m 62ms/step - accuracy: 0.5006 - loss: 0.6937 - val_accuracy: 0.4953 - val_loss: 0.6948\n",
      "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 16ms/step - accuracy: 0.5010 - loss: 0.6943\n",
      "RNN Model Accuracy: 0.4952999949455261\n"
     ]
    }
   ],
   "source": [
    "tokenizer = Tokenizer(num_words=5000, oov_token=\"<OOV>\")\n",
    "tokenizer.fit_on_texts(df['cleaned_review'])\n",
    "X_seq = tokenizer.texts_to_sequences(df['cleaned_review'])\n",
    "X_padded = pad_sequences(X_seq, maxlen=200, padding='post', truncating='post')\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)\n",
    "\n",
    "model = Sequential([\n",
    "    Embedding(input_dim=5000, output_dim=64, input_length=200),\n",
    "    SimpleRNN(64, return_sequences=True),\n",
    "    Dropout(0.2),\n",
    "    SimpleRNN(32),\n",
    "    Dropout(0.2),\n",
    "    Dense(1, activation='sigmoid')\n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
    "history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))\n",
    "\n",
    "test_loss, test_acc = model.evaluate(X_test, y_test)\n",
    "print(\"RNN Model Accuracy:\", test_acc)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "dd1157f3-6196-4267-923b-4c89dad4bcbe",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "D:\\Anoconda\\Lib\\site-packages\\keras\\src\\layers\\reshaping\\reshape.py:39: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(**kwargs)\n"
     ]
    }
   ],
   "source": [
    "\n",
    "nn_model = Sequential ([Reshape((X_train.shape[1],1), input_shape = (X_train.shape[1],)),\n",
    "                        SimpleRNN (128, return_sequences=False),\n",
    "                        Dropout(0.5),\n",
    "                        Dense(1, activation=\"sigmoid\")])\n",
    "\n",
    "\n",
    "\n",
    "# Build LSTM\n",
    "\n",
    "Lstm_model = Sequential([Reshape((X_train.shape[1],1), input_shape = (X_train.shape[1],)),\n",
    "                        LSTM(128, return_sequences=False),\n",
    "                        Dropout(0.5),\n",
    "                        Dense(1, activation=\"sigmoid\")])\n",
    "\n",
    "#Bidirectional LSTM\n",
    "\n",
    "bi1stm_model = Sequential ([Reshape((X_train.shape[1],1), input_shape=(X_train.shape[1],)),\n",
    "                        Bidirectional(LSTM(128, return_sequences=False)),\n",
    "                        Dropout (0.5),\n",
    "                        Dense(1, activation=\"sigmoid\")])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "id": "a281f6d4-9832-4a0f-91d6-e3bfb1f3846c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1250/1250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m91s\u001b[0m 70ms/step - accuracy: 0.5833 - loss: 0.6704 - val_accuracy: 0.5372 - val_loss: 0.6889\n",
      "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 19ms/step\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.52      0.76      0.62      4961\n",
      "           1       0.57      0.31      0.41      5039\n",
      "\n",
      "    accuracy                           0.54     10000\n",
      "   macro avg       0.55      0.54      0.51     10000\n",
      "weighted avg       0.55      0.54      0.51     10000\n",
      "\n",
      "\u001b[1m1250/1250\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m153s\u001b[0m 121ms/step - accuracy: 0.5109 - loss: 0.7102 - val_accuracy: 0.5142 - val_loss: 0.6926\n",
      "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 46ms/step\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.51      0.94      0.66      4961\n",
      "           1       0.61      0.10      0.17      5039\n",
      "\n",
      "    accuracy                           0.51     10000\n",
      "   macro avg       0.56      0.52      0.41     10000\n",
      "weighted avg       0.56      0.51      0.41     10000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "model.compile(optimizer = \"adam\", loss= \"binary_crossentropy\", metrics=[\"accuracy\"])\n",
    "model.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test), verbose=1)\n",
    "\n",
    "pred=model.predict(X_test)\n",
    "pred=(pred>0.5).astype(int)\n",
    "\n",
    "print(classification_report(y_test, pred))\n",
    "\n",
    "\n",
    "#Evaluation metrics for Bilstm model\n",
    "\n",
    "model=bi1stm_model\n",
    "model.compile(optimizer = \"adam\", loss= \"binary_crossentropy\", metrics=[\"accuracy\"])\n",
    "model.fit(X_train,y_train, batch_size=32, validation_data=(X_test, y_test), verbose=1)\n",
    "\n",
    "pred=model.predict(X_test)\n",
    "pred=(pred>0.5).astype(int)\n",
    "\n",
    "print(classification_report(y_test, pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "id": "86b6d195-c81d-44c9-a277-c24aeab39572",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting fasttext\n",
      "  Using cached fasttext-0.9.3.tar.gz (73 kB)\n",
      "  Installing build dependencies: started\n",
      "  Installing build dependencies: finished with status 'done'\n",
      "  Getting requirements to build wheel: started\n",
      "  Getting requirements to build wheel: finished with status 'done'\n",
      "  Preparing metadata (pyproject.toml): started\n",
      "  Preparing metadata (pyproject.toml): finished with status 'done'\n",
      "Collecting pybind11>=2.2 (from fasttext)\n",
      "  Using cached pybind11-2.13.6-py3-none-any.whl.metadata (9.5 kB)\n",
      "Requirement already satisfied: setuptools>=0.7.0 in d:\\anoconda\\lib\\site-packages (from fasttext) (75.1.0)\n",
      "Requirement already satisfied: numpy in d:\\anoconda\\lib\\site-packages (from fasttext) (1.26.4)\n",
      "Using cached pybind11-2.13.6-py3-none-any.whl (243 kB)\n",
      "Building wheels for collected packages: fasttext\n",
      "  Building wheel for fasttext (pyproject.toml): started\n",
      "  Building wheel for fasttext (pyproject.toml): finished with status 'error'\n",
      "Failed to build fasttext\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  error: subprocess-exited-with-error\n",
      "  \n",
      "  Building wheel for fasttext (pyproject.toml) did not run successfully.\n",
      "  exit code: 1\n",
      "  \n",
      "  [31 lines of output]\n",
      "  C:\\Users\\DELL\\AppData\\Local\\Temp\\pip-build-env-xuuzk4qy\\overlay\\Lib\\site-packages\\setuptools\\dist.py:493: SetuptoolsDeprecationWarning: Invalid dash-separated options\n",
      "  !!\n",
      "  \n",
      "          ********************************************************************************\n",
      "          Usage of dash-separated 'description-file' will not be supported in future\n",
      "          versions. Please use the underscore name 'description_file' instead.\n",
      "  \n",
      "          By 2025-Mar-03, you need to update your project and remove deprecated calls\n",
      "          or your builds will no longer be supported.\n",
      "  \n",
      "          See https://setuptools.pypa.io/en/latest/userguide/declarative_config.html for details.\n",
      "          ********************************************************************************\n",
      "  \n",
      "  !!\n",
      "    opt = self.warn_dash_deprecation(opt, section)\n",
      "  running bdist_wheel\n",
      "  running build\n",
      "  running build_py\n",
      "  creating build\\lib.win-amd64-cpython-312\\fasttext\n",
      "  copying python\\fasttext_module\\fasttext\\FastText.py -> build\\lib.win-amd64-cpython-312\\fasttext\n",
      "  copying python\\fasttext_module\\fasttext\\__init__.py -> build\\lib.win-amd64-cpython-312\\fasttext\n",
      "  creating build\\lib.win-amd64-cpython-312\\fasttext\\util\n",
      "  copying python\\fasttext_module\\fasttext\\util\\util.py -> build\\lib.win-amd64-cpython-312\\fasttext\\util\n",
      "  copying python\\fasttext_module\\fasttext\\util\\__init__.py -> build\\lib.win-amd64-cpython-312\\fasttext\\util\n",
      "  creating build\\lib.win-amd64-cpython-312\\fasttext\\tests\n",
      "  copying python\\fasttext_module\\fasttext\\tests\\test_configurations.py -> build\\lib.win-amd64-cpython-312\\fasttext\\tests\n",
      "  copying python\\fasttext_module\\fasttext\\tests\\test_script.py -> build\\lib.win-amd64-cpython-312\\fasttext\\tests\n",
      "  copying python\\fasttext_module\\fasttext\\tests\\__init__.py -> build\\lib.win-amd64-cpython-312\\fasttext\\tests\n",
      "  running build_ext\n",
      "  building 'fasttext_pybind' extension\n",
      "  error: Microsoft Visual C++ 14.0 or greater is required. Get it with \"Microsoft C++ Build Tools\": https://visualstudio.microsoft.com/visual-cpp-build-tools/\n",
      "  [end of output]\n",
      "  \n",
      "  note: This error originates from a subprocess, and is likely not a problem with pip.\n",
      "  ERROR: Failed building wheel for fasttext\n",
      "ERROR: ERROR: Failed to build installable wheels for some pyproject.toml based projects (fasttext)\n"
     ]
    },
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'fasttext'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[148], line 11\u001b[0m\n\u001b[0;32m      5\u001b[0m df_filter \u001b[38;5;241m=\u001b[39m pd\u001b[38;5;241m.\u001b[39mconcat([df_p, df_n], axis \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m1\u001b[39m)\n\u001b[0;32m      7\u001b[0m get_ipython()\u001b[38;5;241m.\u001b[39msystem(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mpip install fasttext\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m---> 11\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mfasttext\u001b[39;00m\n\u001b[0;32m     13\u001b[0m model\u001b[38;5;241m=\u001b[39mfasttext\u001b[38;5;241m.\u001b[39mtrain_supervised(\u001b[38;5;28minput\u001b[39m\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mimdb_train.txt\u001b[39m\u001b[38;5;124m\"\u001b[39m, epoch\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m25\u001b[39m, lr\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1.0\u001b[39m, wordNgrams\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m2\u001b[39m, dim\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m100\u001b[39m)\n\u001b[0;32m     15\u001b[0m model\u001b[38;5;241m.\u001b[39msave_model(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mimdb_sentiment_model.ftz\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'fasttext'"
     ]
    }
   ],
   "source": [
    "df_p = df[df[\"sentiment\" ]==\"positive\"].iloc[0:2500]\n",
    "\n",
    "df_n = df[df[\"sentiment\"]==\"negative\"].iloc[0:2500]\n",
    "\n",
    "df_filter = pd.concat([df_p, df_n], axis = 1)\n",
    "\n",
    "!pip install fasttext\n",
    "\n",
    "\n",
    "\n",
    "import fasttext\n",
    "\n",
    "model=fasttext.train_supervised(input=\"imdb_train.txt\", epoch=25, lr=1.0, wordNgrams=2, dim=100)\n",
    "\n",
    "model.save_model(\"imdb_sentiment_model.ftz\")\n",
    "\n",
    "print(\"Model trained and saved!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f1fcd93b-3825-4f4c-8005-a9f6c971a5e5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6fe48af5-371d-464c-ba3b-a3ef1f1bd852",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
