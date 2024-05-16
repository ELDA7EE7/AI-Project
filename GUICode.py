import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_excel("water_potability.xlsx")
data['ph']=data['ph'].fillna(data.groupby(['Potability'])['ph'].transform('mean'))
data['Sulfate']=data['Sulfate'].fillna(data.groupby(['Potability'])['Sulfate'].transform('mean'))
data['Trihalomethanes']=data['Trihalomethanes'].fillna(data.groupby(['Potability'])['Trihalomethanes'].transform('mean'))
for col in data.columns:
  Q1 = data[col].quantile(0.25)
  Q3 = data[col].quantile(0.75)
  IQR = Q3 - Q1

  #remove outliers
  data = data[~((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))]

data.drop(columns=['Hardness','Conductivity'],inplace=True)

X = data.drop(columns=["Potability"])
y = data["Potability"]
#MinMax Scaler
min=[]
max=[]
for i in X.columns:
    min.append(X[i].min())
    max.append(X[i].max())
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 41)

LR_Model = LogisticRegression(max_iter=10000)
LR_Model.fit(X_train, y_train)
y_pred = LR_Model.predict(X_test)
logAccuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression ",logAccuracy*100)
DT_Model=DecisionTreeClassifier(criterion='gini',max_depth=10,max_features=None,min_samples_leaf=4,min_samples_split=5,splitter='best')
DT_Model.fit(X_train,y_train)
y_pred = DT_Model.predict(X_test)
desicionAccuracy=accuracy_score(y_test, y_pred)
print("Decision Tree ",desicionAccuracy*100)


svm_Model = SVC(C= 10, gamma = 1, kernel= 'rbf')

svm_Model.fit(X_train, y_train)
y_pred = svm_Model.predict(X_test)
svmAccuracy = accuracy_score(y_test, y_pred)
print("SVM ", svmAccuracy*100)
RF=RandomForestClassifier(random_state=42)
RF.fit(X_train, y_train)

# Make predictions on the test set
y_pred = RF.predict(X_test)
RFaccuracy=accuracy_score(y_test, y_pred)
print("RF ", RFaccuracy*100)

knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)
KnnAccuracy=accuracy_score(y_test, y_pred)
print("KNN ", KnnAccuracy*100)

import tkinter as tk
from tkinter import *
import tkinter.ttk as ttk

# Define the feature names
feature_names = ['ph', 'Solids', 'Chloramines', 'Sulfate', 'Organic_carbon',
                 'Trihalomethanes', 'Turbidity']


# Function to perform prediction based on the selected model
def predict():
    # Get user input from entry fields
    inputs = [float(entry.get()) for entry in entry_fields]
    j=0
    for i in inputs:
        tmp_max=max[j]
        tmp_min=min[j]
        if i<tmp_min:
           tmp_min=i
        if i>tmp_max:
            tmp_max=i
        i=(i-tmp_min)/(tmp_max-tmp_min)
        inputs[j]=i
        j+=1

    print(inputs)
    # Create a DataFrame from user inputs
    sample = pd.DataFrame([inputs], columns=feature_names)
    print(sample)
    # Perform prediction based on the selected model
    if model_var.get() == "SVM":
        prediction = svm_Model.predict(sample)
        accuracy=svmAccuracy
    elif model_var.get() == "DT":
        prediction = DT_Model.predict(sample)
        accuracy=desicionAccuracy
    elif model_var.get() == "LR":
        prediction = LR_Model.predict(sample)
        accuracy=logAccuracy
    elif model_var.get() == "RF":
        prediction = RF.predict(sample)
        accuracy=RFaccuracy
    elif model_var.get() == "KNN":
        prediction = knn.predict(sample)
        accuracy=KnnAccuracy
    else:
        prediction = None
        accuracy=0

    # Update the prediction result text and accuracy text after prediction
    if prediction is not None:
        update_prediction_text(prediction)
        update_accuracy_text(accuracy)


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("1920x1080")
window.configure(bg = "#1E1E1E")
window.title("Water Potability Predictor")
icon = PhotoImage(file=relative_to_assets("raindropsD.png"))
window.iconphoto(False, icon)

s = ttk.Style()
s.configure('Black.TRadiobutton',
        background="#1E1E1E",
        foreground='black')

canvas = Canvas(
    window,
    bg = "#1E1E1E",
    height = 1080,
    width = 1920,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)

predictButtonImage = PhotoImage(
    file=relative_to_assets("button_1.png"))
predictButton = Button(
    image=predictButtonImage,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_1 clicked"),
    relief="flat"
)
predictButton.place(
    x=20.0,
    y=910.0,
    width=1880.0,
    height=150.0
)

predictButtonHoverImage = PhotoImage(
    file=relative_to_assets("button_hover_1.png"))

def predictButtonHover(e):
    predictButton.config(
        image=predictButtonHoverImage
    )
def predictButtonLeave(e):
    predictButton.config(
        image=predictButtonImage
    )

predictButton.bind('<Enter>', predictButtonHover)
predictButton.bind('<Leave>', predictButtonLeave)
predictButton.config(command=predict)

entry_fields = []  # List to store references to entry fields


logoImage = PhotoImage(
    file=relative_to_assets("image_1.png"))
logoCanvas = canvas.create_image(
    602.5,
    64.0,
    image=logoImage
)

phImage = PhotoImage(
    file=relative_to_assets("image_2.png"))
phCanvas = canvas.create_image(
    306.0,
    173.0,
    image=phImage
)

phEntryImage = PhotoImage(
    file=relative_to_assets("entry_1.png"))
phEntryCanvas = canvas.create_image(
    306.25,
    231.5999984741211,
    image=phEntryImage
)
phEntry = Entry(
    bd=0,
    bg="#282828",
    fg="#F8F8F8",
    highlightthickness=0,
    font=("Inter", 20 * -1)
)
phEntry.place(
    x=30.0,
    y=207.0,
    width=552.5,
    height=47.19999694824219
)

hardnessImage = PhotoImage(
    file=relative_to_assets("image_3.png"))
hardnessCanvas = canvas.create_image(
    898.5,
    173.0,
    image=hardnessImage
)

hardnessEntryImage = PhotoImage(
    file=relative_to_assets("entry_2.png"))
hardnessEntryCanvas = canvas.create_image(
    898.75,
    231.5999984741211,
    image=hardnessEntryImage
)
hardnessEntry = Entry(
    bd=0,
    bg="#282828",
    fg="#F8F8F8",
    highlightthickness=0,
    font=("Inter", 20 * -1)
)
hardnessEntry.place(
    x=622.5,
    y=207.0,
    width=552.5,
    height=47.19999694824219
)

solidsImage = PhotoImage(
    file=relative_to_assets("image_4.png"))
solidsCanvas = canvas.create_image(
    306.0,
    300.20001220703125,
    image=solidsImage
)

solidsEntryImage = PhotoImage(
    file=relative_to_assets("entry_3.png"))
solidsEntryCanvas = canvas.create_image(
    306.25,
    358.80001068115234,
    image=solidsEntryImage
)
solidsEntry = Entry(
    bd=0,
    bg="#282828",
    fg="#F8F8F8",
    highlightthickness=0,
    font=("Inter", 20 * -1)
)
solidsEntry.place(
    x=30.0,
    y=334.20001220703125,
    width=552.5,
    height=47.19999694824219
)

chloraminesImage = PhotoImage(
    file=relative_to_assets("image_5.png"))
chloraminesCanvas = canvas.create_image(
    898.5,
    300.20001220703125,
    image=chloraminesImage
)

chloraminesEntryImage = PhotoImage(
    file=relative_to_assets("entry_4.png"))
chloraminesEntryCanvas = canvas.create_image(
    898.75,
    358.80001068115234,
    image=chloraminesEntryImage
)
chloraminesEntry = Entry(
    bd=0,
    bg="#282828",
    fg="#F8F8F8",
    highlightthickness=0,
    font=("Inter", 20 * -1)
)
chloraminesEntry.place(
    x=622.5,
    y=334.20001220703125,
    width=552.5,
    height=47.19999694824219
)

sulfateImage = PhotoImage(
    file=relative_to_assets("image_6.png"))
sulfateCanvas = canvas.create_image(
    306.0,
    427.3999938964844,
    image=sulfateImage
)

sulfateEntryImage = PhotoImage(
    file=relative_to_assets("entry_5.png"))
sulfateEntryCanvas = canvas.create_image(
    306.25,
    485.9999885559082,
    image=sulfateEntryImage
)
sulfateEntry = Entry(
    bd=0,
    bg="#282828",
    fg="#F8F8F8",
    highlightthickness=0,
    font=("Inter", 20 * -1)
)
sulfateEntry.place(
    x=30.0,
    y=461.3999938964844,
    width=552.5,
    height=47.199989318847656
)

conductivityImage = PhotoImage(
    file=relative_to_assets("image_7.png"))
conductivityCanvas = canvas.create_image(
    898.5,
    427.3999938964844,
    image=conductivityImage
)

conductivityEntryImage = PhotoImage(
    file=relative_to_assets("entry_6.png"))
conductivityEntryCanvas = canvas.create_image(
    898.75,
    485.9999885559082,
    image=conductivityEntryImage
)
conductivityEntry = Entry(
    bd=0,
    bg="#282828",
    fg="#F8F8F8",
    highlightthickness=0,
    font=("Inter", 20 * -1)
)
conductivityEntry.place(
    x=622.5,
    y=461.3999938964844,
    width=552.5,
    height=47.199989318847656
)

organicCarbonImage = PhotoImage(
    file=relative_to_assets("image_8.png"))
organicCarbonCanvas = canvas.create_image(
    306.0,
    554.5999755859375,
    image=organicCarbonImage
)

organicCarbonEntryImage = PhotoImage(
    file=relative_to_assets("entry_7.png"))
organicCarbonEntryCanvas = canvas.create_image(
    306.25,
    613.1999740600586,
    image=organicCarbonEntryImage
)
organicCarbonEntry = Entry(
    bd=0,
    bg="#282828",
    fg="#F8F8F8",
    highlightthickness=0,
    font=("Inter", 20 * -1)
)
organicCarbonEntry.place(
    x=30.0,
    y=588.5999755859375,
    width=552.5,
    height=47.19999694824219
)

trihalomethanesImage = PhotoImage(
    file=relative_to_assets("image_9.png"))
trihalomethanesCanvas = canvas.create_image(
    898.5,
    554.5999755859375,
    image=trihalomethanesImage
)

trihalomethanesEntryImage = PhotoImage(
    file=relative_to_assets("entry_8.png"))
trihalomethanesEntryCanvas = canvas.create_image(
    898.75,
    613.1999740600586,
    image=trihalomethanesEntryImage
)
trihalomethanesEntry = Entry(
    bd=0,
    bg="#282828",
    fg="#F8F8F8",
    highlightthickness=0,
    font=("Inter", 20 * -1)
)
trihalomethanesEntry.place(
    x=622.5,
    y=588.5999755859375,
    width=552.5,
    height=47.19999694824219
)

turbidityImage = PhotoImage(
    file=relative_to_assets("image_10.png"))
turbidityCanvas = canvas.create_image(
    602.0,
    681.7999877929688,
    image=turbidityImage
)

turbidityEntryImage = PhotoImage(
    file=relative_to_assets("entry_9.png"))
turbidityEntryCanvas = canvas.create_image(
    602.5,
    740.3999862670898,
    image=turbidityEntryImage
)
turbidityEntry = Entry(
    bd=0,
    bg="#282828",
    fg="#F8F8F8",
    highlightthickness=0,
    font=("Inter", 20 * -1)
)
turbidityEntry.place(
    x=30.0,
    y=715.7999877929688,
    width=1145.0,
    height=47.19999694824219
)

modelsSelectionImage = PhotoImage(
    file=relative_to_assets("image_11.png"))
modelSelectionCanvas = canvas.create_image(
    602.0,
    847.0,
    image=modelsSelectionImage
)

resultsPanelImage = PhotoImage(
    file=relative_to_assets("image_12.png"))
resultsPanelCanvas = canvas.create_image(
    1552.0,
    455.0,
    image=resultsPanelImage
)

canvas.create_text(
    1431.0,
    54.0,
    anchor="nw",
    text="Accuracy:",
    fill="#F7F7F7",
    font=("Inter ExtraBold", 48 * -1)
)

canvas.create_text(
    1431.5,
    473.0,
    anchor="nw",
    text="Potability:",
    fill="#F7F7F7",
    font=("Inter ExtraBold", 48 * -1)
)

potableImage = PhotoImage(
    file=relative_to_assets("image_13.png"))
notPotableImage = PhotoImage(
    file=relative_to_assets("image_14.png"))
potabilityResultCanvas = canvas.create_image(
    1552.0,
    718.0,
    image=notPotableImage
)

# Append entry fields to the list
entry_fields.append(phEntry)
#entry_fields.append(hardnessEntry)
entry_fields.append(solidsEntry)
entry_fields.append(chloraminesEntry)
entry_fields.append(sulfateEntry)
#entry_fields.append(conductivityEntry)
entry_fields.append(organicCarbonEntry)
entry_fields.append(trihalomethanesEntry)
entry_fields.append(turbidityEntry)

# Define the coordinates for each radio button
coordinates = [(85, 830), (327, 830), (610, 830), (839, 830), (1103, 830)]

# Create radio buttons for selecting the model
model_var = tk.StringVar()
model_var.set("SVM")  # Default selection
models = ["LR", "DT", "SVM", "RF", "KNN"]

for i, model in enumerate(models):
    radio_button = ttk.Radiobutton(window, variable=model_var, value=model, style = 'Black.TRadiobutton')
    # Place the radio button at the specified coordinates
    radio_button.place(x=coordinates[i][0], y=coordinates[i][1])


# Function to update the prediction result text
def update_prediction_text(prediction):
    canvas.itemconfig(potabilityResultCanvas, image=potableImage if prediction[0] == 1 else notPotableImage)

# Function to update the accuracy text
def update_accuracy_text(accuracy):
    canvas.itemconfig(accuracy_text, text=f"{accuracy*100:.2f}%")

# Define the coordinates for the prediction result text and accuracy text
accuracy_text_coordinates = (1226.5, 174.0)

# Create the accuracy text
accuracy_text = canvas.create_text(
    *accuracy_text_coordinates,
    anchor="nw",
    text="00.00%",
    fill="#F7F7F7",
    font=("Inter ExtraBold", 170 * -1)
)



window.resizable(False, False)
window.mainloop()
