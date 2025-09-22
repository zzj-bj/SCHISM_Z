[GARVIS] stands for _Generative Assistant for Reliable Variables and INI Setup_. This application automatically generates configuration files (.ini) for training segmentation/classification models by combining:
- Dynamic retrieval of available parameters from the official documentation (.md files)
- Interpretation of the user’s natural language prompt (e.g., “I want 20 epochs and metrics: Jaccard & Recall”)
- Automatic generation of a coherent configuration file based on best practices (batch size, early stopping, etc.)
  
---
## :gear: Installation
1. Clone this repository to your local machine:
``` git clone git@github.com:FloFive/SCHISM.git ```
2. Navigate to the cloned directory via a terminal prompt: ```cd <some path> SCHISM/garvis```
3. Install the Streamlit library: ```pip install streamlit```

## 🧰 Prerequisites
1. Python 3.12.10
2. [Streamlit](https://streamlit.io/) (UI)
3. OpenAI API Key

## :question: How to use?

GARVIS offer a web interface to generate an INI file.

### General steps
1. Recover an OpenAI key to run the program. :warning: Be careful, as we are using a custom URL: Don't forget to adapt the code to your purpose to run the app. To adjust the API's URL, please check the ```openai_as_requests``` function from the ```processing.py``` file, and assign the right URL to the ```url``` variable.
2. Open a terminal prompt and navigate to the ```SCHISM/garvis``` folder, and run ```garvis.py```.
3. Answer all the questions in lay terms and plain English (and don't forget OpenAI API key!).
4. Click the "Generate INI" button.
5. And double-check the generated .ini file on the right-hand side.


## 🖼️ Preview
Here is an example of GARVIS running in Streamlit

<img width="772" height="1227" alt="image" src="https://github.com/user-attachments/assets/1eb132eb-3b91-474c-a67d-aa2433736b81"/>


## :scroll: .ini configuration file

Below is an example of an `.ini` configuration file. For detailed explanations of the network settings and the full INI specification, see the [INI file documentation](https://github.com/FloFive/SCHISM/blob/main/docs/ini.md).

```
[Model]
n_block=4
channels=8
num_classes=3
model_type=UnetSegmentor
k_size=3
activation=leakyrelu
 
[Optimizer]
optimizer=Adam
lr=0.01

[Scheduler]
scheduler = ConstantLR

[Loss]
loss= CrossEntropyLoss
ignore_background=True
weights=True

[Training]
batch_size=4
val_split=0.8
epochs=50
metrics=Jaccard, ConfusionMatrix
 
[Data]
crop_size=128
img_res=560
num_samples=7000
```
