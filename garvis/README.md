[GARVIS] stands for _Generative Assistant for Reliable Variables and INI Setup_. This application automatically generates configuration files (.ini) for training segmentation/classification models by combining:
- Interpretation of the user’s natural language prompt (e.g., “I want 20 epochs and metrics: Jaccard & Recall”)
- Automatic generation of a coherent configuration file based on best practices (batch size, early stopping, etc.)
  
---
## :gear: Installation
1. Clone this repository to your local machine: git clone `git@github.com:FloFive/SCHISM.git` 
2. Navigate to the cloned directory via a terminal prompt: `cd <some path> SCHISM/garvis`
3. Install the Streamlit library: `pip install streamlit`

## 🧰 Prerequisites
1. Python 3.12.10
2. [Streamlit](https://streamlit.io/) (UI)
3. OpenAI API Key

## :question: How to use

GARVIS offer a web interface to generate an INI file.

### General steps
1. Recover an OpenAI key to run the program. :warning: Be careful, as we are using a custom URL: Don't forget to adapt the code to your purpose to run the app. To adjust the API's URL, please check the `openai_as_requests` function from the `processing.py` file, and assign the right URL to the `url` variable.
2. Open a terminal prompt and navigate to the `SCHISM/garvis` folder, and run `garvis.py`.
3. Answer all the questions in lay terms and plain English (and don't forget [your OpenAI API key](https://platform.openai.com/api-keys)).
4. Hit the "Generate INI" button.
5. Double-check the generated `.ini` file on the right-hand panel.


## 🖼️ Preview
Here is an example of GARVIS running in Streamlit

<img width="488" height="777" alt="image" src="https://github.com/user-attachments/assets/1eb132eb-3b91-474c-a67d-aa2433736b81"/>

## :brain: Knowledge base

Navigate to `cd <some path> SCHISM/garvis/knowledge_base` to access this part.

The knowledge base provides users with a centralized reference of configurations related to the image processing workflow and deep learning training. A broad set of examples is included, covering many possible scenarios and helping users get started faster with their own use cases.

On the firlst line of GARVIS, there is a line asking you to enter the path of `Project repository`. You have to enter here the path were there is your knowledge base.

There is also a tool allowing you to generate other random ini files. You can launch the program from terminal with `python generation_multiple.py`. You'll have to choose a path to save new files, then you'll have to run 

