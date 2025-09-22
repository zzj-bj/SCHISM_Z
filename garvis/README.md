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

## :question: How to use?

GARVIS offer a web interface to generate an INI file.

### General steps
1. Recover an OpenAI key to run the program. :warning: Be careful, as we are using a custom URL: Don't forget to adapt the code to your purpose to run the app. To adjust the API's URL, please check the ```openai_as_requests``` function from the ```processing.py``` file, and assign the right URL to the ```url``` variable.
2. Open a terminal prompt and navigate to the ```SCHISM/garvis``` folder, and run ```garvis.py```.
3. Answer all the questions in lay terms and plain English (and don't forget OpenAI API key!).
4. Click the "Generate INI" button.
5. And double-check the generated .ini file on the right-hand side.

