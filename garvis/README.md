[GARVIS] stands for _Generative Assistant for Reliable Variables and INI Setup_. This application automatically generates configuration files (.ini) for training segmentation/classification models, by combining:
- Dynamic retrieval of available parameters from the official documentation (.md files)
- Interpretation of the user’s natural language prompt (e.g., “I want 20 epochs and metrics: Jaccard & Recall”)
- Automatic generation of a coherent configuration file based on best practices (batch size, early stopping, etc.)
---
## :gear: Installation
1. Clone this repository to your locale machine:
``` git clone git@github.com:FloFive/SCHISM.git ```
2. Navigate to the cloned directory inside a terminal prompt
3. Install the library Streamlit :
    pip install streamlit

## :question: How to use ?

GARVIS offer a web interface to generate some INI file in case you don't master the parameters.

###General steps
1. Recover a OpenAI key to run the program. Be careful, we are also using a custom URL, so don't forget to adapt the code at your purpose to run the app.
2. Open a terminal prompt inside the repository SCHISM, inside hyperparameterGen folder, and run "streamlit run garvis.py".
3. Fill all the fields (don't forget OpenAI API key).
4. Click on "Generate INI" to launch the magic. 
5. Check the result on the right side.
