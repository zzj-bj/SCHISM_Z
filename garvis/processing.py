from textwrap import dedent
import streamlit as st
import requests
import os
import re
from config import logger, EXTERNAL_URLS
from bs4 import BeautifulSoup
from utils_gen import estimate_call_cost

def openai_ask_requests(messages, model="gpt-4o", response_format=None):
 
	url = "https://api.openai.com/v1/chat/completions"
    #url = f"https://cld.akkodis.com/api/openai/deployments/models-{model}/chat/completions?api-version=2024-12-01-preview"
	headers = {
		"Content-Type": "application/json",
		"Cache-Control": "no-cache",
		"api-key": f"Bearer {st.session_state.get('API_key')}"
	}
 
	data = {
		"temperature": 0.01,
		"max_tokens": 10000,
		"messages": messages
	}
 
	if response_format is not None:
		data["response_format"] = response_format
	response = requests.post(url, headers=headers, json=data).json()
	return response['choices'][0]['message']['content']

def fetch_and_parse_docs(url):
      # Convert GitHub URL to raw URL if necessary.
      raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
      response = requests.get(raw_url)
      if response.status_code != 200:
          raise Exception(f"Error fetching {raw_url}: Status code {response.status_code}")

      md_text = response.text

      # Regex pattern to capture sections defined by lines starting with "## [SectionTitle]"
      # The pattern captures the title inside the brackets and all text until the next section or end of document.
      pattern = re.compile(r'^## \[(.+?)\]\s*(.*?)(?=^## \[|\Z)', re.MULTILINE | re.DOTALL)

      sections = {}
      for match in pattern.finditer(md_text):
          title = match.group(1).strip()
          content = match.group(2).strip()
          sections[title] = content

      return sections

def read_ini_files(root_folder, part):
    """
    Reads all INI files from the specified folder and concatenates only the [Model] and [results] sections
    from each file into a single knowledge base string.
    """
    knowledge_base = ""
    # Loop through sorted files in the folder
    for file in sorted(os.listdir(root_folder)):
        ini_file = os.path.join(root_folder, file)
        if file.endswith(".ini") and os.path.isfile(ini_file):
            try:
                with open(ini_file, 'r', encoding="utf-8") as f:
                    content = f.read()
                    extracted_sections = []
                    # Extract only the [Model] and [results] sections using regex
                    for section in [part, "results"]:
                        # This regex captures the section header and all text until the next header or end-of-file
                        pattern = rf"(?ms)^\[{section}\].+?(?=^\[|\Z)"
                        match = re.search(pattern, content)
                        if match:
                            extracted_sections.append(match.group(0).strip())
                    if extracted_sections:
                        knowledge_base += f"\n[FILE: {ini_file}]\n" +"\n"+ "\n".join(extracted_sections) + "\n"
            except Exception as e:
                logger.error(f"Error reading file {ini_file}: {e}")
    return knowledge_base.strip()

def fetch_github_docs(urls, external_urls=None):
    """Fetches and concatenates documentation from GitHub URLs (and optionally external URLs)."""
    combined_response = ""
    if isinstance(urls, str):
        urls = [urls]
    try:
        for url in urls:
            response = requests.get(url)
            if response.status_code == 200:
                combined_response += "\n" + response.text
            else:
                logger.warning(f"Failed to fetch GitHub docs from {url}, status code: {response.status_code}")
        if external_urls:
            for ext_url in external_urls:
                response = requests.get(ext_url)
                if response.status_code == 200:
                    combined_response += "\n" + response.text
                else:
                    logger.warning(f"Failed to fetch external docs from {ext_url}, status code: {response.status_code}")
        return combined_response.strip()
    except Exception as e:
        logger.error(f"Failed to fetch GitHub docs: {e}")
        raise RuntimeError(f"Failed to fetch GitHub docs: {e}")

def validate_ini_structure(ini_content):
    """Validates that the INI content contains exactly the required sections."""
    required_sections = {"[Model]", "[Optimizer]", "[Scheduler]", "[Loss]", "[Training]", "[Data]"}
    found_sections = set(re.findall(r"^\[([^\]]+)\]$", ini_content, re.MULTILINE))
    return found_sections == required_sections

def fetch_docs_by_category(category, available_opts, base_url_template):
    """
    For a given category (e.g., 'models'), build the URL using the base_url_template,
    fetch each doc from GitHub (converted to its raw URL), and return a dictionary
    where each key is the document name and its value is the content.
    """
    docs = {}
    for doc_name in available_opts.get(category, []):
        url = base_url_template.format(doc_name)
        # Convert GitHub URL to its raw version
        #raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        response = requests.get(url)
        if response.status_code == 200:
            docs[doc_name] = response.text
        else:
            docs[doc_name] = f"Error: Status code {response.status_code}"
    return docs

def fetch_and_clean_doc(url):
    """
    Fetches online documentation from the given URL and extracts only the Parameters section.
    It searches for the first <dt class="field-odd"> element containing "Parameters" and returns the text
    from the following <dd> element. If no such section exists, it returns an empty string.
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return f"Error: Status code {response.status_code} when fetching {url}"

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the <dt> element with class "field-odd" that contains "Parameters"
        dt = soup.find("dt", class_="field-odd", string=lambda t: t and "Parameters" in t)
        if dt:
            # Get the following <dd> which should contain the bullet points/details
            dd = dt.find_next_sibling("dd")
            if dd:
                # Remove any unwanted tags (script, style)
                for tag in dd(["script", "style"]):
                    tag.decompose()
                # Extract text, preserving newlines, and clean up whitespace
                raw_text = dd.get_text(separator="\n", strip=True)
                lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                cleaned_text = "\n".join(lines)
                return cleaned_text
        # If no Parameters section is found, return an empty string.
        return ""

    except Exception as e:
        return f"Error: Exception {e} while fetching {url}"

def get_online_doc(part, choice, external_urls):
    """Returns the URL for the given part and choice directly from the dictionary."""
    if part not in external_urls:
        return f"Part '{part}' not found in dictionary."

    if choice not in external_urls[part]:
        return f"Choice '{choice}' not found under part '{part}'."
    return external_urls[part][choice]

def query_llm_component_choice(prompt, valid_options):
    """
    Queries the OpenAI GPT-4o model to determine the best component choice(s).
    Returns a comma-separated list of valid options.
    """
    messages = [
		{
			"role": "system",
			"content": (
				"You are an AI assistant specialized in deep learning configuration.\n"
				"Your ONLY task is to output a comma-separated list of one or more words that exactly match one or more of these options: "
				+ ", ".join(valid_options)
				+ ". Do not abbreviate or alter the options. If you are unsure, output 'none'."
			)
		},
		{"role": "user", "content": prompt}
	  ]
    
    st.session_state['NbTokens'] += estimate_call_cost(messages, model="gpt-4o")
    print(st.session_state.get('NbTokens'))
    try:
        response_text = openai_ask_requests(messages)
        tokens = [token.strip() for token in response_text.split(",") if token.strip()]
        valid_tokens = [token for token in tokens if token in valid_options]
        return ", ".join(valid_tokens) if valid_tokens else "none"
    except Exception as e:
        logger.error(f"OpenAI query (component choice) failed: {e}")
        return "none"

def query_llm_section(prompt):
    """
    Queries the OpenAI GPT-4o model and returns the INI section content.
    """
    messages = [
		  {
			  "role": "system",
			  "content": (
				  "You are an AI assistant specialized in optimizing training configurations for deep learning models.\n"
				  "Generate an optimized INI configuration file based ONLY on the provided documentation, knowledge base, and user input.\n"
				  "Do NOT include any extra text or markdown formatting. Output only the INI file."
			  )
		  },
		  {"role": "user", "content": prompt}
	  ]
    st.session_state['NbTokens'] += estimate_call_cost(messages, model="gpt-4o")
    print(st.session_state.get('NbTokens'))
    try:
        return openai_ask_requests(messages).strip()
    except Exception as e:
        logger.error(f"OpenAI query (section) failed: {e}")
        raise RuntimeError(f"OpenAI query (section) failed: {e}")

#############################################
# COMPONENT QUERIES BASED ON USER INPUT
#############################################

def determine_component_choice_O_S_L(user_input, available_options, doc_section, knowledge_base, part):
    """
    Determines the best matching option for optimizer/scheduler/loss configurations.
    Fetches the corresponding online documentation and includes it in the prompt.
    """
    # Format the available options for the prompt
    formatted_options = "\n".join(f"- {opt}" for opt in available_options)

    prompt_model = dedent(f"""
    The user's general problematic is as follows:
    {user_input.get("problematic", "")}

    For the {part} configuration, here are the available options:
    {formatted_options}

    About the {part}, the user said:
    {user_input.get(part, "")}

    Given the knowledge base: the [{part}] parametrisation are the associated [results] (where a result close to 1 is good and 0 is bad):
    {knowledge_base}

    Finally, the available general documentation about the models is as follows:
    {doc_section}

    ### Task:
    Select the best matching option from the list based on the user input, following these rules:

    STRICT RULES:
    1. Exact Match Only: Choose an option only if it directly matches one from the list (case-insensitive, but preserve original casing).
    2. Output Format: Return a single word—the chosen option’s name. No extra text or explanations.
    3. Inference: If the user statement refers to a general concept without an explicit match, select the most pragmatic option.
    4. Uncertainty: If the user is unsure (e.g., 'I don't know', 'maybe', 'anything works'), return the most pragmatic option.
    5. Multiple Selections: If multiple options are suggested, choose the most pragmatic one.

    """).strip()
    #print(f"[DEBUG] Options disponibles pour {part} : {available_options}")
    choice = query_llm_component_choice(prompt_model, available_options)

    # Fetch the online documentation for the selected choice
    online_doc = get_online_doc(part, choice, EXTERNAL_URLS)

    user_scheduler_input = user_input.get(part, "")

    prompt_options = dedent(f"""
    Given the selected {part} which is {choice}

    The user explicitly said the following about the {part} configuration:
    "{user_scheduler_input}"

    Given the problematic of the user:
    {user_input.get("Problematic", "")}

    Given the data type of the user:
    {user_input.get("Data", "")}

    Given the knowledge base of experiments and associated [Results] (where a result close to 1 is good and 0 is bad):
    {knowledge_base}

    And given the online documentation concerning {choice}:
    {fetch_and_clean_doc(online_doc)}

    ### Task:
    Generate a valid INI file for the [{part}] section based on the above information.

    STRICT RULES:
    1. Output ONLY a valid INI file with the [{part}] section.
    2. Do NOT include extra text or markdown formatting.
    3. If the user explicitly mentions a parameter (e.g., factor = 0.5), you MUST use that exact value.
    4. Otherwise, infer missing parameters using best practices.
    5. Ensure logical consistency among parameters.
    6. Do NOT use any quotation marks (single or double); values must be strings or numbers.
    7. ALL numeric parameters must be formatted as decimal floats (e.g., 0.33 instead of 1/3).
    8. Fractions or math expressions (like 1/3 or 2*10e-3) are strictly forbidden in the output.
    """).strip()
    model_ini = query_llm_section(prompt_options)
    return model_ini.strip()

def determine_component_choice_training(user_input, available_options, doc_section, knowledge_base, part):
    """
    Sets the metrics, epochs, batch_size, and other training parameters.
    """
    user_metric_preference = user_input.get(part, "").strip()

    # === Step 1 : Direct regex matching for explicit metrics
    explicit_metrics = []
    for opt in available_options:
        if re.search(rf"\b{opt}\b", user_metric_preference, re.IGNORECASE):
            explicit_metrics.append(opt)

    if explicit_metrics:
        selected_metrics = ", ".join(explicit_metrics)
    elif any(k in user_metric_preference.lower() for k in ["all", "everything", "all metrics", "use them all"]):
        selected_metrics = ", ".join(available_options)
    else:
        # === Step 2 : LLM-based inference
        prompt_metrics = dedent(f"""
        The user is configuring a training setup for a deep learning model.

        Available metrics:
        {chr(10).join(f"- {opt}" for opt in available_options)}

        User's input for metrics:
        "{user_input.get(part, "")}"

        STRICT RULES:
        - If the input names one or more metrics, return only those (matching available options).
        - If it names none, return the full list.
        - Output must be comma-separated metrics, nothing else.
        """).strip()

        selected_metrics = query_llm_component_choice(prompt_metrics, available_options)

        # === Step 3 : final filtering to ensure validity
        # Keep only valid metrics from available_options
        filtered = []
        for metric in [m.strip() for m in selected_metrics.split(",")]:
            if metric in available_options:
                filtered.append(metric)
        selected_metrics = ", ".join(filtered) if filtered else ", ".join(available_options)

    # Epoch detection
    epoch_match = re.search(r"(\d+)\s*epoch", user_input.get("Training", ""), re.IGNORECASE)
    explicit_epochs = int(epoch_match.group(1)) if epoch_match else None
    epoch_instruction = f"The user explicitly requested {explicit_epochs} epochs." if explicit_epochs else "No specific epoch count was mentioned."

    # Detect of batch_size
    batch_match = re.search(r"batch\s*size\s*=?\s*(\d+)", user_input.get("Training", ""), re.IGNORECASE)
    explicit_batch_size = int(batch_match.group(1)) if batch_match else None
    batch_instruction = f"The user explicitly requested batch_size = {explicit_batch_size}." if explicit_batch_size else "No specific batch_size was mentioned."

    # Final prompt [Training]
    prompt_training_ini = dedent(f"""
    Using the following details, generate a valid [Training] section for an INI file. Base your configuration on the user input and best practices.

    Context:
      - Hardware and performance needs: {user_input.get("Problematic", "")}
      - Dataset details: {user_input.get("Data", "")}
      - Documentation reference: {doc_section}
      - Epoch instruction: {epoch_instruction}
      - Batch size instruction: {batch_instruction}

    Guidelines:
    1. Batch Size (batch_size):
        - If the user explicitly requested a batch size, use that value: {explicit_batch_size if explicit_batch_size else "none"}.
        - Otherwise, infer from dataset size and hardware (default = 16).
    2. Validation Split (val_split):
        - Always use val_split = 0.8 unless the user explicitly provides another value.
    3. Epochs:
        - Use the epoch value explicitly provided in the training input if present; otherwise, default to 10.
    4. Early Stopping (early_stopping):
        - If the epoch count is less than 15, set early_stopping = False.
        - If epochs is 15 or more (and not overridden by user input), set early_stopping = True.
    5. Metrics:
        - Use the following metrics exactly as determined: {selected_metrics}.

    Task:
      Return only the [Training] section in INI format.
    """).strip()

    return query_llm_section(prompt_training_ini).strip()


def determine_component_choice_loss(user_input, available_options, knowledge_base, part):
    """
    Generates a valid [Loss] section for an INI file.
    Uses the parameters provided to select an appropriate loss function and
    fill in additional keys. If the user's loss input is vague (e.g., "I don't know"),
    the loss function will be set to "Undetermined, please check the doc".

    The INI section must include exactly three keys:
      - loss
      - ignore_background
      - weights
    """
    # Format the available options for the prompt
    formatted_options = "\n".join(f"- {opt}" for opt in available_options)

    # Fetch the online documentation for each loss
    online_doc_cross = fetch_and_clean_doc(get_online_doc(part, "CrossEntropyLoss", EXTERNAL_URLS))
    online_doc_binary = fetch_and_clean_doc(get_online_doc(part, "BCEWithLogitsLoss", EXTERNAL_URLS))
    online_doc_nll = fetch_and_clean_doc(get_online_doc(part, "NLLLoss", EXTERNAL_URLS))

    prompt = dedent(f"""
    You are an AI assistant specialized in deep learning configuration.

    Your task is to generate a valid [Loss] section for an INI file based on the provided parameters.

    **User Input for {part}:**
    - User explicitly mentioned: **{user_input.get(part, "")}**
    - **Data Description:** {user_input.get("Data", "")}

    **Available Loss Functions:**
    {formatted_options}

    **Online Documentation for the Loss Functions:**
    - **CrossEntropyLoss**: {online_doc_cross}
    - **BCEWithLogitsLoss**: {online_doc_binary}
    - **NLLLoss**: {online_doc_nll}

       **Important Rules:**
    1. **Binary Segmentation (1 or 2 classes ONLY):**
       - If the data description mentions "binary", "binary masks", "one or two classes", "2 classes", or any similar phrase, you **MUST** select **BCEWithLogitsLoss**.
       - **CrossEntropyLoss is NEVER valid for binary segmentation. Do NOT use it.**
    2. **Multi-Class Segmentation (strictly 3+ classes):**
       - If the data contains multiple classes (strictly greater or equal to 3), you **MUST** use **CrossEntropyLoss**.
    3. **Negative Log-Likelihood Loss (NLLLoss):** Only use this if the user explicitly mentions logarithmic probability or if it aligns with the documentation.
    4. **User-Specified Choice:** If the user explicitly names a loss function (case insensitive), select that function.
    5. **Uncertainty Handling:** If the user input is vague or uncertain (e.g., "I don't know"), analyze the data description to infer the best option.
       - If the data description doesn't help, return: `"Undetermined, please check the doc"`.

    **Output Format:**
    - Return ONLY a valid INI section with exactly three keys loss, ignore_background, weights:
    - Do **not** modify the text inside `ignore_background` and `weights`.
    - Do **not** include explanations, markdown formatting, or extra text.
      ```
      [Loss]
      loss = (chosen_loss)
      The `ignore_background` and `weights` keys **must** be set to True"
      ```

    Generate the [{part}] section based on these rules.
    """).strip()

    return query_llm_section(prompt).strip()

def determine_data_parameters(user_input, part):
    """
    Generates a valid [Data] section for an INI file by inferring appropriate values for:
      - crop_size: Size of image crops (default: 224px)
      - img_res: Resolution to resize crops (default: 560px)
      - num_samples: Number of samples to use (default: 500)

    The inference is based on the user's input details (e.g., problematic description and data description).

    Return ONLY a valid [Data] section in INI format with exactly the keys: crop_size, img_res, num_samples.
    """
    sample_match = re.search(r"\b(\d{2,6})\b.*(images|samples|data|pictures)", user_input.get("Data", ""), re.IGNORECASE)
    explicit_samples = int(sample_match.group(1)) if sample_match else None

    sample_context = (
        f"The user has provided a dataset with approximately {explicit_samples} samples."
        if explicit_samples else
        "The user did not specify the number of samples."
    )

    prompt = dedent(f"""
    You are an AI assistant specialized in deep learning configuration.

    Your task is to generate a valid [{part}] section for an INI file based solely on the provided user details.

    User details:
      - Problematic: {user_input.get("Problematic", "")}
      - Data: {user_input.get("Data", "")}
      - {sample_context}

    Guidelines:
      1. crop_size: The size of image crops. The default value is 224px. Adjust this value if the performance requirements or dataset details imply that a different crop size would be better.
      2. img_res: The resolution for resizing image crops during training and inference. The default value is 560px. Only change this value if the user explicitly specifies a different resolution.
      3. num_samples: The number of samples to use. The default value is 500 samples. If the user indicates a dataset that is much smaller or larger, adjust this value accordingly.

    Output Format:
      - Return ONLY a valid [{part}] section in INI format with the keys: crop_size, img_res, and num_samples.
      - Do NOT include any explanations, markdown formatting, or extra text.
    """).strip()

    return query_llm_section(prompt).strip()

def determine_component_choice_model(user_input, available_options, model_doc, doc_section, knowledge_base, part):
    ###### MODEL CHOICE
    formatted_options = "\n".join(f"- {opt}" for opt in available_options)

    prompt_model = dedent(f"""
    The user's general problematic is as follows:
    {user_input.get("problematic", "")}

    For the {part} configuration, here are the available options:
    {formatted_options}

    About the {part}, the user said:
    {user_input.get(part, "")}

    Finally, the available general documentation about the models is as follows:
    {doc_section}

    ### Task:
    Select the best matching option from the list based on the user input, following these rules:

    STRICT RULES:
    1. **Strict Containment Rule:** If the user mentions a model name that appears in the available options (case-insensitive, ignoring spaces, dashes or underscores), you MUST select that exact option. This rule OVERRIDES all other rules. Never replace this by a default or pragmatic choice.
    2. **Exact Match Priority:** If multiple options partially match, prefer the longest and most specific option (e.g., "UnetSegmentor" over "Unet").
    3. **Fallback:** Only if no option matches the user text, infer the most pragmatic choice from the list. 
    4. **Output Format:** Return a single word—the chosen option’s name. No extra text or explanations. The answer must be part of the available options.
    5. **Uncertainty Handling:** If the user is unsure (e.g., 'I don't know', 'maybe', 'anything works'), return the most pragmatic option from the list.
    6. **Never answer "none", "any", or something outside the list.
    """).strip()

    choice = query_llm_component_choice(prompt_model, available_options)

    # ---- EXTRACTION des hyperparams explicites depuis le texte utilisateur (sans post-traitement)
    user_model_text = user_input.get(part, "")

    pinned = []  # liste de tuples (key, value_str) a injecter dans le prompt

    # n_block: "n_block=8" OU "8 n_block"
    m = re.search(r"(?:n[_\s-]?block\s*=\s*(\d+)|(\d+)\s*n[_\s-]?block)", user_model_text, re.IGNORECASE)
    if m:
        val = m.group(1) or m.group(2)
        pinned.append(("n_block", val))

    # channels: "channels=64" OU "64 channels"
    m = re.search(r"(?:channels?\s*=\s*(\d+)|(\d+)\s*channels?)", user_model_text, re.IGNORECASE)
    if m:
        val = m.group(1) or m.group(2)
        pinned.append(("channels", val))

    # k_size: "k_size=3" OU "kernel size 3" OU "k size 3"
    m = re.search(r"(?:k[_\s-]?size\s*=\s*(\d+)|kernel\s*size\s*(\d+)|k\s*size\s*(\d+))", user_model_text, re.IGNORECASE)
    if m:
        val = next(g for g in m.groups() if g)
        pinned.append(("k_size", val))

    # activation: "activation=relu" OU "activation relu" OU "use relu"
    m = re.search(r"(?:activation\s*=\s*([A-Za-z0-9_-]+)|activation\s+([A-Za-z0-9_-]+)|use\s+([A-Za-z0-9_-]+)\s*activation)",
                  user_model_text, re.IGNORECASE)
    if m:
        val = next(g for g in m.groups() if g)
        pinned.append(("activation", val))

    pinned_lines = "\n".join(f"- {k} = {v}" for k, v in pinned) if pinned else "None"

    ###### Récupération des hyperparams spécifiques au modèle choisi
    allowed_params = model_doc.get(choice, [])
    allowed_params_str = "\n".join(f"- {p}" for p in allowed_params) if allowed_params else "None"

    ###### MODEL'S OPTIONS
    prompt_options = dedent(f"""
    Given the selected {part} which is {choice}

    User's raw request for the [{part}] section (parse explicit hyperparameters from this text):
    {user_model_text}

    Pinned hyperparameters (MUST be used verbatim; they override defaults and documentation):
    {pinned_lines}

    Allowed hyperparameters for this model (you MUST include all of these, unless explicitly pinned otherwise):
    {allowed_params_str}

    Given this documentation including options and parameters to set:
    {model_doc.get(choice, "")}

    Given the problematic of the user which is:
    {user_input.get("Problematic", "")}

    Given the data type of the user:
    {user_input.get("Data", "")}

    Given the knowledge base of experiment and associated [Results], and understanding that a result close to 1 is considered good, and 0 is considered bad:
    {knowledge_base}

        ### Task:
    Select the best matching option from the list based on the user input and the [{part}] section.

    STRICT RULES:
    1. Output ONLY a valid INI file with these sections: [{part}].
    2. Do NOT include extra text or markdown formatting.
    3. You MUST include ALL allowed parameters listed above for the chosen model, even if the user did not specify them.
    4. Infer missing parameters using best practices.
    5. Ensure logical consistency among parameters.
    6. Never use any quotation marks, either simple or double. Answers must be either string type or numerical type.
    7. Do not infer the 'num_classes'. Here you should always state exactly: num_classes = insert your number of classes here
    8. If the user explicitly specifies a hyperparameter value in the text above or in the pinned list, you MUST keep it exactly as given.
    9. Rule precedence: Rule 8 overrides ALL other rules (including defaults from documentation or knowledge base).
    10. Hardware and dataset constraint:
        - If VRAM <= 12GB and dataset size <= 5000 samples, then channels must not exceed 8 unless explicitly given by the user.
        - If n_block >= 7, favor smaller values (8 channels or less).
    """).strip()

    model_ini = query_llm_section(prompt_options)
    return model_ini.strip()

def determine_data_augmentation_section(user_input, part="Data_augmentation"):
    """
    Generates a valid [Data_augmentation] section for an INI file by inferring appropriate values for:
      - brightness: image brightness adjustment (float, e.g. 0.2)
      - angle: rotation angle in degrees (int, e.g. 15)
      - translate: pixel shift range for translation (list of 2 ints)
      - scale: image zoom factor (float, e.g. 1.0)
      - shear: shear intensity in degrees (list of 2 ints)

    This section is optional. If the user explicitly says data augmentation is not needed, return an empty string.

    Return ONLY a valid [Data_augmentation] section in INI format with exactly the keys above, unless user opted out.
    """
    da_input = user_input.get(part, "").lower().strip()

    # If the user explicitly says no augmentation is needed
    if any(x in da_input for x in ["not necessary", "no", "skip", "none", "don't need"]):
        return ""

    prompt = dedent(f"""
    You are a deep learning assistant helping to configure data augmentation for a model training pipeline.

    User context:
      - Problematic: {user_input.get("Problematic", "")}
      - Data description: {user_input.get("Data", "")}
      - Augmentation preference: "{user_input.get(part, "")}"

    Guidelines:
      1. If the user gives vague instructions (e.g., "basic" or "standard"), return default augmentation values.
      2. If the user gives specific needs (e.g., “strong augmentation”), adjust accordingly while remaining safe.
      3. Only include the following parameters: brightness, angle, translate, scale, shear.
      4. Use these default values if the user is vague:
         - brightness = 0.2
         - angle = 15
         - translate = [10, 10]
         - scale = 1.0
         - shear = [10, 10]

    Output Format:
      - Return ONLY a valid [{part}] section in INI format.
      - Do NOT include markdown formatting, explanations or comments.
    """).strip()

    return query_llm_section(prompt).strip()
