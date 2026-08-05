# ⚠️ Action Required: Model Weights Download

Due to GitHub's 25MB file upload limit, the `final_model.pt` file (28 MB) could not be uploaded directly to this folder. Instead, it is hosted in the **GitHub Releases** section of this repository.

## Instructions for the Evaluator

To properly evaluate the model and maintain the required file structure, please follow these steps:

1. Navigate to the **Releases** section on the right-hand side of the main repository page, or go directly to: 
   [https://github.com/Pathan-Mohammad-Rashid/Topic_Classification/releases](https://github.com/Pathan-Mohammad-Rashid/Topic_Classification/releases)
2. Under the latest release (`v1.0`), download the `final_model.pt` file from the **Assets** dropdown.
3. Move the downloaded `final_model.pt` file into this exact folder alongside this markdown file.

### Required File Structure Check
Before running inference, ensure your directory looks exactly like this:

```text
Topic_Classification/
├── src/
├── final_models/
│   ├── final_model.md    <-- (You are reading this)
│   └── final_model.pt    <-- (Place the downloaded file here)
├── requirements.txt
└── README.md
