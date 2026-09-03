# Fairness Project

This repository contains the materials for **Project 1** of the course **02517 Responsible AI: Algorithmic Fairness and Explainability (Fall 2026)**. 

The project is titled **Auditing & Mitigating Bias**.

Over the course of three weeks, students will evaluate, audit, and mitigate algorithmic bias using the ACSIncome dataset. The project progresses from a baseline fairness audit (Week 1), to a blind data audit of noisy labels (Week 2), and concludes with reference-label evaluation and mitigation strategies (Week 3).

## Final Submission

The final submission is due at the end of **Week 3**. 
You are required to submit a **poster** summarizing your findings across the three weeks. 
- Please strictly use an A0 poster size.
- The detailed structure and requirements for the final poster can be found at the bottom of the `Week3_Mitigation_Reference_Evaluation.ipynb` notebook.

## Local Environment Setup

If you prefer to run the project locally on your machine instead of using Colab, we have provided a setup script to configure a virtual environment, install the required packages, and register the Jupyter kernel for you.

**Setup Instructions:**
1. Clone this repository to your local machine.
2. Open a terminal and navigate to the root of the repository.
3. Run the setup script: (make sure to rename setup.txt -> setup.sh)
   ```bash
   ./setup.sh
   ```
4. Once completed, activate the environment (if you plan to run python scripts from the terminal):
   ```bash
   source .venv/bin/activate
   ```
5. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
6. Open a notebook and ensure the kernel **RAI (Python 3)** is selected in the top right corner.


## Incase you are using Colab

To avoid local environment setup issues, we recommend using Google Colab to run the Jupyter Notebooks.

**Setup Instructions:**
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File > Open notebook** and choose the **GitHub** tab.
3. Paste the URL of this repository and open the relevant Week's notebook.
4. **Data Access**: Make sure to upload the necessary CSV files from the `data/` directory of this repository into your Colab environment's `/content/data` folder before running the notebook. You can do this by clicking the "Folder" icon on the left sidebar in Colab and dragging the files in.
5. Alternatively, you can clone the repository directly inside a Colab cell:
   ```python
   !git clone https://github.com/ADE-17/RAI_Project1_Fairness_Project.git
   ```
   and update the `DATA_DIR` path in the notebooks accordingly.

---

**Created and maintained by:**
Aditya Parikh, PhD Student at DTU Compute, Researcher in Responsible AI  
TA for this course. For any doubts or inquiries regarding this repository, please contact me.
