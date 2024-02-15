import papermill as pm

input_notebook = "/home/gperri-ext/master_thesis_tryout-2/notebooks_try/prova_papermill.ipynb"
output_notebook = "/home/gperri-ext/master_thesis_tryout-2/notebooks_try/prova_papermill.ipynb"

pm.execute_notebook(input_notebook, output_notebook)
