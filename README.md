Repository for "RECOVERING GOVERNING EQUATIONS OF PARTIAL DIFFERENTIAL EQUATIONS VIA FRIEDRICHS LEARNING" (in progress; see TTU_REU_Poster.pdf and TTU_REU_Written_Report.pdf) 

main.py: 
Training dicts are defined, type of $H$ and $G$ chosen, train functions called.

train_1d_uniform_fl.py and train_1d_uniform_ls.py:
Training defined with weak form and strong form.

loss.py:
Gradient and loss functions defined, with strong form, weak form, and various types of weak form. (in progress: testing various loss functions)

data.py:
Different $H$ and $G$ are defined, along with functions that compute $u$ numerically with finite differences. Then also computes $u_t$, $u_x$, $u_{xx}$ numerically with FDM, to be used in loss functions.

data_test.py: (moved to master_test_exact)
Specific example with constant $H$ and $G$, with an exact solution. Defines the same numerical approximations for $u$, $u_t$, $u_x$, and $u_{xx}$, along with exact functions of each.
**To use this instead of data.py, replace all data.py imports with data_test.py imports, inside train and generate files.**

generate.py:
Generation functions, including generating grids of size 1001 x 101 for numerical and exact solution data, generating simpson weights, and generating domains to get neural net outputs.

models.py:
Neural nets used to parameterize $v$, $H$, and $G$.
