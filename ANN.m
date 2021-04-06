load('cleveland_heart_disease_dataset_labelled.mat');
x = transpose(x);
t = transpose(t);
%To avoid the randomness for every re-run of program, 
%The random seed is set to reproduce the same results every time.
setdemorandstream(672880951)
